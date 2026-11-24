// Parity check: the in-browser engine vs the PyTorch/Opacus backend.
//   npx tsx scripts/parity.ts   (from web/)
// Compares (1) every test-set prediction of every served model with the
// PyTorch scores, (2) /api/predict, /api/accountant, /api/mitigation and
// /api/sweep outputs with references written by server/export_static.py.
import { readFileSync } from "node:fs";
import { Engine, forward, type StaticData } from "../src/lib/engine/engine";

const S = (n: string) => JSON.parse(readFileSync(`${process.cwd()}/public/static/${n}`, "utf8"));
const d: StaticData = { engine: S("engine.json"), models: S("models.json"), scores: S("scores.json"), bundle: S("bundle.json") };
const ref = S("parity.json");
const E = new Engine(d);
let fails = 0;
const report = (name: string, worst: number, tol: number, extra = "") => {
  const ok = worst <= tol; if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(44)} max |diff| = ${worst.toExponential(2)} (tol ${tol})${extra}`);
};

// numeric deep diff; booleans/strings must be equal, nulls must match
function diff(a: unknown, b: unknown, path = "", out: { worst: number; where: string; mismatch: string[] } = { worst: 0, where: "", mismatch: [] }) {
  if (typeof a === "number" && typeof b === "number") { const x = Math.abs(a - b); if (x > out.worst) { out.worst = x; out.where = path; } }
  else if (a && b && typeof a === "object" && typeof b === "object") {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) diff((a as never)[k], (b as never)[k], `${path}.${k}`, out);
  } else if (a !== b && !(a == null && b == null)) out.mismatch.push(`${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`);
  return out;
}

// 1. test-set scores: browser forward pass on the encoded test rows is not
//    available (raw rows are not exported), so check the served model outputs
//    through predict() references and the MLP on the lite test rows below.
{
  const f = `${process.cwd()}/../server/artifacts/parity_test_X.bin`;
  let X: Float32Array | null = null;
  try { const b = readFileSync(f); X = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); } catch { /* not exported */ }
  if (!X) console.log("SKIP  test-set scores (run server/export_static.py first)");
  else {
    const dim = d.engine.feature_names.length, n = d.scores.test.y.length;
    let worst = 0, flips = 0, total = 0;
    for (const [k, layers] of Object.entries(d.models)) {
      const ref = d.scores.probs[k].test;
      for (let i = 0; i < n; i++) {
        const p = forward(layers, X.subarray(i * dim, (i + 1) * dim)).prob;
        worst = Math.max(worst, Math.abs(p - ref[i]));
        if ((p >= d.engine.threshold) !== (ref[i] >= d.engine.threshold)) flips++;
        total++;
      }
    }
    report(`test-set scores (${Object.keys(d.models).length} models x ${n} people)`, worst, 1e-5, `  decision flips ${flips}/${total}`);
    if (flips) fails++;
  }
}
// 2. predict
{
  let worst = 0; const mm: string[] = []; let where = "";
  for (const [k, r] of Object.entries(ref.predict as Record<string, { models: unknown }>)) {
    const [label, eps] = k.split("|");
    const s = (d.bundle.meta.samples as { label: string; applicant: Record<string, string | number> }[]).find((x) => x.label === label)!;
    const o = diff(E.predict(s.applicant, Number(eps)), r);
    if (o.worst > worst) { worst = o.worst; where = `${k}${o.where}`; }
    mm.push(...o.mismatch);
  }
  report("predict (prob, explain, recourse, counterfactual)", worst, 2e-4, worst > 0 ? `  at ${where}` : "");
  if (mm.length) { fails++; console.log("  mismatches:", mm.slice(0, 8)); }
}
// 3. accountant
{
  let worst = 0, rel = 0; const mm: string[] = [];
  for (const [k, r] of Object.entries(ref.accountant)) {
    const [s, b, ep] = k.split("|").map(Number);
    const o = diff(E.accountant(s, b, ep), r);
    worst = Math.max(worst, o.worst); mm.push(...o.mismatch);
    rel = Math.max(rel, Math.abs((E.accountant(s, b, ep) as { eps: number }).eps / (r as { eps: number }).eps - 1));
  }
  report("accountant (eps, curve, sigma for target eps)", worst, 1e-4, `  rel eps err ${rel.toExponential(1)}`);
  if (mm.length) { fails++; console.log("  mismatches:", mm.slice(0, 8)); }
}
// 4. mitigation
{
  let worst = 0, where = ""; const mm: string[] = [];
  const k = `${ref.ours}__${Number(d.bundle.meta.epsilons[0])}`;
  for (const [key, r] of Object.entries(ref.mitigation)) {
    const [pol, attr, t] = key.split("|");
    const o = diff(E.mitigation(k, pol, attr, Number(t)), r);
    if (o.worst > worst) { worst = o.worst; where = key + o.where; }
    mm.push(...o.mismatch.map((m) => key + m));
  }
  report("mitigation (3 policies x 3 attributes x 2 t)", worst, 2e-3, worst > 0 ? `  at ${where}` : "");
  if (mm.length) { fails++; console.log("  mismatches:", mm.slice(0, 8)); }
}
// 5. sweep
{
  const o = diff(E.sweep(`${ref.ours}__${Number(d.bundle.meta.epsilons[0])}`, "race"), ref.sweep);
  report("threshold sweep (race)", o.worst, 1e-4);
  if (o.mismatch.length) { fails++; console.log("  mismatches:", o.mismatch.slice(0, 8)); }
}
// 6. evaluation summary recomputed from exported scores == evaluation.json
{
  let worst = 0, where = "";
  for (const [k, ev] of Object.entries(d.bundle.evaluation as Record<string, { summary: Record<string, number> }>)) {
    const r = E.mitigation(k, "single", "sex", d.engine.threshold) as { summary: Record<string, number> };
    for (const m of ["accuracy", "balanced_accuracy", "roc_auc", "intersection_worst_tpr", "intersection_tpr_gap", "sex_selection_rate_gap"]) {
      const x = Math.abs(r.summary[m] - ev.summary[m]);
      if (x > worst) { worst = x; where = `${k}.${m}`; }
    }
  }
  report("all 11 models: metrics vs evaluation.json", worst, 2e-4, worst > 0 ? `  at ${where}` : "");
}
void forward;
console.log(fails ? `\n${fails} check(s) failed` : "\nall parity checks passed");
process.exit(fails ? 1 : 0);
