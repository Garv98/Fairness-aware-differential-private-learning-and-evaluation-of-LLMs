// Smoke test for the in-browser ("lite") trainer: every method trains,
// spends exactly the requested epsilon, and learns something.
//   npx tsx scripts/train-smoke.ts   (from web/)
import { readFileSync } from "node:fs";
import { trainLite, type LiteSpec } from "../src/lib/engine/train";

const dir = `${process.cwd()}/public/static`;
const spec: LiteSpec = JSON.parse(readFileSync(`${dir}/lite.json`, "utf8"));
const eng = JSON.parse(readFileSync(`${dir}/engine.json`, "utf8"));
const b = readFileSync(`${dir}/lite.bin`), all = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
const data = { spec, X: all.subarray(0, spec.n_train * spec.dim), Xte: all.subarray(spec.n_train * spec.dim) };
const L = { minSupport: eng.min_support, minPositives: eng.min_positives, fourFifths: eng.four_fifths };
(async () => {
  let fails = 0;
  for (const m of Object.keys(spec.methods)) {
    const t0 = Date.now(), eps = 1, epochs = 5;
    const rows: { eps_spent?: number; accuracy: number; intersection_worst_tpr: number | null }[] = [];
    const r = await trainLite(data, m, eps, epochs, L, eng.threshold, 0, (row) => rows.push(row));
    const last = rows[rows.length - 1];
    const okEps = !spec.methods[m].private || (r.noise!.epsilon <= eps + 1e-9 && Math.abs(last.eps_spent! - r.noise!.epsilon) < 1e-4);
    const ok = rows.length === epochs && r.final.accuracy > 0.75 && okEps;
    if (!ok) fails++;
    console.log(`${ok ? "PASS" : "FAIL"} ${m.padEnd(15)} ${((Date.now() - t0) / 1000).toFixed(1)}s  acc ${r.final.accuracy.toFixed(3)}  worst-group recall ${r.final.intersection_worst_tpr}  ` +
      (spec.methods[m].private ? `sigma ${r.noise!.sigma_eff.toFixed(3)}  eps spent ${last.eps_spent}` : "non-private"));
  }
  process.exit(fails ? 1 : 0);
})();
