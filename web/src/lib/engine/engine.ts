// In-browser twin of the FastAPI backend (server/app.py + server/core.py).
// Uses the exported weights and preprocessing of the SAME trained models, so
// predictions match PyTorch (checked by web/scripts/parity.mts). Nothing here
// changes a served model; live training runs on a separate, temporary model.

import { calibrateSigma, defaultDelta, epsilonFor, stepsFor } from "./rdp";
import { applyThresholds, clean, fairnessSummary, fitGroupThresholds, groupFairness, quantile, type Groups, type Limits } from "./metrics";

export type Applicant = Record<string, string | number>;
export interface Layer { in: number; out: number; W: number[]; b: number[] }
export interface EngineSpec {
  numeric: { name: string; median: number; mean: number; scale: number }[];
  categorical: { name: string; mode: string; categories: string[] }[];
  feature_names: string[]; feature_field: (string | null)[];
  hidden_defaults: Record<string, number>; education_num: Record<string, number>; education_order: string[];
  groups: { attrs: string[]; age_edges: number[]; age_labels: string[]; race_keep: string[]; race_other: string };
  threshold: number; recourse: { education: { cost_per_level: number; max_levels: number }; "hours-per-week": { step: number; cost_per_step: number; max: number } };
  four_fifths: number; min_support: number; min_positives: number;
}
interface Split { y: number[]; groups: Record<string, Groups> }
export interface Scores { val: Split; test: Split; probs: Record<string, { val: number[]; test: number[] }> }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Bundle = Record<string, any>;
export interface StaticData { engine: EngineSpec; models: Record<string, Layer[]>; scores: Scores; bundle: Bundle }

export class HttpError extends Error { constructor(public status: number, msg: string) { super(msg); } }

export const modelKey = (m: string, eps: number | null) => `${m}__${eps == null ? "inf" : String(Number(eps))}`;

// ------------------------------------------------------------------ MLP
export function forward(layers: Layer[], x: Float64Array | Float32Array, keep = false) {
  let h: Float64Array = Float64Array.from(x);
  const acts: Float64Array[] = [h];
  layers.forEach((L, li) => {
    const z = new Float64Array(L.out);
    for (let o = 0; o < L.out; o++) {
      let s = L.b[o];
      const row = o * L.in;
      for (let i = 0; i < L.in; i++) s += L.W[row + i] * h[i];
      z[o] = li < layers.length - 1 ? (s > 0 ? s : 0) : s;
    }
    h = z;
    if (keep) acts.push(z);
  });
  const logit = h[0];
  return { prob: 1 / (1 + Math.exp(-logit)), acts };
}

/** d sigmoid(f(x)) / dx  (ReLU gradient at 0 is 0, as in PyTorch). */
function inputGradient(layers: Layer[], x: Float64Array): { prob: number; grad: Float64Array } {
  const { prob, acts } = forward(layers, x, true);
  let g = new Float64Array([prob * (1 - prob)]);
  for (let li = layers.length - 1; li >= 0; li--) {
    const L = layers[li], gin = new Float64Array(L.in);
    for (let o = 0; o < L.out; o++) {
      const go = g[o];
      if (go === 0) continue;
      const row = o * L.in;
      for (let i = 0; i < L.in; i++) gin[i] += L.W[row + i] * go;
    }
    if (li > 0) { const a = acts[li]; for (let i = 0; i < L.in; i++) if (!(a[i] > 0)) gin[i] = 0; }
    g = gin;
  }
  return { prob, grad: g };
}

// ------------------------------------------------------------- engine
export class Engine {
  readonly L: Limits;
  readonly meta: Bundle;
  constructor(readonly d: StaticData) {
    this.L = { minSupport: d.engine.min_support, minPositives: d.engine.min_positives, fourFifths: d.engine.four_fifths };
    this.meta = d.bundle.meta;
  }

  // -------------------------------------------------------- preprocessing
  frame(a: Applicant): Record<string, number | string | null> {
    const e = this.d.engine;
    const r: Record<string, number | string | null> = { ...e.hidden_defaults };
    for (const f of this.meta.schema.fields as string[]) r[f] = a[f];
    const vals = Object.values(e.education_num).sort((x, y) => x - y);
    const med = vals.length % 2 ? vals[(vals.length - 1) / 2] : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2;
    r["education-num"] = e.education_num[String(a.education)] ?? med;
    for (const f of Object.keys(this.meta.schema.numeric)) r[f] = Number(r[f]);
    return r;
  }

  encode(a: Applicant): Float64Array {
    const e = this.d.engine, r = this.frame(a);
    const x = new Float64Array(e.feature_names.length);
    let k = 0;
    for (const c of e.numeric) {
      const v = r[c.name];
      const num = v == null || Number.isNaN(Number(v)) ? c.median : Number(v);
      x[k++] = (num - c.mean) / c.scale;
    }
    for (const c of e.categorical) {
      const v = r[c.name] == null ? c.mode : String(r[c.name]);
      for (const cat of c.categories) x[k++] = cat === v ? 1 : 0;
    }
    return x;
  }

  score(k: string, a: Applicant): number { return forward(this.d.models[k], this.encode(a)).prob; }

  validate(a: Applicant): string[] {
    const sc = this.meta.schema, errs: string[] = [];
    for (const f of sc.fields as string[]) {
      if (!(f in a)) { errs.push(`missing field '${f}'`); continue; }
      const v = a[f];
      if (f in sc.numeric) {
        const x = typeof v === "number" ? v : v === "" ? NaN : Number(v);
        if (Number.isNaN(x) && typeof v !== "number") { errs.push(`'${f}' must be a number`); continue; }
        const { min, max } = sc.numeric[f];
        if (!Number.isFinite(x) || x < min || x > max) errs.push(`'${f}' must be between ${min} and ${max}`);
      } else if (!(sc.categorical[f] as string[]).includes(String(v))) errs.push(`'${f}' has unknown value '${v}'`);
    }
    return errs;
  }

  groupOf(a: Applicant) {
    const g = this.d.engine.groups;
    const sex = String(a.sex).trim(), r = String(a.race).trim();
    const race = g.race_keep.includes(r) ? r : g.race_other;
    const age = Number(a.age);
    const edges = [0, ...g.age_edges, Infinity];
    let ageL = "nan";
    for (let i = 0; i < g.age_labels.length; i++) if (age > edges[i] && age <= edges[i + 1]) ageL = g.age_labels[i];
    const parts: Record<string, string> = { sex, race, age: ageL };
    return { ...parts, intersection: g.attrs.map((x) => parts[x]).join("|") };
  }

  explain(k: string, a: Applicant) {
    const x = this.encode(a), { grad } = inputGradient(this.d.models[k], x);
    const agg = new Map<string, number>();
    this.d.engine.feature_field.forEach((f, i) => { if (f) agg.set(f, (agg.get(f) ?? 0) + grad[i] * x[i]); });
    return [...agg.entries()].sort((p, q) => Math.abs(q[1]) - Math.abs(p[1])).slice(0, 6)
      .map(([field, v]) => ({ field, value: Math.round(v * 1e4) / 1e4 }));
  }

  recourse(k: string, a: Applicant) {
    const rc = this.d.engine.recourse, order = this.d.engine.education_order, T = this.d.engine.threshold;
    const cur = order.indexOf(String(a.education));
    if (cur < 0) return { found: false, searched: 0 };
    const hp = rc["hours-per-week"], h0 = Number(a["hours-per-week"]);
    const maxDe = Math.min(rc.education.max_levels, order.length - 1 - cur);
    const nSteps = Number.isFinite(h0) ? Math.trunc(Math.floor(Math.max(0, hp.max - h0) / hp.step)) : 0;
    const hours = Array.from({ length: Math.min(nSteps, 200) + 1 }, (_, j) => h0 + j * hp.step);
    const layers = this.d.models[k];
    let best = -1, bestCost = Infinity, bestP = -Infinity, n = 0, bestChange: Record<string, string | number> = {};
    for (let de = 0; de <= maxDe; de++) {
      for (const h of hours) {
        const p = forward(layers, this.encode({ ...a, education: order[cur + de], "hours-per-week": h })).prob;
        if (n === 0 && p >= T) return null;
        const cost = de * rc.education.cost_per_level + ((h - h0) / hp.step) * hp.cost_per_step;
        if (p >= T && (cost < bestCost || (cost === bestCost && -p < -bestP))) {
          best = n; bestCost = cost; bestP = p;
          bestChange = {};
          if (de) bestChange.education = order[cur + de];
          if (h !== h0) bestChange["hours-per-week"] = h;
        }
        n++;
      }
    }
    if (best < 0) return { found: false, searched: n };
    return { found: true, cost: Math.round(bestCost * 100) / 100, prob: Math.round(bestP * 1e4) / 1e4, changes: bestChange, searched: n };
  }

  // ------------------------------------------------------------ endpoints
  predict(a: Applicant, eps: number) {
    if (!(this.meta.epsilons as number[]).includes(eps)) throw new HttpError(400, `eps must be one of ${this.meta.epsilons}`);
    const errs = this.validate(a);
    if (errs.length) throw new HttpError(422, errs.join("; "));
    const T = this.d.engine.threshold, ref = { ...a, ...this.meta.reference_identity };
    const served = (this.meta.methods as { id: string; name: string; private: boolean }[])
      .map((m) => ({ m, e: m.private ? eps : null })).filter(({ m, e }) => modelKey(m.id, e) in this.d.models);
    const models = served.map(({ m, e }) => {
      const k = modelKey(m.id, e), p = this.score(k, a);
      return { method: m.id, name: m.name, private: e != null, eps: e, prob: Math.round(p * 1e4) / 1e4, decision: p >= T ? 1 : 0,
        explain: this.explain(k, a), recourse: this.recourse(k, a), recourse_reference: this.recourse(k, ref) };
    });
    const cat = this.meta.schema.categorical;
    const combos = (cat.sex as string[]).flatMap((s) => (cat.race as string[]).map((r) => ({ ...a, sex: s, race: r })));
    const counterfactual = served.map(({ m, e }) => ({ method: m.id, name: m.name,
      cells: combos.map((c) => ({ sex: c.sex, race: c.race, prob: Math.round(this.score(modelKey(m.id, e), c) * 1e4) / 1e4 })) }));
    return { group: this.groupOf(a), eps, threshold: T, reference_identity: this.meta.reference_identity, models, counterfactual };
  }

  private probs(k: string) {
    const p = this.d.scores.probs[k];
    if (!p) throw new HttpError(404, "model not found; run train_models.py");
    return p;
  }

  thresholdsFor(policy: string, attr: string, pVal: number[], t: number): Record<string, number> {
    const v = this.d.scores.val, g = v.groups[attr];
    if (policy === "single") return { __default__: t };
    if (policy === "equal_opportunity") return fitGroupThresholds(pVal, v.y, g, t, this.L);
    if (policy === "demographic_parity") {
      const target = pVal.filter((p) => p >= t).length / pVal.length;
      const th: Record<string, number> = { __default__: t };
      g.labels.forEach((label, code) => {
        const s = pVal.filter((_, i) => g.codes[i] === code);
        if (s.length >= this.L.minSupport) th[label] = quantile(s, 1 - target);
      });
      return th;
    }
    throw new HttpError(400, "policy must be one of ('single', 'equal_opportunity', 'demographic_parity')");
  }

  mitigation(k: string, policy: string, attr: string, t: number) {
    const pr = this.probs(k), te = this.d.scores.test, attrs = this.d.engine.groups.attrs;
    const th = this.thresholdsFor(policy, attr, pr.val, t);
    const yhat = applyThresholds(pr.test, te.groups[attr], th);
    const summary = fairnessSummary(te.y, yhat, te.groups, attrs, this.L, pr.test);
    const ethics = Object.fromEntries([...attrs, "intersection"].map((a) => [a, groupFairness(te.y, yhat, pr.test, te.groups[a], this.L)]));
    return clean({ policy, attribute: attr, threshold: t, thresholds: th, summary, ethics });
  }

  private sweepCache = new Map<string, unknown>();
  sweep(k: string, attr: string) {
    const ck = `${k}|${attr}`;
    if (this.sweepCache.has(ck)) return this.sweepCache.get(ck);
    const pr = this.probs(k), te = this.d.scores.test, attrs = this.d.engine.groups.attrs, g = this.meta.threshold_grid;
    const out = [];
    const n = Math.floor((g.max + 1e-9 - g.min) / g.step + 1e-9);
    for (let i = 0; i <= n; i++) {
      const t = Math.round((g.min + i * g.step) * 1e4) / 1e4;
      const yhat = Int8Array.from(pr.test, (p) => (p >= t ? 1 : 0));
      const s = fairnessSummary(te.y, yhat, te.groups, attrs, this.L);
      const gf = groupFairness(te.y, yhat, pr.test, te.groups[attr], this.L);
      out.push({ t, accuracy: s.accuracy, balanced_accuracy: s.balanced_accuracy, worst_tpr: s.intersection_worst_tpr ?? null,
        dp_diff: gf.demographic_parity_diff, di_ratio: gf.disparate_impact_ratio, eo_diff: gf.equal_opportunity_diff });
    }
    const r = clean(out);
    this.sweepCache.set(ck, r);
    return r;
  }

  private accCache = new Map<string, unknown>();
  accountant(sigma: number, batch: number, epochs: number) {
    const ck = `${sigma}|${batch}|${epochs}`;
    if (this.accCache.has(ck)) return this.accCache.get(ck);
    const ui = this.meta.accountant_ui, n = this.meta.info.train_size as number;
    const q = batch / n, steps = stepsFor(epochs, q), delta = defaultDelta(n);
    const grid = [...new Set([1, Math.max(1, Math.floor(epochs / 2)), epochs, Math.min(ui.epochs.max, 2 * epochs), ui.epochs.max])].sort((x, y) => x - y);
    const r = clean({
      eps: epsilonFor(sigma, q, steps, delta), n, q, steps, delta, batch, epochs,
      curve: grid.map((ep) => [ep, epsilonFor(sigma, q, stepsFor(ep, q), delta)]),
      sigma_for: Object.fromEntries((this.meta.epsilons as number[]).map((t) => [Number(t).toFixed(1), calibrateSigma(t, delta, q, steps)])),
    });
    this.accCache.set(ck, r);
    return r;
  }
}
