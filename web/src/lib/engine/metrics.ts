// Fairness metrics, ported from src/fairdp/metrics/fairness.py,
// src/fairdp/metrics/ethics.py and src/fairdp/postprocess.py.
// Groups are integer codes into a sorted label list (same order as pandas groupby).

export interface Groups { labels: string[]; codes: number[] }
export interface Limits { minSupport: number; minPositives: number; fourFifths: number }

type Num = number | null;

/** Python's round(x, 5) for JSON output (core.clean); NaN/inf -> null. */
export function clean<T>(o: T): T {
  if (Array.isArray(o)) return o.map(clean) as T;
  if (o && typeof o === "object") return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, clean(v)])) as T;
  if (typeof o === "number") return (Number.isFinite(o) ? (Number.isInteger(o) ? o : Math.round(o * 1e5) / 1e5) : null) as T;
  return o;
}

export function ece(y: ArrayLike<number>, p: ArrayLike<number>, nBins = 15): number {
  const sumP = new Float64Array(nBins), sumY = new Float64Array(nBins), cnt = new Float64Array(nBins);
  const step = 1 / nBins;
  for (let i = 0; i < p.length; i++) {
    // np.digitize(p, linspace(0,1,nBins+1)) - 1, clipped
    let b = 0;
    while (b <= nBins && b * step <= p[i]) b++;
    b = Math.min(Math.max(b - 1, 0), nBins - 1);
    sumP[b] += p[i]; sumY[b] += y[i]; cnt[b]++;
  }
  let e = 0;
  for (let b = 0; b < nBins; b++) if (cnt[b]) e += (cnt[b] / p.length) * Math.abs(sumP[b] / cnt[b] - sumY[b] / cnt[b]);
  return e;
}

/** ROC AUC with average ranks for ties (equals sklearn.roc_auc_score). */
export function rocAuc(y: ArrayLike<number>, s: ArrayLike<number>): number {
  const n = s.length, idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => s[a] - s[b]);
  let rankPos = 0, nPos = 0;
  for (let i = 0; i < n;) {
    let j = i;
    while (j + 1 < n && s[idx[j + 1]] === s[idx[i]]) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) if (y[idx[k]] === 1) { rankPos += r; nPos++; }
    i = j + 1;
  }
  const nNeg = n - nPos;
  return (rankPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

export function utility(y: ArrayLike<number>, yhat: ArrayLike<number>, prob?: ArrayLike<number>) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < y.length; i++) {
    if (y[i] === 1) { if (yhat[i] === 1) tp++; else fn++; } else { if (yhat[i] === 1) fp++; else tn++; }
  }
  const n = y.length, rec = tp + fn ? tp / (tp + fn) : 0, spec = tn + fp ? tn / (tn + fp) : 0;
  const prec = tp + fp ? tp / (tp + fp) : 0;
  const out: Record<string, number> = {
    accuracy: (tp + tn) / n, balanced_accuracy: (rec + spec) / 2, precision: prec, recall: rec,
    f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0, selection_rate: (tp + fp) / n,
  };
  if (prob && tp + fn > 0 && tn + fp > 0) { out.roc_auc = rocAuc(y, prob); out.ece = ece(y, prob); }
  return out;
}

interface Row { group: string; n: number; n_pos: number; tp: number; fp: number; tn: number; fn: number; sumS: number; idx: number[] }

function tabulate(y: ArrayLike<number>, yhat: ArrayLike<number>, g: Groups, prob?: ArrayLike<number>): Row[] {
  const rows: Row[] = g.labels.map((group) => ({ group, n: 0, n_pos: 0, tp: 0, fp: 0, tn: 0, fn: 0, sumS: 0, idx: [] }));
  for (let i = 0; i < y.length; i++) {
    const r = rows[g.codes[i]];
    r.n++; r.idx.push(i);
    if (prob) r.sumS += prob[i];
    if (y[i] === 1) { r.n_pos++; if (yhat[i] === 1) r.tp++; else r.fn++; } else { if (yhat[i] === 1) r.fp++; else r.tn++; }
  }
  return rows.filter((r) => r.n > 0);
}

/** fairness.summarize_groups for one attribute. */
function summarize(rows: Row[], prefix: string, L: Limits, out: Record<string, number>) {
  const elig = (needPos: boolean) => rows.filter((r) => r.n >= L.minSupport && (!needPos || r.n_pos >= L.minPositives));
  const metric: Record<string, (r: Row) => number> = {
    accuracy: (r) => (r.tp + r.tn) / r.n,
    tpr: (r) => (r.n_pos ? r.tp / r.n_pos : NaN),
    fpr: (r) => (r.n - r.n_pos ? r.fp / (r.n - r.n_pos) : NaN),
    selection_rate: (r) => (r.tp + r.fp) / r.n,
    f1: (r) => (2 * r.tp + r.fp + r.fn ? (2 * r.tp) / (2 * r.tp + r.fp + r.fn) : NaN),
  };
  for (const [m, needPos] of [["accuracy", false], ["tpr", true], ["fpr", false], ["selection_rate", false], ["f1", true]] as const) {
    const v = elig(needPos).map(metric[m]).filter((x) => !Number.isNaN(x));
    if (!v.length) continue;
    out[`${prefix}_${m}_gap`] = Math.max(...v) - Math.min(...v);
    if (m === "accuracy" || m === "tpr" || m === "f1") out[`${prefix}_worst_${m}`] = Math.min(...v);
  }
  const t = elig(true);
  if (t.length) {
    const tpr = t.map(metric.tpr), fpr = t.map(metric.fpr);
    out[`${prefix}_eodds_diff`] = Math.max(Math.max(...tpr) - Math.min(...tpr), Math.max(...fpr) - Math.min(...fpr));
  }
  out[`${prefix}_n_groups_eval`] = elig(false).length;
  out[`${prefix}_n_groups_total`] = rows.length;
}

function dfEps(yhat: ArrayLike<number>, g: Groups, alpha = 1): number {
  const rows = tabulate(yhat, yhat, g);
  const p1 = rows.map((r) => (r.tp + r.fp + alpha) / (r.n + 2 * alpha));
  const l1 = p1.map(Math.log), l0 = p1.map((p) => Math.log(1 - p));
  return Math.max(Math.max(...l1) - Math.min(...l1), Math.max(...l0) - Math.min(...l0));
}

/** fairness.fairness_report -> flat summary (group tables are not needed by the UI here). */
export function fairnessSummary(y: ArrayLike<number>, yhat: ArrayLike<number>, groups: Record<string, Groups>, attrs: string[], L: Limits, prob?: ArrayLike<number>) {
  const out = utility(y, yhat, prob);
  for (const a of [...attrs, "intersection"]) summarize(tabulate(y, yhat, groups[a]), a, L, out);
  out.intersection_df_eps = dfEps(yhat, groups.intersection);
  return out;
}

/** ethics.group_fairness for one attribute. */
export function groupFairness(y: ArrayLike<number>, yhat: ArrayLike<number>, prob: ArrayLike<number>, g: Groups, L: Limits) {
  const rows = tabulate(y, yhat, g, prob).map((r) => {
    const ys = r.idx.map((i) => y[i]), ps = r.idx.map((i) => prob[i]);
    const base = r.n_pos / r.n, mean = r.sumS / r.n;
    return {
      group: r.group, n: r.n, n_pos: r.n_pos,
      selection_rate: (r.tp + r.fp) / r.n,
      tpr: r.n_pos ? r.tp / r.n_pos : null,
      fpr: r.n - r.n_pos ? r.fp / (r.n - r.n_pos) : null,
      precision: r.tp + r.fp ? r.tp / (r.tp + r.fp) : null,
      ece: r.n >= L.minSupport ? ece(ys, ps, 10) : null,
      mean_score: mean, base_rate: base, calibration_error: mean - base,
      eligible: r.n >= L.minSupport && r.n_pos >= L.minPositives,
    };
  });
  const e = rows.filter((r) => r.eligible);
  const gap = (f: (r: (typeof rows)[number]) => number | null): Num => {
    const v = e.map(f).filter((x): x is number => x != null);
    return v.length > 1 ? Math.max(...v) - Math.min(...v) : null;
  };
  const sr = e.map((r) => r.selection_rate);
  const di = sr.length > 1 && Math.max(...sr) > 0 ? Math.min(...sr) / Math.max(...sr) : null;
  const eo = [gap((r) => r.tpr), gap((r) => r.fpr)].filter((x): x is number => x != null);
  const tprs = e.map((r) => r.tpr).filter((x): x is number => x != null);
  return {
    groups: rows, demographic_parity_diff: gap((r) => r.selection_rate), disparate_impact_ratio: di,
    four_fifths_pass: di == null ? null : di >= L.fourFifths,
    equal_opportunity_diff: gap((r) => r.tpr), equalized_odds_diff: eo.length ? Math.max(...eo) : null,
    predictive_parity_diff: gap((r) => r.precision), calibration_gap: gap((r) => r.calibration_error),
    worst_group_tpr: tprs.length ? Math.min(...tprs) : null, n_groups_eligible: e.length, n_groups: rows.length,
  };
}

/** numpy.quantile with linear interpolation. */
export function quantile(a: number[], q: number): number {
  const s = [...a].sort((x, y) => x - y), h = (s.length - 1) * q, lo = Math.floor(h);
  return lo + 1 < s.length ? s[lo] + (h - lo) * (s[lo + 1] - s[lo]) : s[lo];
}

/** postprocess.fit_group_thresholds (equal opportunity, Hardt et al. 2016). */
export function fitGroupThresholds(p: ArrayLike<number>, y: ArrayLike<number>, g: Groups, base: number, L: Limits) {
  let pos = 0, hit = 0;
  for (let i = 0; i < y.length; i++) if (y[i] === 1) { pos++; if (p[i] >= base) hit++; }
  const target = pos ? hit / pos : 0.5;
  const grid = Array.from({ length: 97 }, (_, k) => (k === 96 ? 0.98 : 0.02 + k * (0.96 / 96)));
  const th: Record<string, number> = { __default__: base };
  g.labels.forEach((label, code) => {
    const scores: number[] = []; let n = 0;
    for (let i = 0; i < y.length; i++) if (g.codes[i] === code) { n++; if (y[i] === 1) scores.push(p[i]); }
    if (n < L.minSupport || scores.length < L.minPositives) return;
    let best = 0, bestCost = Infinity;
    grid.forEach((t, k) => {
      const tpr = scores.filter((s) => s >= t).length / scores.length;
      const cost = Math.abs(tpr - target) + 1e-4 * Math.abs(t - base);
      if (cost < bestCost) { bestCost = cost; best = k; }
    });
    th[label] = grid[best];
  });
  return th;
}

export function applyThresholds(p: ArrayLike<number>, g: Groups, th: Record<string, number>): Int8Array {
  const t = g.labels.map((l) => th[l] ?? th.__default__);
  const out = new Int8Array(p.length);
  for (let i = 0; i < p.length; i++) out[i] = p[i] >= t[g.codes[i]] ? 1 : 0;
  return out;
}
