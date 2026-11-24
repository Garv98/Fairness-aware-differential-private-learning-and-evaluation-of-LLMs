// "Lite" live training in the browser: the same algorithms as
// src/fairdp/dp/dpsgd.py (DP-SGD, label balancing, IG-DPSGD) and
// src/fairdp/train.py (ERM, Group DRO), on a subset of the training data with
// a smaller network so it finishes in seconds. Creates a temporary model;
// the served models are never touched.

import { calibrateSigma, epsilonFor, stepsFor } from "./rdp";
import { clean, fairnessSummary, groupFairness, type Groups, type Limits } from "./metrics";
import { forward, type Layer } from "./engine";

export interface LiteSpec {
  n_train: number; n_test: number; dim: number; y_train: number[]; y_test: number[];
  cell_train: number[]; n_cells: number; cell_names: string[]; label_membership: number[][]; attr_membership: number[][][];
  groups_test: Record<string, Groups>; hidden: number[]; batch: number; lr: number; clip: number; aux_frac: number; delta: number; note: string;
  methods: Record<string, { private: boolean; method?: string; clip_mode?: string; dro?: boolean; balance_labels?: boolean; smoothing?: boolean; aux_frac?: number }>;
  dp_defaults: { loss_bound: number; dro_eta: number; dro_warmup_frac: number; weight_cap: number; smoothing_strength: number; ema: number };
  nonprivate: { batch_size: number; lr: number; dro_eta: number }; sensitive: string[];
}
export interface LiteData { spec: LiteSpec; X: Float32Array; Xte: Float32Array }

// ------------------------------------------------------------------ RNG
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gauss(rand: () => number) {
  let u = 0; while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// -------------------------------------------------------------- network
interface Net { layers: Layer[]; sizes: number[]; nParams: number }
function initNet(dims: number[], rand: () => number): Net {
  const layers: Layer[] = [];
  for (let i = 0; i + 1 < dims.length; i++) {
    const fanIn = dims[i], b = 1 / Math.sqrt(fanIn);          // PyTorch nn.Linear default init range
    layers.push({ in: fanIn, out: dims[i + 1], W: Array.from({ length: fanIn * dims[i + 1] }, () => (2 * rand() - 1) * b), b: Array.from({ length: dims[i + 1] }, () => (2 * rand() - 1) * b) });
  }
  const sizes = layers.flatMap((L) => [L.W.length, L.b.length]);
  return { layers, sizes, nParams: sizes.reduce((s, x) => s + x, 0) };
}

/** Per-sample BCE gradient (flattened, layer order W0,b0,W1,b1,...) and loss. */
function sampleGrad(net: Net, x: Float32Array, y: number, out: Float64Array): number {
  const { prob, acts } = forward(net.layers, x, true);
  const p = Math.min(Math.max(prob, 1e-12), 1 - 1e-12);
  const loss = -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  let g = new Float64Array([prob - y]);
  const offs: number[] = []; let o = 0;
  for (const s of net.sizes) { offs.push(o); o += s; }
  for (let li = net.layers.length - 1; li >= 0; li--) {
    const L = net.layers[li], a = acts[li], gin = new Float64Array(L.in);
    const wOff = offs[2 * li], bOff = offs[2 * li + 1];
    for (let j = 0; j < L.out; j++) {
      const gj = g[j];
      out[bOff + j] = gj;
      const row = j * L.in;
      for (let i = 0; i < L.in; i++) { out[wOff + row + i] = gj * a[i]; gin[i] += L.W[row + i] * gj; }
    }
    if (li > 0) for (let i = 0; i < L.in; i++) if (!(a[i] > 0)) gin[i] = 0;
    g = gin;
  }
  return loss;
}

class Adam {
  m: Float64Array; v: Float64Array; t = 0;
  constructor(n: number, private lr: number) { this.m = new Float64Array(n); this.v = new Float64Array(n); }
  step(net: Net, g: Float64Array) {
    this.t++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    let k = 0;
    for (const L of net.layers) for (const arr of [L.W, L.b]) for (let i = 0; i < arr.length; i++, k++) {
      this.m[k] = b1 * this.m[k] + (1 - b1) * g[k];
      this.v[k] = b2 * this.v[k] + (1 - b2) * g[k] * g[k];
      arr[i] -= (this.lr * (this.m[k] / c1)) / (Math.sqrt(this.v[k] / c2) + 1e-8);
    }
  }
}

// ------------------------------------------------- noisy cell statistics
const matVec = (M: number[][], v: ArrayLike<number>) => M.map((row) => row.reduce((s, m, c) => s + m * v[c], 0));
const matTVec = (M: number[][], v: number[]) => M[0].map((_, c) => M.reduce((s, row, r) => s + row[c] * v[r], 0));
const clip01 = (x: number) => Math.min(Math.max(x, 0), 1);

class CellStats {
  v: Record<string, Float64Array> = {};
  constructor(private memb: number[][][], private ema: number, private m: number, private smoothing: boolean) {}
  update(name: string, noisy: Float64Array) {
    const cur = this.v[name];
    this.v[name] = cur ? cur.map((x, i) => this.ema * x + (1 - this.ema) * noisy[i]) : noisy.slice();
  }
  ratio(num: string): Float64Array {
    const S = this.v[num], N = this.v.count.map((x) => Math.max(x, 0));
    if (!this.smoothing) return S.map((s, i) => clip01(s / Math.max(N[i], 1e-3)));
    const totS = S.reduce((a, b) => a + b, 0), totN = Math.max(N.reduce((a, b) => a + b, 0), 1e-3);
    const glob = clip01(totS / totN);
    const parents = this.memb.map((M) => {
      const ms = matVec(M, S), mn = matVec(M, N).map((x) => Math.max(x, 0));
      return matTVec(M, ms.map((s, j) => (s + this.m * glob) / (mn[j] + this.m)));
    });
    return S.map((s, c) => {
      const parent = clip01(parents.reduce((a, p) => a + p[c], 0) / parents.length);
      return clip01((s + this.m * parent) / (N[c] + this.m));
    });
  }
  freq(): Float64Array {
    let N = this.v.count.map((x) => Math.max(x, 0));
    if (this.smoothing) N = N.map((x) => x + 0.1);
    const tot = Math.max(N.reduce((a, b) => a + b, 0), 1e-6);
    return N.map((x) => Math.max(x / tot, 1e-4));
  }
}

// ------------------------------------------------------------- training
export interface EpochRow { epoch: number; loss?: number; accuracy: number; balanced_accuracy: number; recall: number; intersection_worst_tpr: number | null; intersection_tpr_gap: number | null; eps_spent?: number }
export interface LiteResult { noise?: Record<string, number>; final: Record<string, number>; final_ethics: Record<string, unknown> }

const yieldUI = () => new Promise<void>((r) => setTimeout(r, 0));

export async function trainLite(d: LiteData, method: string, eps: number, epochs: number, limits: Limits, threshold: number, seed: number,
                                onEpoch: (row: EpochRow, noise?: Record<string, number>) => void, cancelled: () => boolean = () => false): Promise<LiteResult> {
  const s = d.spec, mc = s.methods[method];
  if (!mc) throw new Error(`unknown method ${method}`);
  const rand = mulberry32(seed * 1000 + 17);
  const net = initNet([s.dim, ...s.hidden, 1], rand);
  const n = s.n_train, C = s.n_cells, row = (i: number) => d.X.subarray(i * s.dim, (i + 1) * s.dim);
  const g = new Float64Array(net.nParams), gi = new Float64Array(net.nParams);

  const evaluate = (withProb: boolean) => {
    const p = new Float64Array(s.n_test);
    for (let i = 0; i < s.n_test; i++) p[i] = forward(net.layers, d.Xte.subarray(i * s.dim, (i + 1) * s.dim)).prob;
    const yhat = Int8Array.from(p, (v) => (v >= threshold ? 1 : 0));
    return { p, yhat, summary: fairnessSummary(s.y_test, yhat, s.groups_test, Object.keys(s.groups_test).filter((a) => a !== "intersection"), limits, withProb ? p : undefined) };
  };
  const epochRow = (epoch: number, extra: Partial<EpochRow>): EpochRow => {
    const q = evaluate(false).summary;
    return clean({ epoch, ...extra, accuracy: q.accuracy, balanced_accuracy: q.balanced_accuracy, recall: q.recall,
      intersection_worst_tpr: q.intersection_worst_tpr ?? null, intersection_tpr_gap: q.intersection_tpr_gap ?? null });
  };

  let noise: Record<string, number> | undefined;
  if (!mc.private) {
    // ------------------------------------------- non-private (train.py)
    const np = s.nonprivate, opt = new Adam(net.nParams, np.lr);
    let qw = new Float64Array(C).fill(1 / C);
    const order = Array.from({ length: n }, (_, i) => i);
    for (let ep = 1; ep <= epochs; ep++) {
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      let tot = 0, nb = 0;
      for (let b0 = 0; b0 < n; b0 += np.batch_size) {
        const idx = order.slice(b0, b0 + np.batch_size);
        const grads: Float64Array[] = [], losses: number[] = [];
        for (const i of idx) { const gg = new Float64Array(net.nParams); losses.push(sampleGrad(net, row(i), s.y_train[i], gg)); grads.push(gg); }
        g.fill(0);
        let weights: number[];
        if (mc.method === "groupdro") {
          const cl = new Float64Array(C), cnt = new Float64Array(C);
          idx.forEach((i, k) => { cl[s.cell_train[i]] += losses[k]; cnt[s.cell_train[i]]++; });
          const ml = cl.map((x, c) => (cnt[c] ? x / cnt[c] : 0));
          qw = qw.map((x, c) => x * Math.exp(np.dro_eta * ml[c] * (cnt[c] > 0 ? 1 : 0)));
          const z = qw.reduce((a, b) => a + b, 0); qw = qw.map((x) => x / z);
          weights = idx.map((i) => qw[s.cell_train[i]] / cnt[s.cell_train[i]]);
          tot += ml.reduce((a, m, c) => a + qw[c] * m, 0);
        } else {
          weights = idx.map(() => 1 / idx.length);
          tot += losses.reduce((a, b) => a + b, 0) / idx.length;
        }
        grads.forEach((gg, k) => { for (let j = 0; j < g.length; j++) g[j] += weights[k] * gg[j]; });
        opt.step(net, g); nb++;
      }
      if (cancelled()) throw new Error("cancelled");
      onEpoch(epochRow(ep, { loss: tot / nb }));
      await yieldUI();
    }
  } else {
    // ----------------------------------------------- DP-SGD (dpsgd.py)
    const df = s.dp_defaults, aux = mc.aux_frac ?? 0;
    const needsStats = !!(mc.dro || mc.balance_labels);
    const k = needsStats ? 1 + (mc.dro ? 1 : 0) : 0;
    const q = s.batch / n, spe = Math.round(1 / q), T = stepsFor(epochs, q);
    const sigmaEff = calibrateSigma(eps, s.delta, q, T);
    const sigmaG = k && aux ? sigmaEff / Math.sqrt(1 - aux) : sigmaEff;
    const sigmaS = k && aux ? sigmaEff * Math.sqrt(k / aux) : Infinity;
    noise = clean({ sigma_eff: sigmaEff, sigma_grad: sigmaG, sample_rate: q, delta: s.delta, steps: T, epsilon: epsilonFor(sigmaEff, q, T, s.delta) });
    const opt = new Adam(net.nParams, s.lr), C0 = s.clip;
    const stats = new CellStats(s.attr_membership, df.ema, df.smoothing_strength, !!mc.smoothing);
    let logR = new Float64Array(C), cellW = new Float64Array(C).fill(1), wScale = 1;
    const warmup = Math.trunc(df.dro_warmup_frac * T);
    for (let step = 0; step < T; step++) {
      const idx: number[] = [];
      for (let i = 0; i < n; i++) if (rand() < q) idx.push(i);
      g.fill(0);
      const cnt = new Float64Array(C), lossSum = new Float64Array(C);
      for (const i of idx) {
        const loss = sampleGrad(net, row(i), s.y_train[i], gi);
        let nrm = 0; for (let j = 0; j < gi.length; j++) nrm += gi[j] * gi[j];
        const c = s.cell_train[i], f = Math.min(C0 / Math.max(Math.sqrt(nrm), 1e-12), 1) * cellW[c];
        for (let j = 0; j < gi.length; j++) g[j] += f * gi[j];
        cnt[c]++; lossSum[c] += clip01(loss / df.loss_bound);
      }
      for (let j = 0; j < g.length; j++) g[j] = ((g[j] + gauss(rand) * sigmaG * C0) / (q * n)) * wScale;
      opt.step(net, g);
      if (k) {
        stats.update("count", cnt.map((x) => x + gauss(rand) * sigmaS));
        if (mc.dro) stats.update("loss", lossSum.map((x) => x + gauss(rand) * sigmaS));
        if (mc.balance_labels) {
          const pf = stats.freq(), My = s.label_membership;
          const pY = matVec(My, pf).map((x) => Math.max(x, 1e-3));
          const inv = matTVec(My, pY.map((x) => 1 / x)), mx = Math.max(...inv);
          cellW = Float64Array.from(inv, (x) => x / mx);
          wScale = 1 / pf.reduce((a, p, c) => a + p * cellW[c], 0);
        }
        if (mc.dro && step >= warmup) {
          const L = stats.ratio("loss"), pHat = stats.freq();
          const r = logR.map(Math.exp), z = pHat.reduce((a, p, c) => a + p * r[c], 0);
          const mean = L.reduce((a, l, c) => a + ((pHat[c] * r[c]) / z) * l, 0);
          const cap = Math.log(df.weight_cap);
          logR = logR.map((x, c) => Math.min(Math.max(x + df.dro_eta * (L[c] - mean), -cap), cap));
          const r2 = logR.map(Math.exp), mx = Math.max(...r2);
          cellW = r2.map((x) => x / mx);
          wScale = 1 / pHat.reduce((a, p, c) => a + p * cellW[c], 0);
        }
      }
      if ((step + 1) % spe === 0) {
        const ep = (step + 1) / spe;
        if (cancelled()) throw new Error("cancelled");
        onEpoch(epochRow(ep, { eps_spent: epsilonFor(sigmaEff, q, stepsFor(ep, q), s.delta) }), noise);
        await yieldUI();
      }
    }
  }
  const fin = evaluate(true);
  const final_ethics = Object.fromEntries(s.sensitive.map((a) => {
    const { groups: _g, ...rest } = groupFairness(s.y_test, fin.yhat, fin.p, s.groups_test[a], limits); void _g;
    return [a, rest];
  }));
  return { noise, final: clean(fin.summary), final_ethics: clean(final_ethics) };
}
