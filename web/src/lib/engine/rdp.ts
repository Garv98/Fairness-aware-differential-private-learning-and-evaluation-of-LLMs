// Rényi-DP accountant for the subsampled Gaussian mechanism.
// Line-by-line port of Opacus 1.6 (opacus/accountants/analysis/rdp.py,
// RDPAccountant.DEFAULT_ALPHAS and accountants/utils.get_noise_multiplier),
// itself based on TF-Privacy (Mironov et al., 2019). Parity with Opacus is
// checked in web/scripts/parity.mts.

export const ALPHAS: number[] = [
  ...Array.from({ length: 99 }, (_, x) => 1 + (x + 1) / 10),
  ...Array.from({ length: 52 }, (_, i) => 12 + i),
];

function logAdd(x: number, y: number): number {
  const a = Math.min(x, y), b = Math.max(x, y);
  if (a === -Infinity) return b;
  return Math.log1p(Math.exp(a - b)) + b;
}

function logSub(x: number, y: number): number {
  if (x < y) throw new Error("log_sub: negative result");
  if (y === -Infinity) return x;
  if (x === y) return -Infinity;
  const d = Math.expm1(x - y);
  return Number.isFinite(d) ? Math.log(d) + y : x;
}

/** Generalised binomial coefficient C(alpha, i) for real alpha, integer i >= 0. */
function binom(alpha: number, i: number): number {
  let c = 1;
  for (let k = 0; k < i; k++) c *= (alpha - k) / (k + 1);
  return c;
}

/** log(erfc(x)) accurate in the tails (series for |x| < 3, continued fraction beyond). */
export function logErfc(x: number): number {
  if (x < 0) {
    // erfc(x) = 2 - erfc(-x); erfc(-x) is small here, no cancellation issue
    return Math.log(2 - Math.exp(logErfc(-x)));
  }
  if (x < 3) {
    // erf series: erf(x) = 2/sqrt(pi) * sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    let term = x, sum = x;
    for (let n = 1; n < 200; n++) {
      term *= (-x * x) / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    return Math.log(1 - (2 / Math.sqrt(Math.PI)) * sum);
  }
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))  (modified Lentz)
  const tiny = 1e-300;
  let f = x, C = x, D = 0;
  for (let n = 1; n < 500; n++) {
    const a = n / 2;
    D = x + a * D; if (Math.abs(D) < tiny) D = tiny; D = 1 / D;
    C = x + a / C; if (Math.abs(C) < tiny) C = tiny;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return -x * x - 0.5 * Math.log(Math.PI) - Math.log(f);
}

function logAInt(q: number, sigma: number, alpha: number): number {
  let logA = -Infinity;
  for (let i = 0; i <= alpha; i++) {
    const logCoef = Math.log(binom(alpha, i)) + i * Math.log(q) + (alpha - i) * Math.log(1 - q);
    logA = logAdd(logA, logCoef + (i * i - i) / (2 * sigma * sigma));
  }
  return logA;
}

function logAFrac(q: number, sigma: number, alpha: number): number {
  let a0 = -Infinity, a1 = -Infinity;
  const z0 = sigma * sigma * Math.log(1 / q - 1) + 0.5;
  for (let i = 0; ; i++) {
    const coef = binom(alpha, i);
    const logCoef = Math.log(Math.abs(coef));
    const j = alpha - i;
    const t0 = logCoef + i * Math.log(q) + j * Math.log(1 - q);
    const t1 = logCoef + j * Math.log(q) + i * Math.log(1 - q);
    const e0 = Math.log(0.5) + logErfc((i - z0) / (Math.SQRT2 * sigma));
    const e1 = Math.log(0.5) + logErfc((z0 - j) / (Math.SQRT2 * sigma));
    const s0 = t0 + (i * i - i) / (2 * sigma * sigma) + e0;
    const s1 = t1 + (j * j - j) / (2 * sigma * sigma) + e1;
    if (coef > 0) { a0 = logAdd(a0, s0); a1 = logAdd(a1, s1); }
    else { a0 = logSub(a0, s0); a1 = logSub(a1, s1); }
    if (Math.max(s0, s1) < -30) break;
  }
  return logAdd(a0, a1);
}

function rdpAt(q: number, sigma: number, alpha: number): number {
  if (q === 0) return 0;
  if (sigma === 0) return Infinity;
  if (q === 1) return alpha / (2 * sigma * sigma);
  const logA = Number.isInteger(alpha) ? logAInt(q, sigma, alpha) : logAFrac(q, sigma, alpha);
  return logA / (alpha - 1);
}

/** epsilon after `steps` steps of the sampled Gaussian mechanism (Balle et al. 2020 conversion). */
export function epsilonFor(sigma: number, q: number, steps: number, delta: number): number {
  let best = Infinity;
  for (const a of ALPHAS) {
    const rdp = rdpAt(q, sigma, a) * steps;
    const eps = rdp - (Math.log(delta) + Math.log(a)) / (a - 1) + Math.log((a - 1) / a);
    if (!Number.isNaN(eps) && eps < best) best = eps;
  }
  return best;
}

/** Same bisection as opacus get_noise_multiplier (returns sigma_high). */
export function calibrateSigma(targetEps: number, delta: number, q: number, steps: number, tol = 0.005): number {
  let epsHigh = Infinity, lo = 0, hi = 10;
  while (epsHigh > targetEps) {
    hi = 2 * hi;
    epsHigh = epsilonFor(hi, q, steps, delta);
    if (hi > 2000) throw new Error("The privacy budget is too low.");
  }
  while (targetEps - epsHigh > tol) {
    const s = (lo + hi) / 2;
    const e = epsilonFor(s, q, steps, delta);
    if (e < targetEps) { hi = s; epsHigh = e; } else lo = s;
  }
  return hi;
}

export const defaultDelta = (n: number) => 1 / n ** 1.1;
export const stepsFor = (epochs: number, q: number) => Math.trunc(epochs) * Math.round(1 / q);
