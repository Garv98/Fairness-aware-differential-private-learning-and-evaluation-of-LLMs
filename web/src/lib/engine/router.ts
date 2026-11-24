// Browser mode: answers the same /api/* paths as server/app.py, from the
// exported models and results in /public/static. LLM calls go to the Next.js
// route handlers (src/app/api/llm), which keep the provider key server-side.

import { Engine, HttpError, modelKey, type StaticData } from "./engine";
import { trainLite, type LiteData, type LiteSpec } from "./train";

const BASE = (process.env.NEXT_PUBLIC_BASE_PATH ?? "") + "/static";
const j = <T,>(name: string) => fetch(`${BASE}/${name}`).then((r) => { if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`); return r.json() as Promise<T>; });

let enginePromise: Promise<Engine> | null = null;
export function engine(): Promise<Engine> {
  enginePromise ??= Promise.all([j<StaticData["engine"]>("engine.json"), j<StaticData["models"]>("models.json"), j<StaticData["scores"]>("scores.json"), j<StaticData["bundle"]>("bundle.json")])
    .then(([e, m, s, b]) => new Engine({ engine: e, models: m, scores: s, bundle: b }))
    .catch((err) => { enginePromise = null; throw err; });
  return enginePromise;
}

let litePromise: Promise<LiteData> | null = null;
function lite(): Promise<LiteData> {
  litePromise ??= Promise.all([j<LiteSpec>("lite.json"), fetch(`${BASE}/lite.bin`).then((r) => r.arrayBuffer())]).then(([spec, buf]) => {
    const all = new Float32Array(buf), n = spec.n_train * spec.dim;
    return { spec, X: all.subarray(0, n), Xte: all.subarray(n, n + spec.n_test * spec.dim) };
  }).catch((err) => { litePromise = null; throw err; });
  return litePromise;
}

// ------------------------------------------------------- live training jobs
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const JOBS = new Map<string, any>();

function pyStr(x: number) { return Number.isInteger(x) ? x.toFixed(1) : String(x); }

function num(qs: URLSearchParams, name: string, def?: number): number | undefined {
  const v = qs.get(name);
  if (v == null || v === "") return def;
  const x = Number(v);
  if (!Number.isFinite(x)) throw new HttpError(422, `${name} must be a number`);
  return x;
}

function checkModel(E: Engine, method: string | null, eps: number | undefined): number | null {
  const m = (E.meta.methods as { id: string; private: boolean }[]).find((x) => x.id === method);
  if (!m) throw new HttpError(400, "unknown method");
  if (!m.private) return null;
  if (eps == null || !(E.meta.epsilons as number[]).includes(eps)) throw new HttpError(400, `eps must be one of ${E.meta.epsilons}`);
  return eps;
}

function checkAttr(E: Engine, attr: string) {
  const ok = [...(E.meta.info.group_attrs as string[]), "intersection"];
  if (!ok.includes(attr)) throw new HttpError(400, `attr must be one of ${ok}`);
}

async function llmInfo(E: Engine) {
  let live: { providers: Record<string, boolean>; models: Record<string, string[]> } = { providers: {}, models: {} };
  try { const r = await fetch("/api/llm"); if (r.ok) live = await r.json(); } catch { /* offline: show stored results only */ }
  return { ...E.d.bundle.llm, ...live };
}

export async function localGet(path: string): Promise<unknown> {
  const u = new URL(path, "http://local"), qs = u.searchParams, p = u.pathname;
  if (p === "/api/health") return { ok: true, mode: "browser" };
  const E = await engine();
  const b = E.d.bundle;
  switch (p) {
    case "/api/meta": return b.meta;
    case "/api/evaluation": case "/api/study": case "/api/training": case "/api/proxy": case "/api/provenance": case "/api/credit":
      return b[p.slice(5)];
    case "/api/scorecard": {
      const e = num(qs, "eps"); const sc = e == null ? undefined : b.scorecard[pyStr(e)];
      if (!sc) throw new HttpError(400, `eps must be one of ${E.meta.epsilons}`);
      return sc;
    }
    case "/api/model_card": {
      const e = checkModel(E, qs.get("method"), num(qs, "eps"));
      const card = b.model_cards[modelKey(qs.get("method")!, e)];
      if (!card) throw new HttpError(404, "no evaluation for this model; run train_models.py");
      return card;
    }
    case "/api/accountant": {
      const ui = E.meta.accountant_ui;
      const sigma = num(qs, "sigma", ui.sigma.default)!, batch = num(qs, "batch", E.meta.info.dp.expected_batch_size)!, epochs = num(qs, "epochs", E.meta.info.epochs)!;
      if (sigma < ui.sigma.min || sigma > ui.sigma.max || batch < ui.batch.min || batch > ui.batch.max || epochs < ui.epochs.min || epochs > ui.epochs.max
          || !Number.isInteger(batch) || !Number.isInteger(epochs)) throw new HttpError(422, "accountant parameters out of range");
      return E.accountant(Math.round(sigma * 1e4) / 1e4, batch, epochs);
    }
    case "/api/mitigation": {
      const e = checkModel(E, qs.get("method"), num(qs, "eps"));
      const policy = qs.get("policy") ?? "single", attr = qs.get("attr") ?? "sex";
      checkAttr(E, attr);
      const g = E.meta.threshold_grid, t = num(qs, "t", E.meta.threshold)!;
      if (t < g.min || t > g.max) throw new HttpError(422, `t must be between ${g.min} and ${g.max}`);
      return E.mitigation(modelKey(qs.get("method")!, e), policy, attr, t);
    }
    case "/api/sweep": {
      const e = checkModel(E, qs.get("method"), num(qs, "eps"));
      const attr = qs.get("attr") ?? "sex";
      checkAttr(E, attr);
      return E.sweep(modelKey(qs.get("method")!, e), attr);
    }
    case "/api/llm": return llmInfo(E);
  }
  const m = p.match(/^\/api\/train\/(\w+)$/);
  if (m) {
    const job = JOBS.get(m[1]);
    if (!job) throw new HttpError(404, "no such job");
    return structuredClone(job);
  }
  throw new HttpError(404, "Not Found");
}

export async function localPost(path: string, body: Record<string, unknown>): Promise<unknown> {
  if (path.startsWith("/api/llm/")) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new HttpError(r.status, (await r.json().catch(() => ({}))).detail ?? `HTTP ${r.status}`);
    return r.json();
  }
  const E = await engine();
  if (path === "/api/predict") return E.predict(body.applicant as Record<string, string | number>, Number(body.eps));
  if (path === "/api/train") {
    const method = String(body.method), meta = E.meta, lim = meta.train_epochs;
    const m = (meta.methods as { id: string; name: string; private: boolean }[]).find((x) => x.id === method);
    if (!m) throw new HttpError(400, "unknown method");
    const epochs = Number(body.epochs), eps = Number(body.eps ?? 1);
    if (!(epochs >= lim.min && epochs <= lim.max)) throw new HttpError(400, `epochs must be ${lim.min}..${lim.max}`);
    if (m.private && !(meta.train_epsilons as number[]).includes(eps)) throw new HttpError(400, `eps must be one of ${meta.train_epsilons}`);
    if ([...JOBS.values()].some((x) => x.status === "running")) throw new HttpError(409, "A training job is already running.");
    const data = await lite();
    const id = Math.random().toString(16).slice(2, 10);
    const job = { id, status: "running", method, name: m.name, private: m.private, eps: m.private ? eps : null, epochs_total: epochs,
      epochs: [] as unknown[], started: Date.now() / 1000, lite: data.spec.note } as Record<string, unknown> & { epochs: unknown[] };
    JOBS.set(id, job);
    trainLite(data, method, eps, epochs, E.L, E.d.engine.threshold, meta.info.seed,
      (row, noise) => { job.epochs.push(row); if (noise) job.noise = noise; })
      .then((r) => { job.final = r.final; job.final_ethics = r.final_ethics; if (r.noise) job.noise = r.noise; job.status = "done"; })
      .catch((e) => { job.status = "error"; job.error = String(e?.message ?? e); });
    return { id };
  }
  throw new HttpError(404, "Not Found");
}
