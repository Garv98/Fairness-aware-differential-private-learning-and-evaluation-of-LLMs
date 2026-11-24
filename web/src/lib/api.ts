"use client";
import { useCallback, useEffect, useMemo, useState } from "react";

export const API = process.env.NEXT_PUBLIC_API ?? "http://127.0.0.1:8000";

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `HTTP ${r.status}`);
  const ct = r.headers.get("content-type") ?? "";
  return (ct.includes("application/json") ? r.json() : r.text()) as Promise<T>;
}
export const get = <T,>(path: string) => fetch(`${API}${path}`).then((r) => handle<T>(r));
export const post = <T,>(path: string, body: unknown) =>
  fetch(`${API}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => handle<T>(r));

// Response cache shared by all components. Entries are keyed by request path;
// `invalidate()` drops everything (used when the backend comes back online).
const cache = new Map<string, unknown>();
export const invalidate = () => cache.clear();

/** Fetch `path` and keep the result in sync with it.
 *  - data always belongs to the current path (reset on change, stale responses ignored)
 *  - error is cleared whenever the path changes or a reload starts */
export function useApi<T>(path: string | null) {
  const [state, setState] = useState<{ path: string | null; data: T | null; error: string | null; loading: boolean }>(
    { path, data: path && cache.has(path) ? (cache.get(path) as T) : null, error: null, loading: false });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!path) { setState({ path, data: null, error: null, loading: false }); return; }
    if (cache.has(path) && nonce === 0) { setState({ path, data: cache.get(path) as T, error: null, loading: false }); return; }
    let live = true;
    setState((s) => ({ path, data: s.path === path ? s.data : null, error: null, loading: true }));
    get<T>(path)
      .then((d) => { cache.set(path, d); if (live) setState({ path, data: d, error: null, loading: false }); })
      .catch((e) => { if (live) setState({ path, data: null, error: String(e.message ?? e), loading: false }); });
    return () => { live = false; };
  }, [path, nonce]);
  const reload = useCallback(() => { if (path) cache.delete(path); setNonce((n) => n + 1); }, [path]);
  const current = state.path === path;
  return { data: current ? state.data : null, error: current ? state.error : null, loading: !current || state.loading, reload };
}

/** Debounce a fast-changing value (e.g. a slider) before using it in a request. */
export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return v;
}

// ---------------------------------------------------------------- types
export type Applicant = Record<string, string | number>;
export interface MethodInfo { id: string; name: string; short: string; private: boolean; ours: boolean; baseline: boolean; aux_frac: number }
export interface Meta {
  schema: { categorical: Record<string, string[]>; numeric: Record<string, { min: number; max: number; typical_max: number; median: number }>; fields: string[]; education_order: string[] };
  models: { method: string; name: string; private: boolean; eps: number | null }[];
  methods: MethodInfo[]; epsilons: number[]; train_epsilons: number[]; train_epochs: { min: number; max: number; default: number };
  threshold: number; samples: { label: string; truth: number; disagreement: number; applicant: Applicant }[];
  scorecard: Record<string, number>; proxy_strength: { strong: number; moderate: number };
  task: { label: string; positive: string; negative: string; dataset_title: string };
  threshold_grid: { min: number; max: number; step: number };
  accountant_ui: { sigma: { min: number; max: number; step: number; default: number }; batch: { min: number; max: number; step: number }; epochs: { min: number; max: number } };
  recourse: Record<string, Record<string, number>>; reference_identity: Record<string, string>;
  info: { dataset: string; dataset_rows: number; train_size: number; val_size: number; test_size: number; n_features: number; group_attrs: string[];
    n_cells: number; min_support: number; min_positives: number; delta: number; seed: number; split: { test_size: number; val_size: number };
    hidden: number[]; epochs: number; dp: { expected_batch_size: number; lr: number; max_grad_norm: number } };
}
export interface Recourse { found: boolean; cost?: number; prob?: number; changes?: Record<string, string | number>; searched: number }
export interface Prediction {
  group: { sex: string; race: string; age: string; intersection: string }; eps: number; threshold: number; reference_identity: Record<string, string>;
  models: { method: string; name: string; private: boolean; eps: number | null; prob: number; decision: number; explain: { field: string; value: number }[]; recourse: Recourse | null; recourse_reference: Recourse | null }[];
  counterfactual: { method: string; name: string; cells: { sex: string; race: string; prob: number }[] }[];
}
export interface GroupRow { group: string; n: number; n_pos: number; accuracy: number; tpr: number | null; fpr: number | null; selection_rate: number; precision: number | null; base_rate: number }
export interface EthicsAttr {
  groups: { group: string; n: number; n_pos: number; selection_rate: number; tpr: number | null; fpr: number | null; precision: number | null; ece: number | null; mean_score: number; base_rate: number; eligible: boolean }[];
  demographic_parity_diff: number | null; disparate_impact_ratio: number | null; four_fifths_pass: boolean | null; equal_opportunity_diff: number | null;
  equalized_odds_diff: number | null; predictive_parity_diff: number | null; calibration_gap: number | null; worst_group_tpr: number | null; n_groups_eligible: number; n_groups: number;
}
export interface EvalEntry {
  method: string; name: string; private: boolean; eps: number | null;
  summary: Record<string, number | null>; groups: Record<string, GroupRow[]>;
  ethics: Record<string, EthicsAttr> & { consistency?: number };
  mia: Record<string, number | null>; mia_groups: { attribute: string; group: string; n: number; n_members: number; mia_auc: number; tpr_global: number; fpr_global: number; advantage: number }[];
  cost_of_privacy?: Record<string, Record<string, number | null>>;
}
export type Evaluation = Record<string, EvalEntry>;
export const key = (m: string, eps: number | null) => `${m}__${eps == null ? "inf" : String(Number(eps))}`;

// ----------------------------------------------------- meta-derived helpers
export function useMeta() {
  const r = useApi<Meta>("/api/meta");
  const helpers = useMemo(() => {
    const m = r.data;
    const byId = new Map((m?.methods ?? []).map((x) => [x.id, x]));
    return {
      order: (m?.methods ?? []).map((x) => x.id),
      privateIds: (m?.methods ?? []).filter((x) => x.private).map((x) => x.id),
      name: (id: string) => byId.get(id)?.name ?? id,
      short: (id: string) => byId.get(id)?.short ?? id,
      isPrivate: (id: string) => !!byId.get(id)?.private,
      oursId: (m?.methods ?? []).find((x) => x.ours)?.id,
      baseId: (m?.methods ?? []).find((x) => x.baseline)?.id ?? (m?.methods ?? []).find((x) => x.private)?.id ?? "",
      epsFor: (id: string) => (byId.get(id)?.private ? m?.epsilons ?? [] : [null]) as (number | null)[],
    };
  }, [r.data]);
  return { ...r, meta: r.data, ...helpers };
}
