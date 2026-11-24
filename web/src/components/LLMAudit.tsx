"use client";
import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Evaluation, MODE, key, post, useApi, useMeta } from "@/lib/api";
import { pct } from "@/lib/models";
import { ApiError, Card, Loading, ModelName, Seg, fmtPct } from "./ui";

interface Info {
  results: Record<string, string | number | null>[]; providers: Record<string, boolean>; models: Record<string, string[]>;
  l2: { model: string; identity: string; flip_rate: number | null; positive_rate: number | null; mean_prob_shift: number | null; positive_rate_gap: number | null }[];
  experiments: { id: string; name: string; measures: string; status: string }[];
}
// LLM series colours continue the categorical palette after the five model slots
const LLM_COLORS = ["#eda100", "#008300", "#9085e9", "#e34948"];
interface Out { model: string; provider: string; results: { sex: string; race: string; decision: number | null; prob: number | null; raw: string }[] }


export default function LLMAudit() {
  const { meta, error, order, short, isPrivate, oursId, baseId } = useMeta();
  const { data: info } = useApi<Info>("/api/llm");
  const { data: ev } = useApi<Evaluation>("/api/evaluation");
  const [i, setI] = useState(0);
  const [provider, setProvider] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState<{ system: string; user: string } | null>(null);
  const [out, setOut] = useState<Out | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const providers = Object.keys(info?.providers ?? {});
  const P = provider ?? providers.find((p) => info?.providers[p]) ?? providers[0];
  const a = meta?.samples[i]?.applicant;
  useEffect(() => { if (a) post<{ system: string; user: string }>("/api/llm/prompt", { applicant: a }).then(setPrompt).catch(() => setPrompt(null)); }, [a]);
  const available = useMemo(() => (P && info?.models?.[P]) || [], [P, info]);
  const studied = useMemo(() => [...new Set((info?.results ?? []).map((r) => String(r.model)))], [info]);
  useEffect(() => { if (P && !model && available.length) setModel(available.find((m) => studied.includes(m)) ?? available[0]); }, [P, model, available, studied]);
  if (error) return <ApiError error={error} />;
  if (!meta || !a) return <Loading />;
  const hasKey = !!(P && info?.providers[P]);
  const truth = meta.samples[i].truth;

  const run = async () => {
    setBusy(true); setErr(null); setOut(null);
    try { setOut(await post<Out>("/api/llm/query", { applicant: a, provider: P, model })); } catch (e) { setErr(String((e as Error).message)); }
    setBusy(false);
  };
  const decided = out?.results.filter((r) => r.decision != null) ?? [];
  const yes = decided.filter((r) => r.decision === 1).length;

  return (
    <div className="grid">
      <div className="grid g2">
        <Card title="Prompt sent to the LLM" hint="Built by the same serializer the experiments use, from a real test-set person. Only sex and race change during the audit."
          right={<select aria-label="applicant" value={i} onChange={(e) => { setI(Number(e.target.value)); setOut(null); }} style={{ padding: "5px 8px", border: "1px solid var(--rule)", borderRadius: 8 }}>{meta.samples.map((s, k) => <option key={k} value={k}>#{k + 1} · {s.label}</option>)}</select>}>
          <pre className="mono prompt">{prompt ? `system: ${prompt.system}\n\nuser: ${prompt.user}` : "…"}</pre>
          <p className="note">True answer for this person: {truth ? "> $50k" : "≤ $50k"}. {MODE === "browser" ? "Each audit sends 10 live requests to the provider." : "Responses are cached on disk, so repeated demos cost no API quota."}</p>
        </Card>
        <Card title="Run the counterfactual audit" hint={`Ask the LLM the same question for every sex × race identity (${meta.schema.categorical.sex.length * meta.schema.categorical.race.length} prompts) and compare answers.`}>
          <div className="grid" style={{ gap: 12 }}>
            <div className="field"><span className="lbl">Provider</span>
              <Seg label="provider" value={P} onChange={(p) => { setProvider(p); setModel(""); }} options={providers.map((p) => ({ v: p, l: <>{p} <span className={`badge ${info?.providers[p] ? "ok" : "mute"}`} style={{ marginLeft: 4, padding: "0 6px" }}>{info?.providers[p] ? "key set" : "no key"}</span></> }))} />
            </div>
            <div className="field"><label htmlFor="mdl">Model {available.length ? `(${available.length} available to your key)` : ""}</label>
              {!info ? <select id="mdl" disabled><option>loading models…</option></select> : available.length ? <select id="mdl" value={model} onChange={(e) => setModel(e.target.value)}>{available.map((m) => <option key={m} value={m}>{m}</option>)}</select>
                : <input id="mdl" value={model} onChange={(e) => setModel(e.target.value)} placeholder="model id" style={{ padding: "7px 9px", border: "1px solid var(--rule)", borderRadius: 8 }} />}</div>
            <div><button className="btn primary" type="button" onClick={run} disabled={busy || !hasKey || !model}>{busy ? "Querying…" : "Run audit"}</button></div>
            {!hasKey && (MODE === "browser"
              ? <div className="callout warn">No API key for <b>{P}</b> is configured on this deployment. Add <code>GROQ_API_KEY</code> to the hosting environment variables and redeploy.</div>
              : <div className="callout warn">No API key for <b>{P}</b>. Set it in the terminal before starting the API (for example <code>set GROQ_API_KEY=…</code>), then restart the API.</div>)}
            {err && <div className="callout bad">{err}</div>}
          </div>
        </Card>
      </div>

      {out && (
        <Card title={`Results: ${out.model}`} hint={`${yes} of ${decided.length} identities predicted > $50k${decided.length < out.results.length ? ` · ${out.results.length - decided.length} unparseable answers` : ""}. A counterfactually fair model gives every identity the same answer.`}
          right={<span className={`badge ${yes === 0 || yes === decided.length ? "ok" : "bad"}`}>{yes === 0 || yes === decided.length ? "consistent" : "decision depends on identity"}</span>}>
          <div className="grid g5" style={{ gap: 10 }}>
            {out.results.map((r) => (
              <div key={r.sex + r.race} className="card" style={{ padding: 12, borderTop: `4px solid ${r.decision === 1 ? "var(--ok)" : r.decision === 0 ? "var(--bad)" : "var(--ink-3)"}` }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{r.sex} · {r.race}</div>
                <div style={{ fontSize: 24, fontWeight: 600 }} className="num">{r.prob != null ? `${Math.round(r.prob * 100)}%` : "–"}</div>
                <span className={`badge ${r.decision === 1 ? "ok" : r.decision === 0 ? "bad" : "mute"}`}>{r.decision === 1 ? "> $50k" : r.decision === 0 ? "≤ $50k" : "no answer"}</span>
                <details style={{ marginTop: 6 }}><summary className="muted" style={{ fontSize: 12, cursor: "pointer" }}>raw answer</summary><pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>{r.raw}</pre></details>
              </div>
            ))}
          </div>
        </Card>
      )}

      {info && info.results.length > 0 && meta && ev && (() => {
        const e0 = meta.epsilons[0];
        type Row = { kind: "ml" | "llm"; id: string; name: string; eps: number | null; s: Record<string, number | string | null>; i: number };
        const rows: Row[] = [
          ...order.filter((m) => m === baseId || m === oursId || !isPrivate(m)).map((m) => ({ kind: "ml" as const, id: m, name: short(m), eps: isPrivate(m) ? e0 : null, s: (ev[key(m, isPrivate(m) ? e0 : null)]?.summary ?? {}) as Record<string, number | null>, i: 0 })),
          ...info.results.map((r, i) => ({ kind: "llm" as const, id: String(r.model), name: String(r.model), eps: null, s: r, i })),
        ];
        const num = (v: number | string | null | undefined) => (typeof v === "number" ? v : null);
        const ids = [...new Set(info.l2.map((r) => r.identity))];
        const llms = [...new Set(info.l2.map((r) => r.model))];
        const flips = ids.map((idn) => ({ identity: idn.replaceAll("_", " "), ...Object.fromEntries(llms.map((m) => [m, info.l2.find((r) => r.model === m && r.identity === idn)?.flip_rate ?? null])) }));
        return (
          <>
            <Card title="L1 · LLMs vs our models on the same people" hint={`Zero-shot LLM decisions on ${info.results.map((r) => r.n).join(" / ")} real test-set people from the same split as the ML models (ML rows use the full test set; private models at ε = ${e0}). On a small LLM sample, worst-group recall covers only the groups large enough to measure.`}>
              <div className="tbl">
                <table>
                  <thead><tr><th>Model</th><th>Privacy</th><th className="n">People</th><th className="n">Accuracy</th><th className="n">Balanced acc.</th><th className="n">Recall</th><th className="n">Selection rate</th><th className="n">Worst-group recall</th><th className="n">Groups measured</th><th className="n">Unparseable</th></tr></thead>
                  <tbody>{rows.map((r) => (
                    <tr key={r.kind + r.id} className={r.id === oursId ? "me" : ""}>
                      <td>{r.kind === "ml" ? <ModelName id={r.id} name={r.name} /> : <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}><span className="sw" style={{ background: LLM_COLORS[r.i % LLM_COLORS.length], margin: 0 }} />{r.name} <span className="badge mute">LLM</span></span>}</td>
                      <td>{r.kind === "llm" ? "none (hosted API)" : r.eps ? `ε = ${r.eps}` : "none"}</td>
                      <td className="n">{r.kind === "llm" ? String(r.s.n) : meta.info.test_size.toLocaleString()}</td>
                      <td className="n">{pct(num(r.s.accuracy))}</td><td className="n">{pct(num(r.s.balanced_accuracy))}</td><td className="n">{pct(num(r.s.recall))}</td>
                      <td className="n">{pct(num(r.s.selection_rate))}</td><td className="n"><b>{pct(num(r.s.intersection_worst_tpr))}</b></td>
                      <td className="n">{r.s.intersection_n_groups_eval ?? "–"}</td><td className="n">{r.kind === "llm" ? pct(num(r.s.parse_failure_rate)) : "–"}</td>
                    </tr>))}</tbody>
                </table>
              </div>
            </Card>
            {info.l2.length > 0 && (
              <div className="grid g2">
                <Card title="L2 · How often does only the identity flip the decision?" hint="Share of people whose decision changes when an explicit race × sex statement is added to an otherwise identical prompt.">
                  <div style={{ height: 320 }}>
                    <ResponsiveContainer>
                      <BarChart data={flips} margin={{ top: 6, right: 8, left: 0, bottom: 40 }}>
                        <CartesianGrid vertical={false} stroke="var(--grid)" />
                        <XAxis dataKey="identity" interval={0} angle={-25} textAnchor="end" height={60} tick={{ fontSize: 11 }} />
                        <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                        <Tooltip formatter={fmtPct} />
                        <Legend itemSorter={() => 0} verticalAlign="top" height={28} />
                        {llms.map((m, i) => <Bar key={m} dataKey={m} name={m} fill={LLM_COLORS[i % LLM_COLORS.length]} radius={[3, 3, 0, 0]} />)}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
                <Card title="L2 · Per-identity outcomes" hint="Positive rate per identity, and mean change in the stated probability vs the prompt without an identity statement.">
                  <div className="tbl" style={{ maxHeight: 340, overflowY: "auto" }}>
                    <table>
                      <thead><tr><th>Model</th><th>Identity</th><th className="n">Flip rate</th><th className="n">Positive rate</th><th className="n">Prob. shift</th></tr></thead>
                      <tbody>{info.l2.map((r) => <tr key={r.model + r.identity}><td style={{ fontSize: 12 }}>{r.model}</td><td>{r.identity.replaceAll("_", " ")}</td><td className="n">{pct(r.flip_rate)}</td><td className="n">{pct(r.positive_rate)}</td><td className="n">{r.mean_prob_shift == null ? "–" : `${r.mean_prob_shift > 0 ? "+" : ""}${(r.mean_prob_shift * 100).toFixed(1)} pts`}</td></tr>)}</tbody>
                    </table>
                  </div>
                  <p className="note">Positive-rate gap across identities: {llms.map((m) => `${m}: ${pct(info.l2.find((r) => r.model === m)?.positive_rate_gap)}`).join(" · ")}. Compare with our models&apos; counterfactual grid in the Decision demo.</p>
                </Card>
              </div>
            )}
          </>
        );
      })()}

      <Card title="LLM study status" hint="Status is detected from the repository: scripts present → code ready; result files present → results.">
        {!info ? <Loading /> : (
          <div className="tbl">
            <table>
              <thead><tr><th>ID</th><th>Experiment</th><th>Measures</th><th>Status</th></tr></thead>
              <tbody>{info.experiments.map((e) => <tr key={e.id}><td className="mono">{e.id}</td><td>{e.name}</td><td className="muted">{e.measures}</td><td><span className={`badge ${e.status === "results" ? "ok" : e.status === "code ready" ? "brand" : "mute"}`}>{e.status}</span></td></tr>)}</tbody>
            </table>
          </div>
        )}
        {info && info.results.length > 0 && <p className="note">{info.results.length} saved L1 result file(s) in results/llm.</p>}
      </Card>
    </div>
  );
}
