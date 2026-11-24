"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Applicant, Evaluation, Prediction, Recourse, key, post, useApi, useMeta } from "@/lib/api";
import { cellLabel, color, label, pct } from "@/lib/models";
import { ApiError, Card, Loading, ModelName, fmt3, fmtPct } from "./ui";

function RecourseText({ r, t }: { r: Recourse | null; t: number }) {
  if (r == null) return <span className="badge ok">already ≥ {t}</span>;
  if (!r.found) return <span className="badge bad">no feasible change ({r.searched} options)</span>;
  return (
    <span>
      {Object.entries(r.changes ?? {}).map(([f, v]) => <span key={f} className="badge brand" style={{ marginRight: 4 }}>{label(f)} → {String(v)}</span>)}
      <span className="muted" style={{ fontSize: 12 }}>cost {r.cost}</span>
    </span>
  );
}

export default function Predict({ eps }: { eps: number }) {
  const { meta, error, short } = useMeta();
  const { data: ev } = useApi<Evaluation>("/api/evaluation");
  const [a, setA] = useState<Applicant | null>(null);
  const [sample, setSample] = useState(0);
  const [res, setRes] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [perr, setPerr] = useState<string | null>(null);
  const [explainM, setExplainM] = useState<string | null>(null);
  const reqId = useRef(0);
  const [edited, setEdited] = useState(false);

  const run = async (app: Applicant | null = a) => {
    if (!app) return;
    const id = ++reqId.current;               // only the latest request may update the view
    setBusy(true); setPerr(null);
    try {
      const r = await post<Prediction>("/api/predict", { applicant: app, eps });
      if (id === reqId.current) setRes(r);
    } catch (e) { if (id === reqId.current) setPerr(String((e as Error).message)); }
    if (id === reqId.current) setBusy(false);
  };
  useEffect(() => { if (meta && !a && meta.samples.length) { const app = { ...meta.samples[0].applicant }; setA(app); } }, [meta, a]);
  useEffect(() => { if (a) run(a); }, [eps, a === null]); // eslint-disable-line react-hooks/exhaustive-deps

  const groupStats = useMemo(() => {
    if (!ev || !res) return null;
    return res.models.map((m) => {
      const e = ev[key(m.method, m.eps)];
      const row = e?.groups.intersection.find((g) => g.group === res.group.intersection);
      return { method: m.method, tpr: row?.tpr ?? null, n: row?.n ?? 0, npos: row?.n_pos ?? 0, overall: e?.summary.recall ?? null };
    });
  }, [ev, res]);

  if (error) return <ApiError error={error} />;
  if (!meta || !a) return <Loading what="models" />;
  const sc = meta.schema;
  const set = (f: string, v: string | number) => { setA({ ...a, [f]: v }); setEdited(true); };
  const task = meta.task;
  const truth = edited ? null : meta.samples[sample]?.truth;      // label only known for the unedited real person
  const T = res?.threshold ?? meta.threshold;
  const explain = res?.models.find((m) => m.method === (explainM ?? res.models[0]?.method));
  const cf0 = res?.counterfactual[0]?.cells ?? [];
  const races = [...new Set(cf0.map((c) => c.race))];
  const sexes = [...new Set(cf0.map((c) => c.sex))];
  const flips = (m: Prediction["counterfactual"][number]) => {
    const own = m.cells.find((c) => c.sex === a.sex && c.race === a.race);
    const d0 = own ? own.prob >= T : null;
    const ps = m.cells.map((c) => c.prob);
    return { changed: m.cells.filter((c) => (c.prob >= T) !== d0).length, total: m.cells.length, spread: Math.max(...ps) - Math.min(...ps) };
  };
  const isRef = res && Object.entries(res.reference_identity).every(([f, v]) => String(a[f]) === v);

  return (
    <div className="grid g-form" style={{ alignItems: "start" }}>
      <Card title="Applicant" hint="Real test-set people where the models disagree most (selected automatically). Edit any field.">
        <div className="field" style={{ marginBottom: 12 }}>
          <label htmlFor="sample">Load applicant</label>
          <select id="sample" value={sample} onChange={(e) => { const i = Number(e.target.value); setSample(i); const app = { ...meta.samples[i].applicant }; setA(app); setEdited(false); run(app); }}>
            {meta.samples.map((s, i) => <option key={i} value={i}>#{i + 1} · {s.label} · truly {s.truth ? task.positive : task.negative}</option>)}
          </select>
        </div>
        <div className="fields">
          {sc.fields.map((f) => (
            <div className="field" key={f} style={f === "native-country" || f === "occupation" ? { gridColumn: "span 2" } : undefined}>
              <label htmlFor={`f-${f}`}>{label(f)}</label>
              {sc.categorical[f] ? (
                <select id={`f-${f}`} value={String(a[f])} onChange={(e) => set(f, e.target.value)}>{sc.categorical[f].map((o) => <option key={o} value={o}>{o}</option>)}</select>
              ) : (
                <input id={`f-${f}`} type="number" value={a[f] === "" ? "" : Number(a[f])} min={sc.numeric[f]?.min} max={sc.numeric[f]?.max} onChange={(e) => set(f, e.target.value === "" ? "" : Number(e.target.value))} />
              )}
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
          <button className="btn primary" type="button" onClick={() => run()} disabled={busy}>{busy ? "Scoring…" : "Run all models"}</button>
          <button className="btn" type="button" onClick={() => { const opts = sc.categorical.sex; const b = { ...a, sex: opts[(opts.indexOf(String(a.sex)) + 1) % opts.length] }; setA(b); setEdited(true); run(b); }}>Flip sex</button>
        </div>
        {perr && <p className="note" style={{ color: "var(--bad)" }}>{perr}</p>}
      </Card>

      <div className="grid">
        <Card title={<>Decisions at privacy budget ε = {eps}</>} hint={res ? <>Task: predict {task.label} ({task.dataset_title}). Group: <b>{cellLabel(res.group.intersection)}</b>{truth != null ? <> · true answer: <b>{truth ? task.positive : task.negative}</b></> : <> · edited applicant (no ground truth)</>}</> : null}
          right={<span className="badge mute">decision threshold {T}</span>}>
          {!res ? <Loading what="predictions" /> : (
            <div className="grid g5" style={{ gap: 12 }}>
              {res.models.map((m) => {
                const yes = m.decision === 1, correct = truth == null ? null : yes === (truth === 1);
                return (
                  <div key={m.method} className="card" style={{ padding: 14, borderTop: `4px solid ${color(m.method)}`, display: "grid", gap: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13.5 }}>{m.name}</div>
                    <div className="muted" style={{ fontSize: 12 }}>{m.private ? `DP, ε = ${m.eps}` : "no privacy"}</div>
                    <div style={{ fontSize: 30, fontWeight: 600 }} className="num">{pct(m.prob, 0)}</div>
                    <div className="pbar" style={{ ["--t" as string]: `${T * 100}%` }}><span style={{ width: `${m.prob * 100}%`, background: color(m.method) }} /></div>
                    <span className={`badge ${yes ? "ok" : "bad"}`}>{yes ? `✓ Predicted ${task.positive}` : `✗ Predicted ${task.negative}`}</span>
                    {correct != null && <span className="muted" style={{ fontSize: 12 }}>{correct ? "correct" : yes ? "wrong (false positive)" : "wrong (false negative)"}</span>}
                  </div>
                );
              })}
            </div>
          )}
        </Card>

        {res && cf0.length > 0 && (
          <Card title="Counterfactual fairness: change only sex and race" hint="Same applicant, every sex × race combination. Cells = predicted probability (%); the outlined cell is the real applicant.">
            <div className="tbl">
              <table>
                <thead>
                  <tr><th rowSpan={2}>Model</th>{sexes.map((s) => <th key={s} colSpan={races.length} style={{ textAlign: "center", borderLeft: "2px solid var(--rule)" }}>{s}</th>)}<th rowSpan={2} className="n">Flips</th><th rowSpan={2} className="n">Spread</th></tr>
                  <tr>{cf0.map((c, k) => <th key={c.sex + c.race} className="n" style={{ fontSize: 11, padding: "6px", borderLeft: k % races.length === 0 ? "2px solid var(--rule)" : undefined }}>{c.race.split("-")[0]}</th>)}</tr>
                </thead>
                <tbody>
                  {res.counterfactual.map((m) => {
                    const f = flips(m);
                    return (
                      <tr key={m.method}>
                        <td><ModelName id={m.method} name={short(m.method)} /></td>
                        {m.cells.map((c, k) => {
                          const me = c.sex === a.sex && c.race === a.race, yes = c.prob >= T;
                          return <td key={c.sex + c.race} className="n" style={{ padding: "8px 6px", background: yes ? "var(--ok-bg)" : "var(--bad-bg)", color: yes ? "var(--ok)" : "var(--bad)", outline: me ? "2px solid var(--ink)" : undefined, outlineOffset: -2, fontWeight: me ? 700 : 500, borderLeft: k % races.length === 0 ? "2px solid var(--rule)" : undefined }}>{(c.prob * 100).toFixed(0)}</td>;
                        })}
                        <td className="n"><span className={`badge ${f.changed ? "bad" : "ok"}`}>{f.changed} / {f.total}</span></td>
                        <td className="n">{(f.spread * 100).toFixed(0)} pts</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="note">Green = predicted {task.positive}. &quot;Flips&quot; = identities that get a different decision than the real applicant. Counterfactual fairness asks for 0 flips.</p>
          </Card>
        )}

        {res && (
          <Card title="Algorithmic recourse: what would change the decision?" hint={<>Cheapest change to actionable features ({Object.keys(meta.recourse).map(label).join(", ").toLowerCase()}) that reaches the threshold, searched on the real model. {isRef ? null : <>Compared with the same person as <b>{Object.values(res.reference_identity).join(" ")}</b> (the most common identity in training data): unequal cost = unequal burden.</>}</>}>
            <div className="tbl">
              <table>
                <thead><tr><th>Model</th><th>This applicant</th>{!isRef && <th>Same person, {Object.values(res.reference_identity).join(" ")}</th>}</tr></thead>
                <tbody>{res.models.map((m) => (
                  <tr key={m.method}><td><ModelName id={m.method} name={short(m.method)} /></td><td><RecourseText r={m.recourse} t={T} /></td>{!isRef && <td><RecourseText r={m.recourse_reference} t={T} /></td>}</tr>))}
                </tbody>
              </table>
            </div>
            <p className="note">Cost units from config: {Object.entries(meta.recourse).map(([f, c]) => `${label(f)} ${Object.entries(c).map(([k, v]) => `${k.replaceAll("_", " ")} ${v}`).join(", ")}`).join(" · ")}.</p>
          </Card>
        )}

        {res && (
          <div className="grid g2">
            <Card title="Why did the model decide this?" hint={`Contribution of each field (gradient × input). Positive pushes towards ${task.positive}.`}
              right={<select aria-label="model to explain" value={explain?.method} onChange={(e) => setExplainM(e.target.value)} style={{ padding: "5px 8px", border: "1px solid var(--rule)", borderRadius: 8 }}>
                {res.models.map((m) => <option key={m.method} value={m.method}>{short(m.method)}</option>)}</select>}>
              {explain && (
                <div style={{ height: 230 }}>
                  <ResponsiveContainer>
                    <BarChart data={explain.explain.map((x) => ({ ...x, name: label(x.field) }))} layout="vertical" margin={{ left: 20, right: 20 }}>
                      <CartesianGrid horizontal={false} stroke="var(--grid)" />
                      <XAxis type="number" tickFormatter={(v) => v.toFixed(2)} />
                      <YAxis type="category" dataKey="name" width={110} />
                      <ReferenceLine x={0} stroke="var(--ink-3)" />
                      <Tooltip formatter={fmt3} />
                      <Bar dataKey="value" radius={4} name="contribution">{explain.explain.map((x) => <Cell key={x.field} fill={x.value >= 0 ? "#15803d" : "#c2410c"} />)}</Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
              {explain?.explain.some((x) => x.field === "sex" || x.field === "race") && <p className="note" style={{ color: "var(--bad)" }}>Sex or race directly influences this decision (direct discrimination risk).</p>}
            </Card>
            <Card title="How each model treats people like this" hint={<>Recall for <b>{cellLabel(res.group.intersection)}</b> on the test set vs everyone.</>}>
              {groupStats && (
                <div style={{ height: 230 }}>
                  <ResponsiveContainer>
                    <BarChart data={groupStats.map((g) => ({ name: short(g.method), group: g.tpr, overall: g.overall, id: g.method }))} margin={{ left: 0, right: 10 }}>
                      <CartesianGrid vertical={false} stroke="var(--grid)" />
                      <XAxis dataKey="name" interval={0} angle={-20} textAnchor="end" height={48} tick={{ fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                      <Tooltip formatter={fmtPct} />
                      <Bar dataKey="group" name="this group" radius={[4, 4, 0, 0]}>{groupStats.map((g) => <Cell key={g.method} fill={color(g.method)} />)}</Bar>
                      <Bar dataKey="overall" name="everyone" fill="#cbd5e1" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
              {groupStats && <p className="note">Coloured = this group, grey = everyone. Test-set group size: {groupStats[0].n} people, {groupStats[0].npos} high earners{groupStats[0].n < meta.info.min_support || groupStats[0].npos < meta.info.min_positives ? " (too small for reliable estimates)" : ""}.</p>}
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
