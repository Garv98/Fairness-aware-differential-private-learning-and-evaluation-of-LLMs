"use client";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Evaluation, key, useApi, useDebounced, useMeta } from "@/lib/api";
import { cellLabel, color, epsLabel, f3 } from "@/lib/models";
import { ApiError, Card, Kpi, Loading, ModelName, Seg, fmt3, fmtPct } from "./ui";

interface Acc { eps: number; n: number; q: number; steps: number; delta: number; batch: number; epochs: number; curve: [number, number][]; sigma_for: Record<string, number> }
interface Study { run: string; rows: { method: string; eps: number | null; seeds: number; [k: string]: number | string | null }[] }
type Training = Record<string, { private: boolean; noise?: Record<string, number>; seconds?: number }>;

export default function Privacy({ eps }: { eps: number }) {
  const { meta, order, privateIds, short, isPrivate, epsFor, oursId } = useMeta();
  const [sigmaSel, setSigma] = useState<number | null>(null);
  const [batch, setBatch] = useState<number | null>(null);
  const [epochs, setEpochs] = useState<number | null>(null);
  const sigma = sigmaSel ?? meta?.accountant_ui.sigma.default ?? 1;
  const b = batch ?? meta?.info.dp.expected_batch_size, ep = epochs ?? meta?.info.epochs;
  const qs = useDebounced(`sigma=${sigma}&batch=${b}&epochs=${ep}`, 200);
  const q = meta ? `/api/accountant?${qs}` : null;
  const { data: acc, error } = useApi<Acc>(q);
  const { data: study } = useApi<Study>("/api/study");
  const { data: ev } = useApi<Evaluation>("/api/evaluation");
  const { data: tr } = useApi<Training>("/api/training");
  const [miaModel, setMiaModel] = useState<string | null>(null);
  const [metric, setMetric] = useState("intersection_worst_tpr");
  if (error) return <ApiError error={error} />;
  if (!meta) return <Loading />;
  const ui = meta.accountant_ui;
  const odds = (e: number) => { const x = Math.exp(e); return Number.isFinite(x) ? (x < 10 ? x.toFixed(2) : x < 1e6 ? x.toFixed(0) : x.toExponential(1)) : "∞"; };

  const rows = study?.rows ?? [];
  const sv = (m: string, e: number | null, k: string) => (rows.find((r) => r.method === m && r.eps === e)?.[k] ?? null) as number | null;
  const studyEps = [...new Set(rows.filter((r) => r.eps != null).map((r) => r.eps as number))].sort((a, z) => a - z);
  const trade: Record<string, number | string | null>[] = studyEps.map((e) => ({ eps: `ε=${e}`, ...Object.fromEntries(privateIds.flatMap((m) => [[m, sv(m, e, metric)], [`${m}_sd`, sv(m, e, `${metric}_sd`)]])) }));
  const tv = trade.flatMap((r) => privateIds.map((m) => r[m] as number | null)).filter((v): v is number => v != null);
  const ref = order.find((m) => !isPrivate(m));
  const seeds = rows[0]?.seeds;

  const miaRows = ev ? order.flatMap((m) => epsFor(m).map((e) => ev[key(m, e)]).filter(Boolean)).map((x) => ({ name: `${short(x.method)}${x.eps ? ` ε=${x.eps}` : ""}`, id: x.method, auc: x.mia.mia_auc ?? 0 })) : [];
  const aucs = miaRows.map((r) => r.auc);
  const mm = miaModel ?? oursId ?? order[0];
  const miaG = ev?.[key(mm, isPrivate(mm) ? eps : null)]?.mia_groups.filter((g) => g.attribute === "intersection") ?? [];
  const aux = meta.methods.filter((m) => m.aux_frac > 0);
  const lim = meta.scorecard.mia_auc_max;

  return (
    <div className="grid">
      <div className="grid g2">
        <Card title="Privacy budget calculator" hint={`Rényi-DP accountant (Opacus) for DP-SGD on this training set (n = ${meta.info.train_size.toLocaleString()}). Every private model is calibrated to an exact ε this way.`}>
          <div className="grid" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: 16 }}>
            <div className="grid" style={{ gap: 14 }}>
              <div className="field"><label htmlFor="s">Noise multiplier σ: <b className="num">{sigma.toFixed(2)}</b></label><input id="s" type="range" min={ui.sigma.min} max={ui.sigma.max} step={ui.sigma.step} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} /></div>
              <div className="field"><label htmlFor="b">Expected batch size: <b className="num">{b}</b></label><input id="b" type="range" min={ui.batch.min} max={ui.batch.max} step={ui.batch.step} value={b} onChange={(e) => setBatch(Number(e.target.value))} /></div>
              <div className="field"><label htmlFor="e">Epochs: <b className="num">{ep}</b></label><input id="e" type="range" min={ui.epochs.min} max={ui.epochs.max} step={1} value={ep} onChange={(e) => setEpochs(Number(e.target.value))} /></div>
              {acc && <div className="mono muted" style={{ fontSize: 12 }}>q = {acc.q.toFixed(4)} · steps = {acc.steps.toLocaleString()} · δ = {acc.delta.toExponential(1)}</div>}
            </div>
            <div className="grid" style={{ gap: 10, alignContent: "start" }}>
              <div className="card kpi ledger-card">
                <span className="k">Privacy spent</span>
                <span className="v display" style={{ fontSize: 40, opacity: q && acc ? 1 : 0.5 }}>ε = {acc ? (acc.eps > 99 ? ">99" : acc.eps.toFixed(2)) : "…"}</span>
                <span className="d">an attacker&apos;s odds about any one person change by at most ×{acc ? odds(acc.eps) : "…"}</span>
              </div>
              {acc && <table><thead><tr><th>Target</th><th className="n">σ needed</th></tr></thead><tbody>{Object.entries(acc.sigma_for).map(([t, s]) => <tr key={t}><td>ε = {Number(t)}</td><td className="n">{s.toFixed(3)}</td></tr>)}</tbody></table>}
            </div>
          </div>
          {acc && (
            <div style={{ height: 180, marginTop: 10 }}>
              <ResponsiveContainer>
                <LineChart data={acc.curve.map(([x, y]) => ({ epochs: x, eps: y }))} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke="var(--grid)" />
                  <XAxis dataKey="epochs" type="number" domain={["dataMin", "dataMax"]} label={{ value: "epochs", position: "insideBottomRight", offset: -2, fontSize: 11 }} />
                  <YAxis tickFormatter={(v) => v.toFixed(1)} width={40} />
                  <Tooltip formatter={fmt3} />
                  <Line dataKey="eps" name="ε" stroke="var(--brand)" strokeWidth={2.5} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card title="Privacy–utility trade-off (multi-seed)" hint={study?.run ? `Mean ± 1 sd over ${seeds} seeds from experiment run "${study.run}".` : "No multi-seed study found."}
          right={<Seg label="metric" value={metric} onChange={setMetric} options={[{ v: "intersection_worst_tpr", l: "Worst-group recall" }, { v: "balanced_accuracy", l: "Balanced acc." }, { v: "accuracy", l: "Accuracy" }]} />}>
          <div style={{ height: 330 }}>
            <ResponsiveContainer>
              <LineChart data={trade} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="eps" />
                <YAxis domain={[Math.max(0, Math.floor((Math.min(...tv, 1) - 0.1) * 10) / 10), Math.min(1, Math.ceil((Math.max(...tv, 0) + 0.1) * 10) / 10)]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip formatter={fmtPct} />
                <Legend itemSorter={() => 0} />
                {ref && sv(ref, null, metric) != null && <ReferenceLine y={sv(ref, null, metric)!} stroke={color(ref)} strokeDasharray="5 4" label={{ value: short(ref), position: "insideTopLeft", fill: color(ref), fontSize: 11 }} />}
                {privateIds.map((m) => <Line key={m} dataKey={m} name={short(m)} stroke={color(m)} strokeWidth={2.5} dot={{ r: 5 }}><ErrorBar dataKey={`${m}_sd`} width={6} stroke={color(m)} /></Line>)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid g2">
        <Card title="Membership-inference attack" hint="Can an attacker tell whether a person was in the training data from the model's loss? AUC 0.5 = coin flip.">
          {!ev ? <Loading /> : (
            <div style={{ height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={miaRows} margin={{ top: 6, right: 8, left: 0, bottom: 50 }}>
                  <CartesianGrid vertical={false} stroke="var(--grid)" />
                  <XAxis dataKey="name" interval={0} angle={-30} textAnchor="end" height={70} tick={{ fontSize: 11 }} />
                  <YAxis domain={[0.4, Math.max(lim + 0.05, ...aucs)]} tickFormatter={(v) => v.toFixed(2)} />
                  <ReferenceLine y={0.5} stroke="var(--ink-2)" strokeDasharray="4 4" label={{ value: "coin flip", position: "insideBottomRight", fontSize: 11 }} />
                  <ReferenceLine y={lim} stroke="var(--bad)" strokeDasharray="2 4" label={{ value: "leakage threshold", position: "insideTopRight", fontSize: 11, fill: "var(--bad)" }} />
                  <Tooltip formatter={fmt3} />
                  <Bar dataKey="auc" name="attack AUC" radius={[3, 3, 0, 0]}>{miaRows.map((r) => <Cell key={r.name} fill={color(r.id)} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {aucs.length > 0 && <p className="note">Attack AUC ranges {f3(Math.min(...aucs))}–{f3(Math.max(...aucs))} across all {aucs.length} models{Math.max(...aucs) <= lim ? `: below the ${lim} leakage threshold, so this loss-threshold attack learns nothing measurable. DP bounds leakage even for models that would memorise.` : "."}</p>}
        </Card>
        <Card title="Attack success per group" hint="Disparate vulnerability: one global attack threshold (flags 10% of non-members); advantage = TPR − FPR within each group. 0 = no leakage."
          right={<select aria-label="model" value={mm} onChange={(e) => setMiaModel(e.target.value)} style={{ padding: "5px 8px", border: "1px solid var(--rule)", borderRadius: 8 }}>{order.map((m) => <option key={m} value={m}>{short(m)}</option>)}</select>}>
          <div className="tbl" style={{ maxHeight: 340, overflowY: "auto" }}>
            <table>
              <thead><tr><th>Group</th><th className="n">People</th><th className="n">TPR</th><th className="n">FPR</th><th className="n">Advantage</th><th className="n">AUC</th></tr></thead>
              <tbody>{[...miaG].sort((a, z) => (z.advantage ?? 0) - (a.advantage ?? 0)).map((g) => (
                <tr key={g.group}><td>{cellLabel(g.group)}</td><td className="n">{g.n}</td><td className="n">{f3(g.tpr_global)}</td><td className="n">{f3(g.fpr_global)}</td>
                  <td className="n" style={{ color: (g.advantage ?? 0) > 0.05 ? "var(--bad)" : undefined }}>{f3(g.advantage)}</td><td className="n">{f3(g.mia_auc)}</td></tr>))}</tbody>
            </table>
          </div>
        </Card>
      </div>

      <Card title="Noise plans of the served models" hint={`Exact parameters used to train each private model. ${aux.map((m) => `${short(m.id)} spends ${Math.round(m.aux_frac * 100)}%`).join(", ")} of the same budget on noisy group statistics.`}>
        {!tr ? <Loading /> : (
          <div className="tbl">
            <table>
              <thead><tr><th>Model</th><th className="n">Target ε</th><th className="n">Accounted ε</th><th className="n">δ</th><th className="n">σ (joint)</th><th className="n">σ gradients</th><th className="n">σ statistics</th><th className="n">Steps</th><th className="n">Train time</th></tr></thead>
              <tbody>{Object.entries(tr).filter(([, v]) => v.private && v.noise).map(([k, v]) => {
                const [m, e] = k.split("__"), n = v.noise!;
                return <tr key={k}><td><ModelName id={m} name={short(m)} /></td><td className="n">{epsLabel(Number(e))}</td><td className="n">{f3(n.epsilon)}</td><td className="n">{n.delta?.toExponential(1)}</td><td className="n">{f3(n.sigma_eff)}</td><td className="n">{f3(n.sigma_grad)}</td><td className="n">{n.sigma_stats ? f3(n.sigma_stats, 2) : "–"}</td><td className="n">{n.steps}</td><td className="n">{v.seconds}s</td></tr>;
              })}</tbody>
            </table>
          </div>
        )}
      </Card>
      <div className="grid g3">
        <Kpi k="Guarantee" v="(ε, δ)-DP" d={`training data only, via DP-SGD + RDP accounting · ε ∈ {${meta.epsilons.join(", ")}}`} />
        <Kpi k="δ (failure probability)" v={meta.info.delta.toExponential(2)} d={`1 / n^1.1 with n = ${meta.info.train_size.toLocaleString()}`} />
        <Kpi k="Not covered by the guarantee" v="preprocessing" d="scaler / encoder statistics and group categories (stated limitation)" />
      </div>
    </div>
  );
}
