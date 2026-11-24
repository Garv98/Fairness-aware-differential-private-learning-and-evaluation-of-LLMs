"use client";
import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { EthicsAttr, get, useApi, useDebounced, useMeta } from "@/lib/api";
import { cellLabel, f3, pct } from "@/lib/models";
import { ApiError, Card, Kpi, Loading, Pass, Seg, fmt3 } from "./ui";

interface Result { policy: string; attribute: string; threshold: number; thresholds: Record<string, number>; summary: Record<string, number | null>; ethics: Record<string, EthicsAttr> }
type Sweep = { t: number; accuracy: number; balanced_accuracy: number; worst_tpr: number | null; dp_diff: number | null; di_ratio: number | null; eo_diff: number | null }[];
const POLICIES = [
  { v: "single", l: "One threshold for all", d: "Same cut-off for everyone (status quo)." },
  { v: "equal_opportunity", l: "Equal opportunity", d: "Per-group thresholds so qualified people are recognised at the same rate (equal TPR), fitted on validation data." },
  { v: "demographic_parity", l: "Demographic parity", d: "Per-group thresholds so every group is selected at the same rate, fitted on validation data." },
] as const;

export default function Mitigation({ eps }: { eps: number }) {
  const { meta, error, order, short, isPrivate, oursId } = useMeta();
  const [model, setModel] = useState<string | null>(null);
  const [attrSel, setAttr] = useState<string | null>(null);
  const [policy, setPolicy] = useState<string>("equal_opportunity");
  const [t, setT] = useState<number | null>(null);
  const [all, setAll] = useState<Record<string, Result>>({});
  const [err, setErr] = useState<string | null>(null);
  const M = model ?? oursId ?? order[0];
  const attr = attrSel ?? meta?.info.group_attrs[0] ?? "";
  const T = t ?? meta?.threshold ?? null;
  const Td = useDebounced(T, 250);            // slider: one request burst after the user stops dragging
  const e = M && isPrivate(M) ? `&eps=${eps}` : "";
  const { data: sweep } = useApi<Sweep>(M && attr ? `/api/sweep?method=${M}${e}&attr=${attr}` : null);

  useEffect(() => {
    if (!M || !meta || Td == null || !attr) return;
    let live = true;                          // ignore responses from superseded requests
    setErr(null); setAll({});
    Promise.all(POLICIES.map((p) => get<Result>(`/api/mitigation?method=${M}${e}&policy=${p.v}&attr=${attr}&t=${Td}`)))
      .then((rs) => { if (live) setAll(Object.fromEntries(rs.map((r) => [r.policy, r]))); })
      .catch((x) => { if (live) setErr(String(x.message)); });
    return () => { live = false; };
  }, [M, e, attr, Td, meta]);

  if (error) return <ApiError error={error} />;
  if (!meta || T == null) return <Loading />;
  const r = all[policy];
  const grid = meta.threshold_grid;
  const four = meta.scorecard.four_fifths_ratio;
  const attrs = [...meta.info.group_attrs, "intersection"];

  return (
    <div className="grid">
      <Card title="Post-processing mitigation" hint="Change only the decision thresholds of an already-trained model. Thresholds are fitted on validation data; the model and its privacy guarantee stay unchanged (post-processing)."
        right={<div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <select aria-label="model" value={M} onChange={(x) => setModel(x.target.value)} style={{ padding: "6px 8px", border: "1px solid var(--rule)", borderRadius: 8 }}>{order.map((m) => <option key={m} value={m}>{short(m)}{isPrivate(m) ? ` (ε = ${eps})` : ""}</option>)}</select>
          <Seg label="protected attribute" value={attr} onChange={setAttr} options={attrs.map((a) => ({ v: a, l: a === "intersection" ? "Intersection" : a[0].toUpperCase() + a.slice(1) }))} /></div>}>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "end" }}>
          <div className="field" role="group" aria-label="Fairness policy"><span className="lbl">Fairness policy</span><Seg label="policy" value={policy} onChange={setPolicy} options={POLICIES.map((p) => ({ v: p.v, l: p.l }))} /></div>
          <div className="field" style={{ width: 240 }}><label htmlFor="thr">Base threshold: <b className="num">{T.toFixed(2)}</b></label><input id="thr" type="range" min={grid.min} max={grid.max} step={grid.step} value={T} onChange={(x) => setT(Number(x.target.value))} /></div>
        </div>
        <p className="note">{POLICIES.find((p) => p.v === policy)?.d}</p>
        {err && <div className="callout bad">{err}</div>}
      </Card>

      {r && (
        <div className="grid g5">
          <Kpi k="Accuracy" v={pct(r.summary.accuracy)} />
          <Kpi k="Worst-group recall" v={pct(r.summary.intersection_worst_tpr)} />
          <Kpi k={`Disparate-impact ratio (${attr})`} v={f3(r.ethics[attr]?.disparate_impact_ratio, 2)} d={<Pass v={r.ethics[attr]?.four_fifths_pass} />} />
          <Kpi k={`Equal-opportunity diff. (${attr})`} v={pct(r.ethics[attr]?.equal_opportunity_diff)} />
          <Kpi k={`Calibration gap (${attr})`} v={pct(r.ethics[attr]?.calibration_gap)} />
        </div>
      )}

      <div className="grid">
        <Card title="The three policies side by side" hint={`Same model, same data, base threshold ${T.toFixed(2)}. No policy wins on every criterion: the fairness impossibility theorem in practice.`}>
          <div className="tbl">
            <table>
              <thead><tr><th>Policy</th><th className="n">Accuracy</th><th className="n">Worst-group recall</th><th className="n">DP diff.</th><th className="n">DI ratio</th><th>Four-fifths</th><th className="n">EO diff.</th><th className="n">Calibration gap</th></tr></thead>
              <tbody>{POLICIES.map((p) => { const x = all[p.v]; const a = x?.ethics[attr]; return (
                <tr key={p.v} className={p.v === policy ? "me" : ""}><td>{p.l}</td><td className="n">{pct(x?.summary.accuracy)}</td><td className="n">{pct(x?.summary.intersection_worst_tpr)}</td>
                  <td className="n">{pct(a?.demographic_parity_diff)}</td><td className="n">{f3(a?.disparate_impact_ratio, 2)}</td><td><Pass v={a?.four_fifths_pass} /></td><td className="n">{pct(a?.equal_opportunity_diff)}</td><td className="n">{pct(a?.calibration_gap)}</td></tr>); })}</tbody>
            </table>
          </div>
        </Card>
        <Card title="Per-group thresholds and outcomes" hint={`Policy: ${POLICIES.find((p) => p.v === policy)?.l}. Groups without enough validation data keep the base threshold.`}>
          {!r ? <Loading /> : (
            <div className="tbl" style={{ maxHeight: 330, overflowY: "auto" }}>
              <table>
                <thead><tr><th>Group</th><th className="n">Threshold</th><th className="n">Selection rate</th><th className="n">Recall</th><th className="n">False-positive rate</th></tr></thead>
                <tbody>{r.ethics[attr].groups.filter((g) => g.eligible).map((g) => (
                  <tr key={g.group}><td>{cellLabel(g.group)}</td><td className="n">{f3(r.thresholds[g.group] ?? r.thresholds.__default__, 2)}</td><td className="n">{pct(g.selection_rate)}</td><td className="n">{pct(g.tpr)}</td><td className="n">{pct(g.fpr)}</td></tr>))}</tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <Card title={`Threshold sweep (one threshold for all, ${attr})`} hint="How accuracy and fairness criteria move as the single decision threshold changes.">
        {!sweep ? <Loading /> : (
          <div style={{ height: 340 }}>
            <ResponsiveContainer>
              <LineChart data={sweep} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => v.toFixed(1)} label={{ value: "threshold", position: "insideBottomRight", offset: -2, fontSize: 11 }} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
                <Tooltip formatter={fmt3} labelFormatter={(v) => `threshold ${v}`} />
                <Legend itemSorter={() => 0} />
                <ReferenceLine x={T} stroke="var(--ink)" strokeDasharray="3 3" />
                <ReferenceLine y={four} stroke="var(--bad)" strokeDasharray="2 4" label={{ value: `four-fifths (${four})`, position: "insideTopRight", fontSize: 11, fill: "var(--bad)" }} />
                <Line dataKey="accuracy" name="Accuracy" stroke="#2a78d6" strokeWidth={2.5} dot={false} />
                <Line dataKey="worst_tpr" name="Worst-group recall" stroke="#6250d6" strokeWidth={2.5} dot={false} />
                <Line dataKey="di_ratio" name="Disparate-impact ratio" stroke="#eb6834" strokeWidth={2.5} dot={false} />
                <Line dataKey="eo_diff" name="Equal-opportunity diff." stroke="#1baf7a" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
    </div>
  );
}
