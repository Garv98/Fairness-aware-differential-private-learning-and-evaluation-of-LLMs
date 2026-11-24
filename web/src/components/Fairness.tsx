"use client";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { Evaluation, GroupRow, key, useApi, useMeta } from "@/lib/api";
import { cellLabel, color, f3, pct } from "@/lib/models";
import { ApiError, Card, Loading, ModelName, Pass, Seg, fmtPct } from "./ui";

const METRICS: { v: keyof GroupRow; l: string; hint: string }[] = [
  { v: "tpr", l: "Recall (TPR)", hint: "share of true positives the model identifies (higher is better)" },
  { v: "fpr", l: "False-positive rate", hint: "share of true negatives wrongly predicted positive" },
  { v: "selection_rate", l: "Selection rate", hint: "share predicted positive (demographic parity compares this)" },
  { v: "accuracy", l: "Accuracy", hint: "share of correct decisions" },
];
const SEQ = ["var(--seq-0)", "var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)", "var(--seq-5)"];

export default function Fairness({ eps }: { eps: number }) {
  const { data: ev, error } = useApi<Evaluation>("/api/evaluation");
  const { meta, order, privateIds, short, isPrivate, oursId } = useMeta();
  const [attrSel, setAttr] = useState<string | null>(null);
  const [metric, setMetric] = useState<keyof GroupRow>("tpr");
  const [suiteSel, setSuiteAttr] = useState<string | null>(null);
  if (error) return <ApiError error={error} />;
  if (!ev || !meta) return <Loading what="evaluation" />;

  const info = meta.info, attrs = [...info.group_attrs, "intersection"];
  const attr = attrSel ?? info.group_attrs[1] ?? info.group_attrs[0], suiteAttr = suiteSel ?? info.group_attrs[0];
  const at = (m: string) => ev[key(m, isPrivate(m) ? eps : null)];
  const models = order.filter((m) => at(m)?.ethics?.[suiteAttr] && at(m)?.groups?.[attr]);
  const ref = order.find((m) => !isPrivate(m) && at(m));
  if (!ref || models.length === 0) return <Loading what="evaluation for this budget" />;
  const small = (g: GroupRow) => g.n < info.min_support || g.n_pos < info.min_positives;
  const val = (m: string, a: string, g: string, k: keyof GroupRow = metric) => (at(m)?.groups[a].find((r) => r.group === g)?.[k] as number | null) ?? null;
  const M = METRICS.find((x) => x.v === metric)!;
  const groups = at(ref).groups[attr].filter((g) => attr !== "intersection" || !small(g)).map((g) => g.group);
  const bars = groups.map((g) => ({ group: cellLabel(g), ...Object.fromEntries(models.map((m) => [m, val(m, attr, g)])) }));
  const cells = [...at(ref).groups.intersection].sort((a, b) => b.n_pos - a.n_pos);
  const costRows = at(ref).groups.intersection.filter((g) => !small(g)).sort((a, b) => b.n_pos - a.n_pos).map((g) => ({
    group: cellLabel(g.group),
    ...Object.fromEntries(privateIds.map((m) => { const s = val(ref, "intersection", g.group, "tpr"), p = val(m, "intersection", g.group, "tpr"); return [m, s != null && p != null ? p - s : null]; })),
  }));
  const calibAll = models.flatMap((m) => (at(m).ethics[suiteAttr]?.groups ?? []).filter((g) => g.eligible).map((g) => ({ m, group: g.group, x: g.base_rate, y: g.mean_score, n: g.n })));
  const calib = calibAll;
  const dom = Math.min(1, Math.ceil(Math.max(0.1, ...calibAll.flatMap((c) => [c.x, c.y])) * 10) / 10);

  return (
    <div className="grid">
      <Card title={<>Fairness metrics suite at ε = {eps}</>} hint={`Group-level criteria from the fairness literature, for one attribute. Groups below ${info.min_support} people / ${info.min_positives} positives excluded. Four-fifths rule: lowest selection rate ≥ ${meta.scorecard.four_fifths_ratio} × highest (EEOC).`}
        right={<Seg label="attribute" value={suiteAttr} onChange={setSuiteAttr} options={attrs.map((a) => ({ v: a, l: a === "intersection" ? "Intersection" : a[0].toUpperCase() + a.slice(1) }))} />}>
        <div className="tbl">
          <table>
            <thead><tr><th>Model</th><th className="n">Demographic parity diff.</th><th className="n">Disparate-impact ratio</th><th>Four-fifths rule</th><th className="n">Equal opportunity diff.</th><th className="n">Equalized odds diff.</th><th className="n">Predictive parity diff.</th><th className="n">Calibration gap</th><th className="n">Individual consistency</th></tr></thead>
            <tbody>{models.map((m) => { const e = at(m).ethics[suiteAttr]; return (
              <tr key={m} className={m === oursId ? "me" : ""}>
                <td><ModelName id={m} name={short(m)} /></td>
                <td className="n">{pct(e.demographic_parity_diff)}</td><td className="n">{f3(e.disparate_impact_ratio, 2)}</td><td><Pass v={e.four_fifths_pass} /></td>
                <td className="n">{pct(e.equal_opportunity_diff)}</td><td className="n">{pct(e.equalized_odds_diff)}</td><td className="n">{pct(e.predictive_parity_diff)}</td>
                <td className="n">{pct(e.calibration_gap)}</td><td className="n">{f3(at(m).ethics.consistency)}</td>
              </tr>); })}</tbody>
          </table>
        </div>
        <p className="note">These criteria conflict: when base rates differ between groups, a model cannot satisfy demographic parity, equalized odds and calibration at once (Kleinberg et al. 2016; Chouldechova 2017). The Mitigation tab shows this trade-off live.</p>
      </Card>

      <Card title={<>{M.l} by group at ε = {eps}</>} hint={M.hint}
        right={<div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}><Seg label="attribute" value={attr} onChange={setAttr} options={attrs.map((a) => ({ v: a, l: a === "intersection" ? `${info.group_attrs.join(" × ")}` : a[0].toUpperCase() + a.slice(1) }))} /><Seg label="metric" value={metric} onChange={setMetric} options={METRICS.map((x) => ({ v: x.v, l: x.l }))} /></div>}>
        <div style={{ height: attr === "intersection" ? 380 : 320 }}>
          <ResponsiveContainer>
            <BarChart data={bars} margin={{ top: 6, right: 8, left: 0, bottom: attr === "intersection" ? 60 : 0 }} barGap={1}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="group" interval={0} angle={attr === "intersection" ? -35 : 0} textAnchor={attr === "intersection" ? "end" : "middle"} height={attr === "intersection" ? 80 : 30} tick={{ fontSize: 11.5 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip formatter={fmtPct} />
              <Legend itemSorter={() => 0} verticalAlign="top" height={30} />
              {models.map((m) => <Bar key={m} dataKey={m} name={short(m)} fill={color(m)} radius={[3, 3, 0, 0]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
        {attr === "intersection" && <p className="note">Only groups with ≥ {info.min_support} people and ≥ {info.min_positives} positives are shown; all {info.n_cells} groups are in the heatmap below.</p>}
      </Card>

      <div className="grid g2">
        <Card title="Cost of privacy for each group" hint={<>Recall change vs {short(ref)} at ε = {eps}. Below 0 = the group lost recall to privacy.</>}>
          <div style={{ height: 420 }}>
            <ResponsiveContainer>
              <BarChart data={costRows} layout="vertical" margin={{ left: 10, right: 16 }} barGap={1}>
                <CartesianGrid horizontal={false} stroke="var(--grid)" />
                <XAxis type="number" tickFormatter={(v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}`} />
                <YAxis type="category" dataKey="group" width={150} tick={{ fontSize: 11.5 }} />
                <ReferenceLine x={0} stroke="var(--ink-2)" />
                <Tooltip formatter={(v) => (typeof v === "number" ? `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)} pts` : String(v))} />
                <Legend itemSorter={() => 0} verticalAlign="top" height={28} />
                {privateIds.map((m) => <Bar key={m} dataKey={m} name={short(m)} fill={color(m)} radius={2} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Calibration by group" hint={`Does a predicted score mean the same thing for every ${suiteAttr === "intersection" ? "group" : suiteAttr} group? Points on the diagonal = calibrated. x = actual rate, y = mean predicted score.`}>
          <div style={{ height: 420 }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 10, right: 16, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis type="number" dataKey="x" domain={[0, dom]} tickFormatter={(v) => `${Math.round(v * 100)}%`} name="actual rate" />
                <YAxis type="number" dataKey="y" domain={[0, dom]} tickFormatter={(v) => `${Math.round(v * 100)}%`} name="mean score" />
                <ZAxis type="number" dataKey="n" range={[40, 260]} name="people" />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: dom, y: dom }]} stroke="var(--ink-3)" strokeDasharray="4 4" />
                <Tooltip formatter={(v, n) => (n === "people" ? String(v) : fmtPct(v))} labelFormatter={() => ""} />
                <Legend itemSorter={() => 0} />
                {models.map((m) => <Scatter key={m} name={short(m)} data={calib.filter((c) => c.m === m)} fill={color(m)} />)}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Recall heatmap: every intersectional group" hint="Darker = higher recall. Faded rows are too small for reliable estimates.">
        <div className="tbl">
          <table>
            <thead><tr><th>Group</th><th className="n">People</th><th className="n">Positives</th>{models.map((m) => <th key={m} className="n" style={{ whiteSpace: "normal" }}>{short(m)}</th>)}</tr></thead>
            <tbody>
              {cells.map((g) => (
                <tr key={g.group} style={{ opacity: small(g) ? 0.45 : 1 }}>
                  <td style={{ fontSize: 12.5 }}>{cellLabel(g.group)}</td><td className="n">{g.n}</td><td className="n">{g.n_pos}</td>
                  {models.map((m) => {
                    const v = val(m, "intersection", g.group, "tpr"), k = v == null ? -1 : Math.min(5, Math.floor(v * 6));
                    return <td key={m} className="n" style={{ background: k < 0 ? "var(--grid)" : SEQ[k], color: k >= 3 ? "#fff" : "var(--ink)", padding: "6px 8px" }}>{v == null ? "–" : (v * 100).toFixed(0)}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
