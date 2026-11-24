"use client";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi, useMeta } from "@/lib/api";
import { pct } from "@/lib/models";
import { ApiError, Card, Kpi, Loading, Seg, fmtPct } from "./ui";

type Row = Record<string, number | string | null>;
interface Credit { performance: Row[]; gaps: Row[]; dp_levels: Row[]; groups: Record<string, Record<string, Row[]>>; test_size: number | null }
// standard = blue; DP noise levels = one orange ramp (darker = more noise)
const RAMP = ["#2a78d6", "#f3a37f", "#eb6834", "#a8431a", "#6e2a10"];

export default function CreditStudy() {
  const { data, error } = useApi<Credit>("/api/credit");
  const { meta } = useMeta();
  const [attr, setAttr] = useState<string | null>(null);
  if (error) return <ApiError error={error} />;
  if (!data || !meta) return <Loading />;
  if (Object.keys(data.groups).length === 0) return <div className="callout warn">No credit-study results found in results/.</div>;
  const A = attr ?? Object.keys(data.groups)[0];
  const models = Object.keys(data.groups[A]);
  const groups = (data.groups[A][models[0]] ?? []).map((r) => String(r.Group));
  const chart = groups.map((g) => ({ group: g, ...Object.fromEntries(models.map((m) => [m, Number(data.groups[A][m].find((r) => String(r.Group) === g)?.Recall ?? NaN)])) }));
  const std = data.performance.find((r) => String(r.Model).startsWith("Standard")), dp = data.performance.find((r) => String(r.Model).startsWith("DP"));
  const g0 = data.gaps.find((r) => String(r.Model).startsWith("Standard")), g1 = data.gaps.find((r) => String(r.Model).startsWith("DP"));
  const gapKeys = Object.keys(g0 ?? {}).filter((k) => k.endsWith("_Recall_Gap_pp"));
  const small = (m: string) => (data.groups[A][m] ?? []).filter((r) => Number(r.Samples) < meta.info.min_support).map((r) => `${r.Group} (n=${r.Samples})`);

  return (
    <div className="grid">
      <div className="callout">First workstream: credit-risk data, predict loan default. Test set: {data.test_size?.toLocaleString() ?? "–"} applicants. Groups: {Object.keys(data.groups).join(", ").toLowerCase()} (the data has no race or sex fields). Results as exported by the credit notebooks (single seed).</div>
      <div className="grid g4">
        <Kpi k={`Accuracy: standard → DP${dp?.Epsilon != null ? ` (ε = ${Number(dp.Epsilon).toFixed(2)})` : ""}`} v={`${pct(Number(std?.Accuracy))} → ${pct(Number(dp?.Accuracy))}`} />
        <Kpi k="Recall: standard → DP" v={`${pct(Number(std?.Recall))} → ${pct(Number(dp?.Recall))}`} d="share of defaulters caught" />
        {gapKeys.slice(0, 2).map((k) => <Kpi key={k} k={`${k.split("_")[0]} recall gap (best − worst group)`} v={`${Number(g0?.[k]).toFixed(1)} → ${Number(g1?.[k]).toFixed(1)} pp`} color="var(--bad)" />)}
      </div>
      <Card title="Recall by group: standard vs DP at each noise level" hint="Share of real defaulters caught in each group. More noise = stronger privacy (lower ε)."
        right={<Seg label="attribute" value={A} onChange={setAttr} options={Object.keys(data.groups).map((a) => ({ v: a, l: a }))} />}>
        <div style={{ height: 340 }}>
          <ResponsiveContainer>
            <BarChart data={chart} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="group" />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip formatter={fmtPct} />
              <Legend itemSorter={() => 0} />
              {models.map((m, k) => <Bar key={m} dataKey={m} fill={RAMP[k % RAMP.length]} radius={[3, 3, 0, 0]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
        {small(models[0]).length > 0 && <p className="note">Unstable (fewer than {meta.info.min_support} people): {small(models[0]).join(", ")}.</p>}
      </Card>
      <Card title="Privacy–utility across noise levels">
        <div className="tbl">
          <table>
            <thead><tr>{Object.keys(data.dp_levels[0] ?? {}).map((c) => <th key={c} className="n">{c.replaceAll("_", " ").replace(" pp", " (pp)")}</th>)}</tr></thead>
            <tbody>{data.dp_levels.map((r, i) => <tr key={i}>{Object.entries(r).map(([c, v]) => <td key={c} className="n">{typeof v === "number" ? (c.endsWith("_pp") || c === "Noise_Level" || c === "Epsilon" ? v.toFixed(2) : pct(v)) : String(v)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
