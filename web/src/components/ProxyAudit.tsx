"use client";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi, useMeta } from "@/lib/api";
import { f3, label, pct } from "@/lib/models";
import { ApiError, Card, Loading, Pass, fmt3 } from "./ui";

interface Audit { auc: number; positive: string; base_rate: number; top_features: { field: string; mutual_info: number }[] }
interface Side { summary: Record<string, number | null>; ethics: Record<string, { disparate_impact_ratio: number | null; four_fifths_pass: boolean | null; equal_opportunity_diff: number | null; equalized_odds_diff: number | null }> }
interface Proxy { audits: Record<string, Audit>; blind: Side; aware: Side | null; removed: string[] }

function verdict(auc: number, th: { strong: number; moderate: number }) {
  if (auc >= th.strong) return { t: "strong proxy", c: "bad" };
  if (auc >= th.moderate) return { t: "moderate proxy", c: "warn" };
  return { t: "weak proxy", c: "ok" };
}

export default function ProxyAudit() {
  const { data, error } = useApi<Proxy>("/api/proxy");
  const { meta, short, order, isPrivate } = useMeta();
  if (error) return <ApiError error={error} />;
  if (!data || !meta) return <Loading />;
  const ref = order.find((m) => !isPrivate(m));
  const rows: [string, (s: Side) => number | null | undefined, boolean][] = [
    ["Accuracy", (s) => s.summary.accuracy, true],
    ["Worst-group recall", (s) => s.summary.intersection_worst_tpr, true],
    ...data.removed.flatMap((a) => [
      [`Disparate-impact ratio (${a})`, (s: Side) => s.ethics[a]?.disparate_impact_ratio, true] as [string, (s: Side) => number | null | undefined, boolean],
      [`Equal-opportunity diff. (${a})`, (s: Side) => s.ethics[a]?.equal_opportunity_diff, false] as [string, (s: Side) => number | null | undefined, boolean],
    ]),
  ];
  const blindWorst = data.blind.summary.intersection_worst_tpr, awareWorst = data.aware?.summary.intersection_worst_tpr;

  return (
    <div className="grid">
      <div className="callout">Can we make a model fair by hiding sex and race from it? This audit checks whether the remaining features still reveal them (proxies), and trains the same model with {data.removed.join(" and ")} removed.</div>
      <div className="grid g2">
        {Object.entries(data.audits).map(([a, au]) => {
          const v = verdict(au.auc, meta.proxy_strength);
          return (
            <Card key={a} title={<>Can the other features reveal <b>{a}</b>?</>} hint={`Cross-validated logistic regression predicting ${a} = ${au.positive} from all non-sensitive features (base rate ${pct(au.base_rate)}).`}
              right={<span className={`badge ${v.c}`}>{v.t}</span>}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}><span className="display" style={{ fontSize: 40, fontWeight: 700 }}>AUC {f3(au.auc)}</span><span className="muted">0.5 = no information, 1.0 = fully revealed</span></div>
              <div style={{ height: 220, marginTop: 8 }}>
                <ResponsiveContainer>
                  <BarChart data={au.top_features.map((f) => ({ name: label(f.field), mi: f.mutual_info }))} layout="vertical" margin={{ left: 20, right: 20 }}>
                    <CartesianGrid horizontal={false} stroke="var(--grid)" />
                    <XAxis type="number" tickFormatter={(x) => x.toFixed(2)} />
                    <YAxis type="category" dataKey="name" width={120} />
                    <Tooltip formatter={fmt3} />
                    <Bar dataKey="mi" name="mutual information (nats)" fill="var(--brand)" radius={4} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="note">Features carrying the most information about {a}.</p>
            </Card>
          );
        })}
      </div>
      {data.aware && (
        <Card title="Fairness through unawareness: does removing the attributes help?" hint={`Same architecture and training as ${ref ? short(ref) : "the standard model"}, once with and once without ${data.removed.join(" and ")} as inputs.`}
          right={blindWorst != null && awareWorst != null ? <span className={`badge ${blindWorst > awareWorst ? "ok" : "bad"}`}>{blindWorst > awareWorst ? "removal improved worst-group recall" : "removal did not improve worst-group recall"}</span> : null}>
          <div className="tbl">
            <table>
              <thead><tr><th>Metric</th><th className="n">With {data.removed.join(" & ")}</th><th className="n">Without (blind)</th><th className="n">Change</th></tr></thead>
              <tbody>{rows.map(([n, f, higherBetter]) => {
                const x = f(data.aware!), y = f(data.blind), d = x != null && y != null ? y - x : null;
                const good = d == null ? null : higherBetter ? d > 0 : d < 0;
                return <tr key={n}><td>{n}</td><td className="n">{n.startsWith("Disparate") ? f3(x, 2) : pct(x)}</td><td className="n">{n.startsWith("Disparate") ? f3(y, 2) : pct(y)}</td>
                  <td className="n" style={{ color: good == null ? undefined : good ? "var(--ok)" : "var(--bad)" }}>{d == null ? "–" : `${d > 0 ? "+" : ""}${n.startsWith("Disparate") ? d.toFixed(2) : (d * 100).toFixed(1) + " pts"}`}</td></tr>;
              })}
              {data.removed.map((a) => <tr key={`ff${a}`}><td>Four-fifths rule ({a})</td><td className="n"><Pass v={data.aware!.ethics[a]?.four_fifths_pass} /></td><td className="n"><Pass v={data.blind.ethics[a]?.four_fifths_pass} /></td><td /></tr>)}
              </tbody>
            </table>
          </div>
          <p className="note">If the other features predict the attribute well (high AUC above), the model can still discriminate through them, and removing the attribute also removes the ability to measure and correct the bias.</p>
        </Card>
      )}
    </div>
  );
}
