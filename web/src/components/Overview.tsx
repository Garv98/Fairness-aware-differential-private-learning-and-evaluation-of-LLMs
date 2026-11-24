"use client";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Evaluation, key, useApi, useMeta } from "@/lib/api";
import { color, epsLabel, f3, pct } from "@/lib/models";
import { ApiError, Card, Kpi, Loading, ModelName, fmtPct } from "./ui";

const METRICS = [["accuracy", "Accuracy"], ["balanced_accuracy", "Balanced acc."], ["recall", "Recall (all)"], ["intersection_worst_tpr", "Worst-group recall"]] as const;

export default function Overview({ eps }: { eps: number }) {
  const { data: ev, error } = useApi<Evaluation>("/api/evaluation");
  const { meta, order, privateIds, short, isPrivate, oursId, baseId: base, epsFor } = useMeta();
  if (error) return <ApiError error={error} />;
  if (!ev || !meta) return <Loading what="evaluation" />;

  const info = meta.info;
  const at = (m: string) => ev[key(m, isPrivate(m) ? eps : null)];
  const models = order.filter((m) => at(m));
  const S = (m: string, k: string) => at(m)?.summary[k] ?? null;
  const cost = (m: string) => at(m)?.cost_of_privacy?.intersection?.di_tpr_max ?? null;
  const nonPriv = order.filter((m) => !isPrivate(m));
  const compare = METRICS.map(([k, l]) => ({ metric: l, ...Object.fromEntries(models.map((m) => [m, S(m, k)])) }));
  const trend = meta.epsilons.map((e) => ({ eps: `ε=${e}`, ...Object.fromEntries(privateIds.map((m) => [m, ev[key(m, e)]?.summary.intersection_worst_tpr ?? null])) }));
  const aux = meta.methods.find((m) => m.id === oursId)?.aux_frac ?? 0;
  const pipe: [string, string][] = [
    ["Data", `${info.dataset} · ${info.dataset_rows.toLocaleString()} people · ${info.group_attrs.join(" × ")} = ${info.n_cells} groups`],
    ["Split", `${info.train_size.toLocaleString()} train · ${info.val_size.toLocaleString()} validation · ${info.test_size.toLocaleString()} test (seed ${info.seed})`],
    ["DP-SGD", `clip each person's gradient at ${info.dp.max_grad_norm}, add Gaussian noise; RDP accountant fixes ε (δ = ${info.delta.toExponential(1)})`],
    ["IG reweighting", `${Math.round(aux * 100)}% of the same ε buys noisy group statistics → upweight the worst groups`],
    ["Model", `MLP ${info.hidden.join("→")}→1 · ${info.epochs} epochs · ${info.n_features} features`],
    ["Audit", "fairness metrics, recourse, proxy audit, membership attack, model cards"],
  ];

  return (
    <div className="grid">
      <div className="grid g4">
        <Kpi k="Models trained & served" v={meta.models.length} d={`${nonPriv.length} without privacy · ${privateIds.length} private methods × ${meta.epsilons.length} budgets`} />
        <Kpi k={<>Worst-group recall at ε = {eps}</>} v={<span>{pct(S(base, "intersection_worst_tpr"), 0)} → <span style={{ color: oursId ? color(oursId) : undefined }}>{pct(oursId ? S(oursId, "intersection_worst_tpr") : null, 0)}</span></span>} d={`${short(base)} → ${oursId ? short(oursId) : "–"}`} />
        <Kpi k="Recall lost by the worst-hit group vs no privacy" v={<span>{pct(cost(base), 0)} → <span style={{ color: oursId ? color(oursId) : undefined }}>{pct(oursId ? cost(oursId) : null, 0)}</span></span>} d={`${short(base)} → ${oursId ? short(oursId) : "–"} (cost of privacy)`} />
        <Kpi k="Membership-attack AUC (0.5 = coin flip)" v={f3(oursId ? at(oursId)?.mia.mia_auc : null)} d={`${oursId ? short(oursId) : ""} at ε = ${eps}`} />
      </div>

      <Card title="System pipeline" hint="Every number below is computed from the trained models, not typed in.">
        <div className="pipe">
          {pipe.map(([h, b], i) => (
            <div key={h} className={`pipe-step ${i === 2 || i === 3 ? "hl" : ""}`}>
              <div className="pipe-h">{h}</div><div className="pipe-b">{b}</div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid g2">
        <Card title={<>Model comparison at ε = {eps}</>} hint="Non-private models shown for reference (ε = ∞).">
          <div style={{ height: 310 }}>
            <ResponsiveContainer>
              <BarChart data={compare} margin={{ top: 6, right: 8, left: 0, bottom: 0 }} barGap={2}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="metric" />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip formatter={fmtPct} />
                <Legend itemSorter={() => 0} />
                {models.map((m) => <Bar key={m} dataKey={m} name={short(m)} fill={color(m)} radius={[3, 3, 0, 0]} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Worst-group recall vs privacy budget" hint="Lower ε = stronger privacy. Dashed lines: models without privacy.">
          <div style={{ height: 310 }}>
            <ResponsiveContainer>
              <LineChart data={trend} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="eps" />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip formatter={fmtPct} />
                <Legend itemSorter={() => 0} />
                {nonPriv.map((m) => S(m, "intersection_worst_tpr") != null && <ReferenceLine key={m} y={S(m, "intersection_worst_tpr")!} stroke={color(m)} strokeDasharray="5 4" label={{ value: short(m), position: "insideTopLeft", fill: color(m), fontSize: 11 }} />)}
                {privateIds.map((m) => <Line key={m} dataKey={m} name={short(m)} stroke={color(m)} strokeWidth={2.5} dot={{ r: 5 }} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="All served models" hint={`Test set, threshold ${meta.threshold}. Worst-group = lowest recall among the ${info.n_cells} intersectional groups with ≥ ${info.min_support} people and ≥ ${info.min_positives} positives.`}>
        <div className="tbl">
          <table>
            <thead><tr><th>Model</th><th className="n">ε</th><th className="n">Accuracy</th><th className="n">Balanced acc.</th><th className="n">Recall</th><th className="n">Worst-group recall</th><th className="n">Group recall gap</th><th className="n">Cost of privacy</th><th className="n">Consistency</th><th className="n">Attack AUC</th></tr></thead>
            <tbody>
              {order.flatMap((m) => epsFor(m).map((e) => ev[key(m, e)]).filter(Boolean).map((x) => (
                <tr key={key(x.method, x.eps)} className={x.method === oursId && x.eps === eps ? "me" : ""}>
                  <td><ModelName id={x.method} name={x.name} /></td>
                  <td className="n">{epsLabel(x.eps)}</td>
                  <td className="n">{pct(x.summary.accuracy)}</td>
                  <td className="n">{pct(x.summary.balanced_accuracy)}</td>
                  <td className="n">{pct(x.summary.recall)}</td>
                  <td className="n"><b>{pct(x.summary.intersection_worst_tpr)}</b></td>
                  <td className="n">{pct(x.summary.intersection_tpr_gap)}</td>
                  <td className="n">{x.private ? pct(x.cost_of_privacy?.intersection?.di_tpr_max) : "–"}</td>
                  <td className="n">{f3(x.ethics.consistency)}</td>
                  <td className="n">{f3(x.mia.mia_auc)}</td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
