"use client";
import { useApi, useMeta } from "@/lib/api";
import { color } from "@/lib/models";
import { ApiError, Card, Loading, ModelName } from "./ui";

interface Check { principle: string; check: string; value: number | string | null; pass: boolean | null }
interface SC { thresholds: Record<string, number>; models: { key: string; method: string; name: string; eps: number | null; checks: Check[] }[] }

const fmtV = (v: Check["value"]) => (v == null ? "–" : typeof v === "number" ? v.toFixed(3) : v);

export default function Scorecard({ eps }: { eps: number }) {
  const { data, error } = useApi<SC>(`/api/scorecard?eps=${eps}`);
  const { short, oursId } = useMeta();
  if (error) return <ApiError error={error} />;
  if (!data) return <Loading />;
  const checks = data.models[0]?.checks.map((c) => `${c.principle}|${c.check}`) ?? [];
  const principles = [...new Set(data.models[0]?.checks.map((c) => c.principle) ?? [])];
  const score = (m: SC["models"][number]) => { const s = m.checks.filter((c) => c.pass != null); return { p: s.filter((c) => c.pass).length, n: s.length }; };

  return (
    <div className="grid">
      <div className="grid g5">
        {data.models.map((m) => { const s = score(m); return (
          <div key={m.key} className="card kpi" style={{ borderTop: `4px solid ${color(m.method)}`, background: m.method === oursId ? "var(--brand-soft)" : undefined }}>
            <span className="k"><ModelName id={m.method} name={short(m.method)} /></span>
            <span className="v display">{s.p}<span className="muted" style={{ fontSize: 18 }}> / {s.n}</span></span>
            <span className="d">checks passed {m.eps ? `at ε = ${m.eps}` : "(no privacy)"}</span>
            <div className="pbar" style={{ ["--t" as string]: "100%" }}><span style={{ width: `${(s.p / Math.max(1, s.n)) * 100}%`, background: color(m.method) }} /></div>
          </div>); })}
      </div>
      <Card title={<>Responsible-AI scorecard at ε = {eps}</>} hint={`Pass/fail thresholds from server/config.yaml: ${Object.entries(data.thresholds).map(([k, v]) => `${k.replaceAll("_", " ")} ${v}`).join(" · ")}.`}>
        <div className="tbl">
          <table>
            <thead><tr><th>Principle</th><th>Check</th>{data.models.map((m) => <th key={m.key} style={{ whiteSpace: "normal" }}>{short(m.method)}</th>)}</tr></thead>
            <tbody>
              {principles.flatMap((p) => checks.filter((c) => c.startsWith(`${p}|`)).map((c, i) => (
                <tr key={c}>
                  {i === 0 && <td rowSpan={checks.filter((x) => x.startsWith(`${p}|`)).length} style={{ fontWeight: 600, verticalAlign: "top" }}>{p}</td>}
                  <td>{c.split("|")[1]}</td>
                  {data.models.map((m) => { const x = m.checks.find((k) => `${k.principle}|${k.check}` === c); return (
                    <td key={m.key} className={`verdict ${x?.pass == null ? "na" : x.pass ? "ok" : "bad"}`}><span className="stamp">{x?.pass == null ? "info" : x.pass ? "pass" : "fail"}</span> <span className="num muted">{fmtV(x?.value ?? null)}</span></td>); })}
                </tr>)))}
            </tbody>
          </table>
        </div>
        <p className="note">Failing a check is a finding, not a bug: it shows where a model needs mitigation (Mitigation tab) or should not be deployed. Fairness checks apply to sex and race separately.</p>
      </Card>
    </div>
  );
}
