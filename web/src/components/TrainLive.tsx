"use client";
import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { get, post, useMeta } from "@/lib/api";
import { color, f3, pct } from "@/lib/models";
import { ApiError, Card, Kpi, Loading, ModelName, Pass, Seg } from "./ui";

interface Ep { epoch: number; accuracy: number; balanced_accuracy: number; recall: number; intersection_worst_tpr: number; eps_spent?: number; loss?: number }
interface Job { id: string; status: "running" | "done" | "error"; method: string; name: string; private: boolean; eps: number | null; epochs_total: number; epochs: Ep[];
  noise?: Record<string, number>; final?: Record<string, number>; final_ethics?: Record<string, { disparate_impact_ratio: number | null; four_fifths_pass: boolean | null; equal_opportunity_diff: number | null }>; error?: string }

export default function TrainLive() {
  const { meta, error, order, short, isPrivate, oursId } = useMeta();
  const [method, setMethod] = useState<string | null>(null);
  const [eps, setEps] = useState<number | null>(null);
  const [epochs, setEpochs] = useState<number | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [done, setDone] = useState<Job[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [metric, setMetric] = useState<keyof Ep>("intersection_worst_tpr");
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);
  if (error) return <ApiError error={error} />;
  if (!meta) return <Loading />;

  const defaultEps = meta.train_epsilons.includes(meta.epsilons[0]) ? meta.epsilons[0] : meta.train_epsilons[0];
  const M = method ?? oursId ?? order[0], E = eps ?? defaultEps, N = epochs ?? meta.train_epochs.default;
  const start = async () => {
    if (starting) return;
    setErr(null); setStarting(true);
    if (timer.current) clearInterval(timer.current);
    try {
      const { id } = await post<{ id: string }>("/api/train", { method: M, eps: E, epochs: N });
      let inFlight = false;
      timer.current = setInterval(async () => {
        if (inFlight) return;                 // never overlap polls
        inFlight = true;
        try {
          const j = await get<Job>(`/api/train/${id}`);
          setJob(j);
          if (j.status !== "running") {
            if (timer.current) clearInterval(timer.current);
            if (j.status === "done") setDone((d) => [...d.filter((x) => !(x.method === j.method && x.eps === j.eps)), j]);
            if (j.status === "error") setErr(j.error ?? "training failed");
          }
        } catch (e) { setErr(String((e as Error).message)); if (timer.current) clearInterval(timer.current); }
        finally { inFlight = false; }
      }, 700);
    } catch (e) { setErr(String((e as Error).message)); }
    setStarting(false);
  };

  const running = starting || job?.status === "running";
  const runs = [...done.filter((d) => d.id !== job?.id), ...(job ? [job] : [])];
  const maxEp = Math.max(N, ...runs.map((r) => r.epochs_total));
  const data = Array.from({ length: maxEp }, (_, i) => ({ epoch: i + 1, ...Object.fromEntries(runs.map((r) => [r.id, (r.epochs[i]?.[metric] as number | undefined) ?? null])) }));
  const last = job?.epochs[job.epochs.length - 1];
  const spent = last?.eps_spent ?? 0;

  return (
    <div className="grid">
      <Card title="Train a model now" hint={`Runs the real training code on the backend (${meta.info.train_size.toLocaleString()} people, PyTorch). Each epoch the model is evaluated on the test set.`}>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "end" }}>
          <div className="field" role="group" aria-label="Method"><span className="lbl">Method</span><Seg label="method" value={M} onChange={setMethod} options={order.map((m) => ({ v: m, l: short(m) }))} /></div>
          <div className="field" style={{ opacity: isPrivate(M) ? 1 : 0.4 }}><span className="lbl">Privacy budget</span><Seg label="epsilon" value={E} onChange={setEps} options={meta.train_epsilons.map((e) => ({ v: e, l: `ε = ${e}` }))} /></div>
          <div className="field" style={{ width: 180 }}><label htmlFor="ep">Epochs: <b>{N}</b></label><input id="ep" type="range" min={meta.train_epochs.min} max={meta.train_epochs.max} value={N} onChange={(e) => setEpochs(Number(e.target.value))} /></div>
          <button className="btn primary" type="button" onClick={start} disabled={running}>{running ? "Training…" : "Start training"}</button>
          {done.length > 0 && !running && <button className="btn" type="button" onClick={() => { setDone([]); setJob(null); }}>Clear runs</button>}
        </div>
        {err && <p className="note" style={{ color: "var(--bad)" }}>{err}</p>}
      </Card>

      {job && (
        <div className="grid g4">
          <Kpi k="Status" v={<span style={{ fontSize: 22 }}>{job.status === "running" ? `Epoch ${job.epochs.length} / ${job.epochs_total}` : job.status === "done" ? "Finished" : "Error"}</span>} d={<ModelName id={job.method} name={job.name} />} />
          <Kpi k="Privacy spent so far" v={job.private ? `ε = ${spent.toFixed(2)}` : "no privacy"} d={job.private && job.eps ? <div className="pbar" style={{ marginTop: 6, ["--t" as string]: "100%" }}><span style={{ width: `${Math.min(100, (spent / job.eps) * 100)}%`, background: "var(--brand)" }} /></div> : "ε = ∞"} />
          <Kpi k="Worst-group recall" v={pct(last?.intersection_worst_tpr)} d={`balanced accuracy ${pct(last?.balanced_accuracy)}`} />
          <Kpi k="Gradient noise σ" v={job.noise ? f3(job.noise.sigma_grad, 2) : "–"} d={job.noise ? `${job.noise.steps} steps · accounted ε ${f3(job.noise.epsilon)}` : "no noise"} />
        </div>
      )}

      <Card title="Learning curves" hint="Runs are kept so you can compare, e.g. the baseline and our method at the same ε."
        right={<Seg label="metric" value={metric} onChange={setMetric} options={[{ v: "intersection_worst_tpr", l: "Worst-group recall" }, { v: "balanced_accuracy", l: "Balanced acc." }, { v: "accuracy", l: "Accuracy" }, { v: "eps_spent", l: "ε spent" }]} />}>
        {runs.length === 0 ? <p className="muted" style={{ padding: 30, textAlign: "center" }}>Pick a method and press <b>Start training</b>.</p> : (
          <div style={{ height: 340 }}>
            <ResponsiveContainer>
              <LineChart data={data} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="epoch" label={{ value: "epoch", position: "insideBottomRight", offset: -2, fontSize: 11 }} />
                <YAxis domain={metric === "eps_spent" ? [0, "auto"] : [0, 1]} tickFormatter={(v) => (metric === "eps_spent" ? v.toFixed(1) : `${Math.round(v * 100)}%`)} />
                <Tooltip formatter={(v) => (typeof v === "number" ? (metric === "eps_spent" ? v.toFixed(3) : pct(v)) : String(v))} />
                <Legend itemSorter={() => 0} />
                {runs.map((r) => <Line key={r.id} dataKey={r.id} name={`${short(r.method)}${r.private ? ` ε=${r.eps}` : ""}`} stroke={color(r.method)} strokeWidth={2.5} dot={{ r: 3 }} strokeDasharray={r.id === job?.id && running ? "6 3" : undefined} connectNulls />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      {done.length > 0 && (
        <Card title="Finished runs" hint="Fairness checks on the test set for each run.">
          <div className="tbl">
            <table>
              <thead><tr><th>Model</th><th className="n">ε</th><th className="n">Epochs</th><th className="n">Accuracy</th><th className="n">Balanced acc.</th><th className="n">Worst-group recall</th><th>Four-fifths (sex)</th><th>Four-fifths (race)</th></tr></thead>
              <tbody>{done.map((r) => <tr key={r.id}><td><ModelName id={r.method} name={r.name} /></td><td className="n">{r.private ? r.eps : "∞"}</td><td className="n">{r.epochs_total}</td><td className="n">{pct(r.final?.accuracy)}</td><td className="n">{pct(r.final?.balanced_accuracy)}</td><td className="n"><b>{pct(r.final?.intersection_worst_tpr)}</b></td><td><Pass v={r.final_ethics?.sex?.four_fifths_pass} /> <span className="muted num">{f3(r.final_ethics?.sex?.disparate_impact_ratio, 2)}</span></td><td><Pass v={r.final_ethics?.race?.four_fifths_pass} /> <span className="muted num">{f3(r.final_ethics?.race?.disparate_impact_ratio, 2)}</span></td></tr>)}</tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
