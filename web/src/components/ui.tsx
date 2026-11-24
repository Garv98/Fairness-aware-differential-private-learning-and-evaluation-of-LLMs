"use client";
import type { ReactNode } from "react";
import { API } from "@/lib/api";
import { color } from "@/lib/models";

export function Seg<T extends string | number>({ value, options, onChange, label }: { value: T; options: { v: T; l: ReactNode }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => <button key={String(o.v)} type="button" aria-pressed={o.v === value} onClick={() => onChange(o.v)}>{o.l}</button>)}
    </div>
  );
}

export function Card({ title, hint, right, children, style }: { title?: ReactNode; hint?: ReactNode; right?: ReactNode; children: ReactNode; style?: React.CSSProperties }) {
  return (
    <section className="card" style={style}>
      {(title || right) && <div className="card-head"><div>{title && <h2>{title}</h2>}{hint && <p className="hint">{hint}</p>}</div>{right}</div>}
      {children}
    </section>
  );
}

export function Kpi({ k, v, d, color: c }: { k: ReactNode; v: ReactNode; d?: ReactNode; color?: string }) {
  return <div className="card kpi"><span className="k">{k}</span><span className="v" style={{ color: c }}>{v}</span>{d && <span className="d">{d}</span>}</div>;
}

export function ModelName({ id, name }: { id: string; name: string }) {
  return <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}><span className="sw" style={{ background: color(id), margin: 0 }} />{name}</span>;
}

export function Pass({ v }: { v: boolean | null | undefined }) {
  if (v == null) return <span className="badge mute">n/a</span>;
  return v ? <span className="badge ok">✓ pass</span> : <span className="badge bad">✗ fail</span>;
}

export function ApiError({ error }: { error: string }) {
  return (
    <div className="err">
      <b>Can&apos;t reach the FairDP backend</b> ({error}). Start it with:
      <pre style={{ margin: "8px 0 0", fontSize: 12.5 }}>{`cd FairDP-Research/server\n../.venv/Scripts/python -m uvicorn app:app --port 8000`}</pre>
      <span className="muted">Expected at {API}</span>
    </div>
  );
}

export function Loading({ what = "data" }: { what?: string }) {
  return <p className="muted" style={{ padding: 12 }}>Loading {what}…</p>;
}

export function EpsPicker({ eps, setEps, options }: { eps: number; setEps: (e: number) => void; options: number[] }) {
  const lo = Math.min(...options), hi = Math.max(...options);
  return <Seg label="privacy budget" value={eps} onChange={setEps} options={options.map((e) => ({ v: e, l: <>ε = {e}{e === lo ? " · strongest" : e === hi ? " · weakest" : ""}</> }))} />;
}

export const fmtPct = (v: unknown) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v));
export const fmt3 = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : String(v));
