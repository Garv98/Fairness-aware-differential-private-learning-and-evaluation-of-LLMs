"use client";
import { Fragment, ReactNode, useState } from "react";
import { useApi, useMeta } from "@/lib/api";
import { ApiError, Card, Loading, Seg } from "./ui";

interface Prov { dataset: string; dataset_rows: number; dataset_fingerprint: string; git_commit: string | null; git_dirty: boolean | null; python: string; packages: Record<string, string>; seed: number; train_size: number; val_size: number; test_size: number; models: Record<string, { trained_at?: string; seconds?: number }> }

// Minimal Markdown renderer for the generated model cards (headings, lists, tables, emphasis, code).
function inline(s: string): ReactNode[] {
  return s.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g).filter(Boolean).map((p, i) =>
    p.startsWith("`") ? <code key={i}>{p.slice(1, -1)}</code> : p.startsWith("**") ? <b key={i}>{p.slice(2, -2)}</b> : p.startsWith("*") ? <i key={i}>{p.slice(1, -1)}</i> : <Fragment key={i}>{p}</Fragment>);
}
function Markdown({ text }: { text: string }) {
  const lines = text.split("\n"), out: ReactNode[] = [];
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.startsWith("# ")) out.push(<h1 key={i} className="md-h1">{inline(l.slice(2))}</h1>);
    else if (l.startsWith("## ")) out.push(<h2 key={i} className="md-h2">{inline(l.slice(3))}</h2>);
    else if (l.startsWith("- ")) { const items = []; while (i < lines.length && lines[i].startsWith("- ")) items.push(<li key={i}>{inline(lines[i++].slice(2))}</li>); i--; out.push(<ul key={`u${i}`}>{items}</ul>); }
    else if (l.startsWith("|")) {
      const rows = []; while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]); i--;
      const cells = (r: string) => r.split("|").slice(1, -1).map((c) => c.trim());
      out.push(<div className="tbl" key={`t${i}`}><table><thead><tr>{cells(rows[0]).map((c, k) => <th key={k}>{c}</th>)}</tr></thead><tbody>{rows.slice(2).map((r, k) => <tr key={k}>{cells(r).map((c, j) => <td key={j}>{inline(c)}</td>)}</tr>)}</tbody></table></div>);
    } else if (l.trim()) out.push(<p key={i}>{inline(l)}</p>);
  }
  return <div className="md">{out}</div>;
}

export default function ModelCards({ eps }: { eps: number }) {
  const { meta, order, short, isPrivate, oursId } = useMeta();
  const [m, setM] = useState<string | null>(null);
  const M = m ?? oursId ?? order[0];
  const { data: card, error } = useApi<string>(M ? `/api/model_card?method=${M}${isPrivate(M) ? `&eps=${eps}` : ""}` : null);
  const { data: pv } = useApi<Prov>("/api/provenance");
  if (error) return <ApiError error={error} />;
  if (!meta) return <Loading />;
  const download = () => {
    if (!card) return;
    const url = URL.createObjectURL(new Blob([card], { type: "text/markdown" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `model-card_${M}${isPrivate(M) ? `_eps${eps}` : ""}.md` });
    a.click(); URL.revokeObjectURL(url);
  };

  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1fr)", alignItems: "start" }}>
      <Card title="Model card" hint="Generated from the evaluation artifacts (Mitchell et al., 2019 format): intended use, data, metrics by group, privacy, limitations."
        right={<div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}><Seg label="model" value={M} onChange={setM} options={order.map((x) => ({ v: x, l: short(x) }))} /><button className="btn" type="button" onClick={download} disabled={!card}>Download .md</button></div>}>
        {card ? <Markdown text={card} /> : <Loading what="model card" />}
      </Card>
      <Card title="Provenance & audit trail" hint="What exactly was trained, on what, with which code.">
        {!pv ? <Loading /> : (
          <div className="grid" style={{ gap: 10 }}>
            <dl className="kv">
              <dt>Dataset</dt><dd>{pv.dataset} · {pv.dataset_rows.toLocaleString()} rows</dd>
              <dt>Data fingerprint</dt><dd className="mono">{pv.dataset_fingerprint}</dd>
              <dt>Split (seed {pv.seed})</dt><dd>{pv.train_size.toLocaleString()} / {pv.val_size.toLocaleString()} / {pv.test_size.toLocaleString()}</dd>
              <dt>Code version</dt><dd className="mono">{pv.git_commit ?? "not a git checkout"}{pv.git_dirty ? " + local changes" : ""}</dd>
              <dt>Python</dt><dd className="mono">{pv.python}</dd>
              {Object.entries(pv.packages).map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd className="mono">{v}</dd></Fragment>)}
            </dl>
            <h3 className="sub-h">Training log</h3>
            <div className="tbl" style={{ maxHeight: 280, overflowY: "auto" }}>
              <table><thead><tr><th>Model</th><th>Trained (UTC)</th><th className="n">Time</th></tr></thead>
                <tbody>{Object.entries(pv.models).map(([k, v]) => <tr key={k}><td className="mono" style={{ fontSize: 12 }}>{k.replace("__", " ε=").replace("=inf", "=∞")}</td><td className="mono" style={{ fontSize: 12 }}>{v.trained_at?.replace("T", " ").replace("+00:00", "") ?? "–"}</td><td className="n">{v.seconds ?? "–"}s</td></tr>)}</tbody></table>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
