"use client";
import { useEffect, useRef, useState } from "react";
import { Activity, BadgeCheck, Bot, FileText, Gauge, Landmark, Scale, ScanSearch, ShieldCheck, SlidersHorizontal, UserCheck, Users } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import Predict from "@/components/Predict";
import Overview from "@/components/Overview";
import Fairness from "@/components/Fairness";
import Privacy from "@/components/Privacy";
import Mitigation from "@/components/Mitigation";
import ProxyAudit from "@/components/ProxyAudit";
import Scorecard from "@/components/Scorecard";
import ModelCards from "@/components/ModelCards";
import TrainLive from "@/components/TrainLive";
import LLMAudit from "@/components/LLMAudit";
import CreditStudy from "@/components/CreditStudy";
import { ApiError, EpsPicker } from "@/components/ui";
import { API, get, invalidate, useMeta } from "@/lib/api";

interface Tab { id: string; label: string; icon: LucideIcon; title: string; sub: string; eps: boolean; section: string }
const TABS: Tab[] = [
  { id: "predict", section: "Demo", label: "Decision demo", icon: UserCheck, title: "Decision demo", sub: "Score one applicant with every model, change only their identity, and find what would change the decision", eps: true },
  { id: "overview", section: "Analysis", label: "Overview", icon: Gauge, title: "Overview", sub: "Accuracy, fairness and privacy of every served model", eps: true },
  { id: "fairness", section: "Analysis", label: "Fairness", icon: Users, title: "Fairness analysis", sub: "Group, intersectional and individual fairness criteria", eps: true },
  { id: "privacy", section: "Analysis", label: "Privacy", icon: ShieldCheck, title: "Privacy analysis", sub: "Budget accounting, privacy–utility trade-off, membership-inference attack", eps: true },
  { id: "mitigation", section: "Responsible AI", label: "Mitigation", icon: SlidersHorizontal, title: "Fairness mitigation", sub: "Post-processing thresholds and the trade-off between fairness criteria", eps: true },
  { id: "proxy", section: "Responsible AI", label: "Proxy audit", icon: ScanSearch, title: "Proxy audit", sub: "Does hiding sex and race make a model fair?", eps: false },
  { id: "scorecard", section: "Responsible AI", label: "Scorecard", icon: BadgeCheck, title: "Responsible-AI scorecard", sub: "Privacy, fairness, transparency, accountability and robustness checks per model", eps: true },
  { id: "cards", section: "Responsible AI", label: "Model cards", icon: FileText, title: "Model cards & provenance", sub: "Documentation and audit trail generated from the artifacts", eps: true },
  { id: "train", section: "Experiments", label: "Train live", icon: Activity, title: "Train live", sub: "Train a model on the backend and watch it learn", eps: false },
  { id: "llm", section: "Experiments", label: "LLM audit", icon: Bot, title: "LLM audit", sub: "Do modern LLMs change their answer when only race or sex changes?", eps: false },
  { id: "credit", section: "Experiments", label: "Credit study", icon: Landmark, title: "Credit-risk study", sub: "First workstream: DP-SGD on loan-default data", eps: false },
];

function Ledger({ eps, delta, n }: { eps: number; delta: number; n: number }) {
  return (
    <div className="ledger" aria-label="privacy ledger">
      <span className="ledger-tag"><Scale size={14} strokeWidth={2} /> privacy ledger</span>
      <span><i>ε</i><b>{eps}</b></span>
      <span><i>δ</i><b>{delta.toExponential(1)}</b></span>
      <span title="maximum factor by which seeing the model can change an attacker's odds about any one person"><i>odds ≤</i><b>×{Math.exp(eps) < 100 ? Math.exp(eps).toFixed(2) : Math.exp(eps).toFixed(0)}</b></span>
      <span><i>people</i><b>{n.toLocaleString()}</b></span>
    </div>
  );
}

export default function Page() {
  const [tab, setTab] = useState("predict");
  const [visited, setVisited] = useState<string[]>(["predict"]);
  const { meta, error: metaError, reload } = useMeta();
  const reloadRef = useRef(reload);
  reloadRef.current = reload;
  const [eps, setEps] = useState<number | null>(null);
  const [health, setHealth] = useState<"ok" | "down" | "checking">("checking");
  const E = eps ?? meta?.epsilons[0] ?? null;

  useEffect(() => {
    const h = window.location.hash.slice(1);
    if (TABS.some((t) => t.id === h)) setTab(h);
    let was: string | null = null;
    const check = () => get("/api/health").then(() => {
      if (was === "down") { invalidate(); reloadRef.current(); }   // backend came back: refetch everything
      was = "ok"; setHealth("ok");
    }).catch(() => { was = "down"; setHealth("down"); });
    check(); const t = setInterval(check, 10000); return () => clearInterval(t);
  }, []);
  useEffect(() => {
    window.history.replaceState(null, "", `#${tab}`);
    setVisited((v) => (v.includes(tab) ? v : [...v, tab]));
  }, [tab]);

  const T = TABS.find((t) => t.id === tab)!;
  const sections = [...new Set(TABS.map((t) => t.section))];
  const render = (id: string) => {
    if (E == null) return null;
    switch (id) {
      case "predict": return <Predict eps={E} />;
      case "overview": return <Overview eps={E} />;
      case "fairness": return <Fairness eps={E} />;
      case "privacy": return <Privacy eps={E} />;
      case "mitigation": return <Mitigation eps={E} />;
      case "proxy": return <ProxyAudit />;
      case "scorecard": return <Scorecard eps={E} />;
      case "cards": return <ModelCards eps={E} />;
      case "train": return <TrainLive />;
      case "llm": return <LLMAudit />;
      default: return <CreditStudy />;
    }
  };

  return (
    <div className="app">
      <aside className="side">
        <div className="logo"><div className="mark">ε</div><div><b>FairDP</b><small>audit console</small></div></div>
        <nav className="nav" aria-label="Sections">
          {sections.map((s) => (
            <div key={s} style={{ display: "contents" }}>
              <div className="sec">{s}</div>
              {TABS.filter((t) => t.section === s).map((t) => {
                const Icon = t.icon;
                return <button key={t.id} type="button" aria-current={tab === t.id ? "page" : undefined} onClick={() => setTab(t.id)}><Icon size={17} strokeWidth={1.8} />{t.label}</button>;
              })}
            </div>
          ))}
        </nav>
        <div className="foot">
          <span><span className="dot" style={{ background: health === "ok" ? "#34d399" : health === "down" ? "#f87171" : "#a3a3a3" }} />Backend {health === "ok" ? "online" : health === "down" ? "offline" : "…"}</span>
          <span className="mono" style={{ fontSize: 10.5 }}>{API}</span>
          {meta && <span>{meta.info.dataset} · {meta.models.length} models · seed {meta.info.seed}</span>}
        </div>
      </aside>
      <div style={{ minWidth: 0 }}>
        <header className="top">
          <div style={{ minWidth: 0 }}><h1>{T.title}</h1><div className="sub">{T.sub}</div></div>
          <div className="spacer" />
          {T.eps && meta && E != null && (
            <div className="top-controls">
              <EpsPicker eps={E} setEps={setEps} options={meta.epsilons} />
              <Ledger eps={E} delta={meta.info.delta} n={meta.info.train_size} />
            </div>
          )}
        </header>
        <main className="content">
          {metaError ? (
            <div className="grid"><ApiError error={metaError} /><div><button className="btn primary" type="button" onClick={reload}>Retry</button></div></div>
          ) : E == null ? <p className="muted">Connecting to the backend…</p> : (
            // visited tabs stay mounted (hidden) so their state survives tab switches
            visited.map((id) => <div key={id} className={`grid tab-panel ${id === tab ? "" : "offstage"}`} aria-hidden={id !== tab} inert={id !== tab}>{render(id)}</div>)
          )}
        </main>
      </div>
    </div>
  );
}
