import { useEffect, useState } from "react";
import { GoalMode } from "./components/GoalMode";
import { GuardianPanel } from "./components/GuardianPanel";
import { SkillLab } from "./components/SkillLab";
import { TwinPanel } from "./components/TwinPanel";
import { api, type HealthInfo } from "./lib/api";

type Tab = "goal" | "skills" | "twin" | "guardian";

const TABS: Array<{ id: Tab; label: string; hint: string }> = [
  { id: "goal", label: "Goal Mode", hint: "Autonomous DAG execution" },
  { id: "skills", label: "Skill Lab", hint: "Learn from YouTube" },
  { id: "twin", label: "Digital Twin", hint: "Long-term memory" },
  { id: "guardian", label: "Guardian", hint: "Constitutional filter" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("goal");
  const [health, setHealth] = useState<HealthInfo | null>(null);

  useEffect(() => {
    const load = () => api.health().then(setHealth).catch(() => setHealth(null));
    load();
    const timer = setInterval(load, 15000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-full">
      <header className="border-b border-vision-border bg-vision-panel/60 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-5 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-vision-accent to-vision-accent2 font-bold text-white">
              V
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight text-white">Vision AI</h1>
              <p className="text-[11px] text-slate-500">
                Your second digital self — it learns, plans, and executes like you
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-[11px]">
            {health ? (
              <>
                <Badge
                  tone={health.mock_mode ? "warn" : "ok"}
                  label={health.mock_mode ? "mock LLM" : health.llm_provider}
                />
                <Badge tone="neutral" label={`exec: ${health.executor}`} />
                <Badge
                  tone={health.guardian.enabled ? "ok" : "danger"}
                  label={health.guardian.enabled ? `guardian ${health.guardian.mode}` : "guardian off"}
                />
                <Badge tone="neutral" label={`${health.skills.total_skills} skills`} />
                <Badge tone="neutral" label={`${health.memory.total_memories} memories`} />
              </>
            ) : (
              <Badge tone="danger" label="backend offline" />
            )}
          </div>
        </div>

        <nav className="mx-auto flex max-w-7xl gap-1 px-5">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              title={t.hint}
              className={`relative px-4 py-2.5 text-sm transition ${
                tab === t.id
                  ? "text-white"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {t.label}
              {tab === t.id && (
                <span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-gradient-to-r from-vision-accent to-vision-accent2" />
              )}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-5">
        {health?.mock_mode && (
          <div className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-2 text-xs text-amber-300">
            Running on the offline mock LLM — every module works, but reasoning is
            deterministic stub output. Set <code className="font-mono">OPENAI_API_KEY</code> and{" "}
            <code className="font-mono">VISION_LLM_PROVIDER=openai</code> in{" "}
            <code className="font-mono">.env</code> for real intelligence.
          </div>
        )}
        {tab === "goal" && <GoalMode />}
        {tab === "skills" && <SkillLab />}
        {tab === "twin" && <TwinPanel />}
        {tab === "guardian" && <GuardianPanel />}
      </main>
    </div>
  );
}

function Badge({ label, tone }: { label: string; tone: "ok" | "warn" | "danger" | "neutral" }) {
  const styles = {
    ok: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    warn: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    danger: "border-rose-500/30 bg-rose-500/10 text-rose-300",
    neutral: "border-vision-border bg-black/30 text-slate-400",
  }[tone];
  return <span className={`rounded-full border px-2 py-0.5 ${styles}`}>{label}</span>;
}
