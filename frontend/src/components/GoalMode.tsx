import { useCallback, useEffect, useRef, useState } from "react";
import { streamGoal, type AgentTask, type GoalEvent } from "../lib/api";
import { Markdown } from "../lib/markdown";

const AGENT_COLORS: Record<string, string> = {
  research: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  coder: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  writer: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  analyst: "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/30",
  critic: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  generalist: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

const STATUS_DOT: Record<string, string> = {
  pending: "bg-slate-600",
  running: "bg-cyan-400 animate-pulse-slow",
  done: "bg-emerald-400",
  failed: "bg-rose-500",
  blocked: "bg-amber-500",
};

const EXAMPLES = [
  "Research the top 3 vector databases and write a comparison table",
  "Build a Python script that analyses CSV sales data and charts revenue by region",
  "Draft a launch plan for an AI note-taking app aimed at students",
];

function eventLabel(e: GoalEvent): { text: string; tone: string } {
  switch (e.type) {
    case "goal_started":
      return { text: `Goal received: ${String(e.goal).slice(0, 80)}`, tone: "text-slate-300" };
    case "skills_activated":
      return { text: `Skills activated: ${(e.skills as string[]).join(", ")}`, tone: "text-cyan-300" };
    case "plan_ready":
      return { text: `Plan ready — ${(e.tasks as unknown[]).length} sub-tasks`, tone: "text-indigo-300" };
    case "plan_repaired":
      return { text: `Plan repaired: ${(e.problems as string[]).join("; ")}`, tone: "text-amber-300" };
    case "wave_started":
      return {
        text: `Wave ${e.wave}: spawning ${(e.tasks as Array<{ agent: string }>)
          .map((t) => t.agent)
          .join(", ")}`,
        tone: "text-indigo-300",
      };
    case "task_started":
      return { text: `▶ ${e.agent} started "${e.title}"`, tone: "text-slate-400" };
    case "task_completed":
      return { text: `✓ ${e.agent} finished "${e.title}"`, tone: "text-emerald-300" };
    case "task_failed":
      return { text: `✗ "${e.title}" failed: ${e.error}`, tone: "text-rose-300" };
    case "task_rejected":
      return { text: `↺ Manager rejected "${e.title}": ${e.feedback}`, tone: "text-amber-300" };
    case "task_reassigned":
      return { text: `⇄ Reassigned ${e.task_id}: ${e.from_agent} → ${e.to_agent}`, tone: "text-amber-300" };
    case "retry_budget_spent":
      return { text: `Retry budget spent on ${e.task_id}`, tone: "text-amber-300" };
    case "goal_completed":
      return { text: `Goal complete (${e.completed} done, ${e.failed} failed)`, tone: "text-emerald-300" };
    case "goal_failed":
      return { text: `Goal failed: ${e.reason}`, tone: "text-rose-300" };
    case "reflection_written":
      return { text: `Reflection stored: ${e.lesson ?? "—"}`, tone: "text-fuchsia-300" };
    default:
      return { text: e.type, tone: "text-slate-500" };
  }
}

export function GoalMode() {
  const [goal, setGoal] = useState("");
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<GoalEvent[]>([]);
  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [answer, setAnswer] = useState("");
  const [reflection, setReflection] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const closerRef = useRef<(() => void) | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [events]);

  useEffect(() => () => closerRef.current?.(), []);

  const start = useCallback(() => {
    const trimmed = goal.trim();
    if (!trimmed || running) return;
    setRunning(true);
    setEvents([]);
    setTasks([]);
    setAnswer("");
    setReflection(null);
    setError("");

    closerRef.current = streamGoal(trimmed, {
      onEvent: (e) => {
        setEvents((prev) => [...prev, e]);
        if (e.type === "plan_ready") setTasks(e.tasks as AgentTask[]);
        if (e.type === "task_started" || e.type === "task_completed" || e.type === "task_failed") {
          setTasks((prev) =>
            prev.map((t) =>
              t.id === e.task_id
                ? {
                    ...t,
                    status:
                      e.type === "task_started"
                        ? "running"
                        : e.type === "task_completed"
                          ? "done"
                          : "failed",
                  }
                : t,
            ),
          );
        }
      },
      onFinal: (payload) => {
        setAnswer(payload.final_answer);
        setTasks(payload.tasks);
        setReflection(payload.reflection);
        setRunning(false);
      },
      onError: (message) => {
        setError(message);
        setRunning(false);
      },
      onClose: () => setRunning(false),
    });
  }, [goal, running]);

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_380px]">
      <div className="space-y-4">
        <div className="rounded-xl border border-vision-border bg-vision-panel p-4">
          <label className="mb-2 block text-sm font-medium text-slate-300">
            Goal Mode — give Vision AI an objective, it plans a DAG and spawns sub-agents
          </label>
          <div className="flex gap-2">
            <input
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && start()}
              placeholder="e.g. Research vector databases and write a comparison"
              disabled={running}
              className="flex-1 rounded-lg border border-vision-border bg-black/40 px-3 py-2 text-sm
                         outline-none placeholder:text-slate-600 focus:border-vision-accent disabled:opacity-60"
            />
            <button
              onClick={start}
              disabled={running || !goal.trim()}
              className="rounded-lg bg-vision-accent px-4 py-2 text-sm font-medium text-white
                         transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {running ? "Running…" : "Execute"}
            </button>
          </div>
          {!running && !events.length && (
            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setGoal(ex)}
                  className="rounded-full border border-vision-border px-3 py-1 text-xs text-slate-400
                             transition hover:border-vision-accent hover:text-slate-200"
                >
                  {ex.slice(0, 46)}…
                </button>
              ))}
            </div>
          )}
          {error && (
            <p className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 p-2 text-sm text-rose-300">
              {error}
            </p>
          )}
        </div>

        {tasks.length > 0 && (
          <div className="rounded-xl border border-vision-border bg-vision-panel p-4">
            <h3 className="mb-3 text-sm font-semibold text-slate-300">Task DAG</h3>
            <div className="space-y-2">
              {tasks.map((t) => (
                <div
                  key={t.id}
                  className="flex items-start gap-3 rounded-lg border border-vision-border/60 bg-black/20 p-3"
                >
                  <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${STATUS_DOT[t.status]}`} />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm text-slate-200">{t.title}</span>
                      <span
                        className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
                          AGENT_COLORS[t.agent] ?? AGENT_COLORS.generalist
                        }`}
                      >
                        {t.agent}
                      </span>
                      {t.depends_on.length > 0 && (
                        <span className="text-[10px] text-slate-500">
                          after {t.depends_on.join(", ")}
                        </span>
                      )}
                      {t.attempts > 1 && (
                        <span className="text-[10px] text-amber-400">retry ×{t.attempts - 1}</span>
                      )}
                    </div>
                    {t.result && (
                      <p className="mt-1 line-clamp-2 text-xs text-slate-500">
                        {t.result.slice(0, 160)}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {answer && (
          <div className="rounded-xl border border-vision-border bg-vision-panel p-5">
            <h3 className="mb-3 text-sm font-semibold text-emerald-300">Final deliverable</h3>
            <Markdown>{answer}</Markdown>
          </div>
        )}

        {reflection && Object.keys(reflection).length > 0 && (
          <div className="rounded-xl border border-fuchsia-500/25 bg-fuchsia-500/5 p-4">
            <h3 className="mb-2 text-sm font-semibold text-fuchsia-300">
              Digital Twin reflection
            </h3>
            {typeof reflection.lesson === "string" && (
              <p className="text-sm text-slate-300">
                <span className="text-slate-500">Lesson learned: </span>
                {reflection.lesson}
              </p>
            )}
            <p className="mt-1 text-xs text-slate-500">
              {String(reflection.stored ?? 0)} memories written to long-term storage
            </p>
          </div>
        )}
      </div>

      <div className="rounded-xl border border-vision-border bg-vision-panel p-4">
        <h3 className="mb-3 text-sm font-semibold text-slate-300">Live agent trace</h3>
        <div ref={logRef} className="max-h-[70vh] space-y-1.5 overflow-y-auto font-mono text-xs">
          {events.length === 0 && (
            <p className="text-slate-600">Awaiting a goal…</p>
          )}
          {events.map((e, i) => {
            const { text, tone } = eventLabel(e);
            return (
              <div key={i} className={`leading-relaxed ${tone}`}>
                {text}
              </div>
            );
          })}
          {running && <div className="animate-pulse text-cyan-400">▌</div>}
        </div>
      </div>
    </div>
  );
}
