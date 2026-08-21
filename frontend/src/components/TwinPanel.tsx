import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";

const KIND_STYLES: Record<string, string> = {
  preference: "bg-indigo-500/15 text-indigo-300",
  lesson: "bg-fuchsia-500/15 text-fuchsia-300",
  episode: "bg-slate-500/15 text-slate-300",
  pattern: "bg-cyan-500/15 text-cyan-300",
  feedback: "bg-emerald-500/15 text-emerald-300",
};

export function TwinPanel() {
  const [stats, setStats] = useState<{ total_memories: number; by_kind: Record<string, number> } | null>(
    null,
  );
  const [query, setQuery] = useState("");
  const [memories, setMemories] = useState<
    Array<{ content: string; kind: string; confidence: number; score: number }>
  >([]);

  const refresh = useCallback(async () => {
    try {
      const s = (await api.memoryStats()) as unknown as {
        total_memories: number;
        by_kind: Record<string, number>;
      };
      setStats(s);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const recall = async () => {
    if (!query.trim()) return;
    try {
      const data = await api.recallMemory(query.trim());
      setMemories(data.memories);
    } catch {
      setMemories([]);
    }
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
      <div className="space-y-4">
        <div className="rounded-xl border border-vision-border bg-vision-panel p-4">
          <h3 className="mb-1 text-sm font-semibold text-slate-200">Digital Twin</h3>
          <p className="mb-3 text-xs text-slate-500">
            Vision AI writes a reflection after every task, reinforcing what it believes about you.
          </p>
          <div className="text-3xl font-semibold text-white">
            {stats?.total_memories ?? 0}
          </div>
          <div className="text-xs text-slate-500">total memories</div>
          <div className="mt-3 space-y-1.5">
            {Object.entries(stats?.by_kind ?? {}).map(([kind, count]) => (
              <div key={kind} className="flex items-center justify-between text-xs">
                <span className={`rounded px-1.5 py-0.5 ${KIND_STYLES[kind] ?? ""}`}>{kind}</span>
                <span className="text-slate-400">{count}</span>
              </div>
            ))}
          </div>
          <button
            onClick={refresh}
            className="mt-4 w-full rounded-lg border border-vision-border py-1.5 text-xs text-slate-400
                       hover:border-vision-accent hover:text-slate-200"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-vision-border bg-vision-panel p-4">
        <h3 className="mb-3 text-sm font-semibold text-slate-200">Recall memory</h3>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && recall()}
            placeholder="What does Vision AI know about how I work?"
            className="flex-1 rounded-lg border border-vision-border bg-black/40 px-3 py-2 text-sm
                       outline-none placeholder:text-slate-600 focus:border-vision-accent"
          />
          <button
            onClick={recall}
            className="rounded-lg border border-vision-border px-4 py-2 text-sm text-slate-300 hover:border-vision-accent"
          >
            Recall
          </button>
        </div>
        <div className="mt-4 space-y-2">
          {memories.map((m, i) => (
            <div key={i} className="rounded-lg border border-vision-border/60 bg-black/20 p-3">
              <div className="mb-1 flex items-center gap-2">
                <span className={`rounded px-1.5 py-0.5 text-[10px] ${KIND_STYLES[m.kind] ?? ""}`}>
                  {m.kind}
                </span>
                <span className="text-[10px] text-slate-500">
                  confidence {m.confidence.toFixed(2)}
                </span>
              </div>
              <p className="text-xs leading-relaxed text-slate-300">{m.content}</p>
            </div>
          ))}
          {memories.length === 0 && (
            <p className="text-xs text-slate-600">
              No memories recalled yet. Run a goal first — the reflection loop populates this.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
