import { useCallback, useEffect, useState } from "react";
import { api, type Skill } from "../lib/api";

export function SkillLab() {
  const [url, setUrl] = useState("");
  const [skills, setSkills] = useState<Skill[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null);
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<
    Array<{ title: string; score: number; text: string; citation: string }>
  >([]);

  const refresh = useCallback(async () => {
    try {
      const data = await api.listSkills();
      setSkills(data.skills);
    } catch {
      /* backend not ready yet */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const learn = async () => {
    if (!url.trim() || busy) return;
    setBusy(true);
    setMessage(null);
    try {
      const report = await api.learnYouTube(url.trim());
      setMessage({
        text: `Learned "${report.title}" — ${report.chunks} chunks via ${report.method}.`,
        ok: true,
      });
      setUrl("");
      await refresh();
    } catch (err) {
      setMessage({ text: (err as Error).message, ok: false });
    } finally {
      setBusy(false);
    }
  };

  const search = async () => {
    if (!query.trim()) return;
    try {
      const data = await api.searchSkills(query.trim());
      setMatches(data.matches);
    } catch (err) {
      setMessage({ text: (err as Error).message, ok: false });
    }
  };

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-4">
        <div className="rounded-xl border border-vision-border bg-vision-panel p-4">
          <h3 className="mb-1 text-sm font-semibold text-slate-200">
            YouTube Skill Acquisition
          </h3>
          <p className="mb-3 text-xs text-slate-500">
            Paste a tutorial URL. Vision AI pulls the transcript (captions, or Whisper when
            captions are missing), chunks it with timestamps, and files it in the Skill DB.
          </p>
          <div className="flex gap-2">
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && learn()}
              placeholder="https://www.youtube.com/watch?v=…"
              disabled={busy}
              className="flex-1 rounded-lg border border-vision-border bg-black/40 px-3 py-2 text-sm
                         outline-none placeholder:text-slate-600 focus:border-vision-accent"
            />
            <button
              onClick={learn}
              disabled={busy || !url.trim()}
              className="rounded-lg bg-vision-accent2 px-4 py-2 text-sm font-medium text-slate-900
                         transition hover:bg-cyan-300 disabled:opacity-40"
            >
              {busy ? "Learning…" : "Learn"}
            </button>
          </div>
          {message && (
            <p
              className={`mt-3 rounded-lg border p-2 text-xs ${
                message.ok
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                  : "border-rose-500/30 bg-rose-500/10 text-rose-300"
              }`}
            >
              {message.text}
            </p>
          )}
        </div>

        <div className="rounded-xl border border-vision-border bg-vision-panel p-4">
          <h3 className="mb-3 text-sm font-semibold text-slate-200">
            Acquired skills ({skills.length})
          </h3>
          {skills.length === 0 ? (
            <p className="text-xs text-slate-600">
              No skills yet. Ingest a video to teach Vision AI something new.
            </p>
          ) : (
            <ul className="space-y-2">
              {skills.map((s) => (
                <li
                  key={s.skill_id}
                  className="rounded-lg border border-vision-border/60 bg-black/20 p-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm text-slate-200">{s.title}</span>
                    <span className="shrink-0 rounded bg-cyan-500/10 px-1.5 py-0.5 text-[10px] text-cyan-300">
                      {s.method}
                    </span>
                  </div>
                  <p className="mt-1 line-clamp-2 text-xs text-slate-500">{s.preview}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-vision-border bg-vision-panel p-4">
        <h3 className="mb-1 text-sm font-semibold text-slate-200">Skill Activation preview</h3>
        <p className="mb-3 text-xs text-slate-500">
          Query the Skill DB exactly as the agent does before acting.
        </p>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()}
            placeholder="e.g. how do I colour grade footage?"
            className="flex-1 rounded-lg border border-vision-border bg-black/40 px-3 py-2 text-sm
                       outline-none placeholder:text-slate-600 focus:border-vision-accent"
          />
          <button
            onClick={search}
            className="rounded-lg border border-vision-border px-4 py-2 text-sm text-slate-300 hover:border-vision-accent"
          >
            Search
          </button>
        </div>
        <div className="mt-4 space-y-3">
          {matches.map((m, i) => (
            <div key={i} className="rounded-lg border border-vision-border/60 bg-black/20 p-3">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-slate-200">{m.title}</span>
                <span className="rounded bg-vision-accent/15 px-1.5 py-0.5 text-[10px] text-indigo-300">
                  {m.score.toFixed(3)}
                </span>
              </div>
              <p className="text-xs leading-relaxed text-slate-400">{m.text.slice(0, 260)}…</p>
            </div>
          ))}
          {matches.length === 0 && (
            <p className="text-xs text-slate-600">No matches yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}
