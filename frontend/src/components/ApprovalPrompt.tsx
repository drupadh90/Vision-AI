import { useEffect, useState } from "react";
import type { ApprovalRequest } from "../lib/api";

/**
 * Blocking approval card for a `needs_approval` verdict.
 *
 * The countdown is not decoration: if it reaches zero the backend denies the
 * action (fail-closed), so the UI must make the deadline visible rather than
 * letting the user believe the agent will wait forever.
 */
export function ApprovalPrompt({
  request,
  onDecide,
}: {
  request: ApprovalRequest;
  onDecide: (id: string, approved: boolean, note: string) => void;
}) {
  const [remaining, setRemaining] = useState(request.seconds_remaining);
  const [note, setNote] = useState("");
  const [submitted, setSubmitted] = useState<"approved" | "denied" | null>(null);

  useEffect(() => {
    const timer = setInterval(() => setRemaining((r) => Math.max(0, r - 1)), 1000);
    return () => clearInterval(timer);
  }, []);

  const expired = remaining <= 0;
  const decide = (approved: boolean) => {
    if (submitted || expired) return;
    setSubmitted(approved ? "approved" : "denied");
    onDecide(request.id, approved, note);
  };

  return (
    <div className="rounded-xl border border-amber-500/40 bg-amber-500/[0.07] p-4 shadow-lg shadow-amber-500/5">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 animate-pulse-slow rounded-full bg-amber-400" />
          <h4 className="text-sm font-semibold text-amber-200">
            Approval required — the agent is paused
          </h4>
        </div>
        <span
          className={`rounded-full border px-2 py-0.5 font-mono text-[10px] ${
            expired
              ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
              : remaining < 30
                ? "border-rose-500/40 text-rose-300"
                : "border-amber-500/40 text-amber-300"
          }`}
        >
          {expired ? "expired — denied" : `${Math.floor(remaining)}s to decide`}
        </span>
      </div>

      <p className="mb-2 text-xs text-slate-300">{request.reason}</p>

      <div className="mb-2 rounded-lg border border-vision-border bg-black/50 p-2.5">
        <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wide text-slate-500">
          <span>tool: {request.tool}</span>
          {request.task_id && <span>· task {request.task_id}</span>}
        </div>
        <code className="block whitespace-pre-wrap break-all font-mono text-xs text-cyan-300">
          {request.payload}
        </code>
      </div>

      {request.safe_alternative && (
        <p className="mb-2 text-[11px] text-slate-400">
          <span className="text-slate-500">Suggested alternative: </span>
          {request.safe_alternative}
        </p>
      )}

      {submitted ? (
        <p
          className={`text-xs font-medium ${
            submitted === "approved" ? "text-emerald-300" : "text-rose-300"
          }`}
        >
          {submitted === "approved"
            ? "Approved — the agent is continuing."
            : "Denied — the agent will look for a legal alternative."}
        </p>
      ) : (
        <>
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            disabled={expired}
            placeholder="Optional note to the agent (e.g. use uv instead)"
            className="mb-2 w-full rounded-lg border border-vision-border bg-black/40 px-2.5 py-1.5
                       text-xs outline-none placeholder:text-slate-600 focus:border-vision-accent
                       disabled:opacity-50"
          />
          <div className="flex gap-2">
            <button
              onClick={() => decide(true)}
              disabled={expired}
              className="rounded-lg bg-emerald-500/90 px-3 py-1.5 text-xs font-medium text-slate-900
                         transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Approve
            </button>
            <button
              onClick={() => decide(false)}
              disabled={expired}
              className="rounded-lg border border-rose-500/50 px-3 py-1.5 text-xs font-medium text-rose-300
                         transition hover:bg-rose-500/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Deny
            </button>
          </div>
        </>
      )}
    </div>
  );
}
