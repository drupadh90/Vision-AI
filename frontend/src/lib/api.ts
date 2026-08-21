/**
 * API client. Every URL is relative so the Vite dev server (or any reverse
 * proxy in production) forwards it to FastAPI — the browser never needs to
 * know where the backend lives.
 */

export type TaskStatus = "pending" | "running" | "done" | "failed" | "blocked";

export interface AgentTask {
  id: string;
  title: string;
  description: string;
  agent: string;
  depends_on: string[];
  status: TaskStatus;
  attempts: number;
  result: string;
  error: string;
  feedback: string;
  tool_results: ToolResult[];
}

export interface ToolResult {
  tool: string;
  ok: boolean;
  output: string;
  blocked: boolean;
  artifacts: string[];
  verdict?: Verdict | null;
}

export interface Verdict {
  decision: "allow" | "deny" | "needs_approval";
  reason: string;
  layer: string;
  violated_principles: string[];
  safe_alternative: string | null;
  confidence: number;
}

export interface GoalEvent {
  type: string;
  ts: number;
  [key: string]: unknown;
}

export interface Skill {
  skill_id: string;
  title: string;
  video_id: string;
  channel: string;
  duration: number;
  method: string;
  url: string;
  preview: string;
}

export interface HealthInfo {
  status: string;
  llm_provider: string;
  mock_mode: boolean;
  embedding_provider: string;
  executor: string;
  guardian: { enabled: boolean; mode: string; llm_review: boolean };
  memory: { total_memories: number; by_kind: Record<string, number> };
  skills: { total_chunks: number; total_skills: number };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthInfo>("/api/health"),

  listSkills: () =>
    request<{ skills: Skill[]; stats: { total_skills: number; total_chunks: number } }>(
      "/api/skills",
    ),

  learnYouTube: (url: string, skillName?: string) =>
    request<Record<string, unknown>>("/api/skills/learn", {
      method: "POST",
      body: JSON.stringify({ url, skill_name: skillName || null }),
    }),

  searchSkills: (query: string, k = 5) =>
    request<{ matches: Array<{ title: string; score: number; text: string; citation: string }> }>(
      "/api/skills/search",
      { method: "POST", body: JSON.stringify({ query, k }) },
    ),

  memoryStats: () => request<Record<string, unknown>>("/api/memory/stats"),

  recallMemory: (query: string, k = 8) =>
    request<{ memories: Array<{ content: string; kind: string; confidence: number; score: number }> }>(
      "/api/memory/recall",
      { method: "POST", body: JSON.stringify({ query, k }) },
    ),

  constitution: () =>
    request<{ principles: Array<{ id: string; title: string; rule: string; examples: string[] }> }>(
      "/api/guardian/constitution",
    ),

  checkAction: (payload: string, actionType = "shell") =>
    request<Verdict>("/api/guardian/check", {
      method: "POST",
      body: JSON.stringify({ payload, action_type: actionType }),
    }),

  audit: () => request<{ entries: Array<Record<string, unknown>> }>("/api/guardian/audit"),

  intuition: (context: string) =>
    request<{ suggestion: { suggestion: string; confidence: number; rationale: string } | null }>(
      "/api/intuition",
      { method: "POST", body: JSON.stringify({ context }) },
    ),

  artifacts: () =>
    request<{ artifacts: Array<{ path: string; size: number; modified: number }> }>("/api/artifacts"),
};

/** Open a Goal Mode websocket. Returns a closer. */
export function streamGoal(
  goal: string,
  handlers: {
    onEvent: (e: GoalEvent) => void;
    onFinal: (payload: {
      final_answer: string;
      tasks: AgentTask[];
      reflection: Record<string, unknown>;
      activated_skills: Array<{ title: string; score: number }>;
    }) => void;
    onError: (message: string) => void;
    onClose?: () => void;
  },
): () => void {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/ws/goal`);

  ws.onopen = () => ws.send(JSON.stringify({ goal }));
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === "final") handlers.onFinal(data);
    else if (data.type === "error") handlers.onError(String(data.message));
    else handlers.onEvent(data as GoalEvent);
  };
  ws.onerror = () => handlers.onError("WebSocket connection failed.");
  ws.onclose = () => handlers.onClose?.();

  return () => ws.close();
}
