# Vision AI

> An autonomous agent that becomes a second version of you — it learns your
> preferences, acquires new skills from YouTube, and executes high-level goals
> through a graph of specialised sub-agents, under a hard security constitution.

Vision AI is positioned as an alternative to Hermes-style agents. The
differences that matter are architectural, not cosmetic:

| | Typical agent | **Vision AI** |
|---|---|---|
| Memory | Chat history in a context window | **Digital Twin** — persistent, confidence-weighted beliefs that strengthen with evidence |
| Learning | Fixed capabilities | **YouTube Skill Acquisition** — watch a tutorial, gain a timestamped, retrievable skill |
| Execution | Sequential tool loop | **LangGraph DAG** — independent sub-tasks run concurrently; a Manager reviews and re-assigns |
| Safety | Prompt-level "please be careful" | **Constitutional pre-execution filter** — path jail + deterministic pattern blocks + Guardian LLM, failing closed |
| Initiative | Waits for commands | **Proactive Intuition** — offers a relevant learned skill, on a strict interruption budget |

---

## Quick start

Vision AI boots with **zero API keys**. A deterministic mock LLM and local hash
embeddings let you exercise every module offline, then you flip one env var for
real intelligence.

```bash
# 1. backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 2. frontend (second terminal)
cd frontend
npm install
npm run dev            # http://localhost:5173
```

Then open the UI, or drive it headlessly:

```bash
python scripts/demo.py        # end-to-end tour of all five modules
pytest                        # 76 tests, no network required
```

### Going live

```dotenv
VISION_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
VISION_EMBEDDING_PROVIDER=openai
VISION_TRANSCRIBER=auto        # captions first, Whisper fallback
```

---

## Architecture

```
                    ┌──────────────── React + Tailwind UI ────────────────┐
                    │  Goal Mode · Skill Lab · Digital Twin · Guardian    │
                    └───────────────┬─────────────────┬───────────────────┘
                            REST /api          WebSocket /ws/goal
                    ┌───────────────┴─────────────────┴───────────────────┐
                    │                    FastAPI                          │
                    └───────────────────────┬─────────────────────────────┘
                                            │
        ┌───────────────┬───────────────────┼───────────────┬──────────────┐
        │               │                   │               │              │
   Digital Twin    Skill Library      Goal Mode graph    Guardian     Intuition
   (ChromaDB)      (ChromaDB)          (LangGraph)     (constitution)  (engine)
        │               │                   │               │
        │               │            spawns sub-agents      │
        │               │        research · coder · writer  │
        │               │        analyst · critic           │
        │               │                   │               │
        └───────reflection loop─────────────┘          ToolBelt ──► Docker sandbox
```

### Goal Mode state machine

```
prepare ─► plan ─► execute_wave ─► review ─┬─ more ready work ─► execute_wave
                                           └─ done / budget ───► synthesize ─► reflect ─► END
```

* **prepare** — loads Digital Twin context and activates relevant skills.
* **plan** — emits a validated DAG. Cycles, dangling deps and oversized plans
  are rejected and repaired rather than deadlocking.
* **execute_wave** — every task whose dependencies are met runs **concurrently**
  (bounded by `VISION_MAX_CONCURRENCY`).
* **review** — the Manager grades each output and can reject it, attach
  feedback, and re-assign it to a different specialist.
* **synthesize / reflect** — merges the deliverable, then writes lessons and
  preference updates back into long-term memory.

---

## The five modules

### 1. Digital Twin (`backend/app/memory/`)
Persistent ChromaDB memory of episodes, preferences, patterns, lessons and
feedback. Recall re-ranks by `relevance × confidence × importance × recency`
(30-day half-life). Repeated observations **reinforce** an existing belief —
confidence climbs toward 1.0 instead of spawning duplicates.

### 2. YouTube Skill Acquisition (`backend/app/skills/`)
`URL → metadata → transcript → timestamped chunks → Skill DB`. Captions are
tried first (free, instant); Whisper is the fallback, with oversized audio
auto-split under the 25 MB limit. **Skill Activation** injects retrieved
know-how — with deep-linked timestamps — into the system prompt before acting.

### 3. Goal Mode (`backend/app/agents/`)
LangGraph DAG, dynamic sub-agent spawning, concurrent waves, manager review and
retry with re-assignment. Every run streams events over WebSocket.

### 4. Constitutional AI (`backend/app/guardian/`)
Four layers, cheapest first:

| Layer | Catches |
|---|---|
| `path_jail` | any read/write outside the workspace |
| `pattern_filter` | `rm -rf /`, `mkfs`, fork bombs, credential exfiltration, `curl \| bash`, offensive tooling |
| `guardian_llm` | semantic violations the regexes miss |
| `alternative` | every denial must propose a legal path forward |

The pattern layer is **not** delegated to the model: `rm -rf /` is blocked even
if the LLM is offline or jailbroken. Guardian errors **fail closed**.

### 5. Proactive Intuition & multi-modal output
Context-triggered suggestions gated by cooldown, confidence floor and
de-duplication. Agents write files, render charts (matplotlib) and emit
markdown that the UI renders safely as React elements — never `innerHTML`.

---

## Security model

* Code execution defaults to an **ephemeral Docker container**: no network,
  read-only rootfs, dropped capabilities, non-root user, memory/CPU/PID caps.
* If Docker is configured but unavailable, Vision AI **refuses to run the
  command** rather than silently falling back to the host.
* All filesystem access is jailed to `data/workspace`, traversal included.
* Every verdict is recorded in an audit log exposed at `/api/guardian/audit`.

> `VISION_EXECUTOR=subprocess` trades isolation for convenience, and
> `VISION_GUARDIAN_MODE=monitor` logs violations instead of blocking them.
> Both are for local experiments only.

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | providers, executor, guardian, memory/skill counts |
| `POST` | `/api/goal` | run Goal Mode synchronously |
| `WS` | `/ws/goal` | stream plan, tasks, verdicts, reflection live |
| `POST` | `/api/skills/learn` | ingest a YouTube video |
| `POST` | `/api/skills/activate` | preview the injected skill prompt |
| `POST` | `/api/memory/recall` | query the Digital Twin |
| `POST` | `/api/memory/reflect` | run the reflection loop manually |
| `GET` | `/api/guardian/constitution` | the machine-readable constitution |
| `POST` | `/api/guardian/check` | screen an action without executing it |
| `POST` | `/api/intuition` | ask for a proactive suggestion |
| `GET` | `/api/artifacts` | list files the agent produced |

Interactive docs at `http://localhost:8000/docs`.

---

## Layout

```
backend/app/
  main.py            FastAPI app, REST + WebSocket
  config.py          env-driven settings, graceful key-less degradation
  core/              llm.py (provider-agnostic) · embeddings.py · executor.py
  memory/            vector_store.py · digital_twin.py (+ reflection loop)
  skills/            youtube_ingest.py · skill_store.py
  agents/            goal_graph.py · sub_agents.py · tools.py · intuition.py
  guardian/          constitution.py · guardian.py
frontend/src/        App.tsx · components/ · lib/api.ts · lib/markdown.tsx
tests/               76 tests (guardian, goal DAG, skills, memory)
scripts/demo.py      end-to-end offline demo
```

## Testing

```bash
pytest                      # everything, offline
pytest tests/test_guardian.py -v     # adversarial security cases
```

Coverage highlights: 18 dangerous commands blocked with 0 false positives on 11
benign ones, DAG cycle/dangling/duplicate detection, verified **parallel**
wave execution, manager rejection + re-assignment, skill retrieval precision,
and preference reinforcement.

## Roadmap

- Persist Goal Mode runs so long jobs survive a restart (LangGraph checkpointer)
- Human-in-the-loop approval UI for `needs_approval` verdicts
- Web search + browser tools for the Research Agent
- Multi-user auth with per-user memory namespaces
