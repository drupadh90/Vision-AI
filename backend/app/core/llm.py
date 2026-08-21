"""Provider-agnostic LLM layer.

Every reasoning component in Vision AI (planner, workers, Guardian, reflection)
talks to this interface and never to a vendor SDK directly. That means:

* you can run the Guardian on a cheap local model and the planner on GPT-4o;
* swapping vendors is a config change, not a refactor;
* the `MockProvider` makes the entire system testable and demoable offline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings, get_settings

logger = logging.getLogger("vision.llm")


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """Raised when a provider call fails irrecoverably."""


# ---------------------------------------------------------------------------
# JSON coercion — LLMs love to wrap JSON in prose and code fences.
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Best-effort structured extraction from a model response.

    Tries: raw parse -> fenced block -> first balanced {...} / [...] span.
    Raises ValueError if nothing parses, so callers can retry or fall back.
    """
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(m.strip() for m in _FENCE_RE.findall(text))

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON object found in model output: {text[:400]!r}")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class BaseLLMProvider(ABC):
    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def complete(
        self, messages: list[Message], *, model: str, temperature: float, max_tokens: int
    ) -> LLMResponse: ...

    async def aclose(self) -> None:  # pragma: no cover - overridden where needed
        return None


class OpenAIProvider(BaseLLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        base = settings.openai_base_url or "https://api.openai.com/v1"
        self._client = httpx.AsyncClient(
            base_url=base.rstrip("/"),
            timeout=settings.llm_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        )

    async def complete(
        self, messages: list[Message], *, model: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        payload = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = await _post_json(self._client, "/chat/completions", payload)
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=data["choices"][0]["message"]["content"] or "",
            model=model,
            provider=self.name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            timeout=settings.llm_timeout_seconds,
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
        )

    async def complete(
        self, messages: list[Message], *, model: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        # Anthropic takes the system prompt out-of-band.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m.as_dict() for m in messages if m.role != "system"]
        payload: dict[str, Any] = {
            "model": model,
            "messages": turns or [{"role": "user", "content": system or "Hello"}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        data = await _post_json(self._client, "/messages", payload)
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=settings.llm_timeout_seconds,
        )

    async def complete(
        self, messages: list[Message], *, model: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        payload = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = await _post_json(self._client, "/api/chat", payload)
        return LLMResponse(
            text=(data.get("message") or {}).get("content", ""),
            model=model,
            provider=self.name,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class MockProvider(BaseLLMProvider):
    """Deterministic offline provider.

    This is not a toy: it inspects the prompt and returns *schema-valid* output
    for each role, so the LangGraph state machine, the Guardian and the
    reflection loop can all be exercised (and unit-tested) with zero API keys.
    """

    name = "mock"

    async def complete(
        self, messages: list[Message], *, model: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        await asyncio.sleep(0)  # stay genuinely async
        joined = "\n".join(m.content for m in messages)
        marker = _role_marker(joined)
        text = _MOCK_HANDLERS[marker](joined)
        return LLMResponse(
            text=text,
            model=f"mock:{model}",
            provider=self.name,
            prompt_tokens=len(joined) // 4,
            completion_tokens=len(text) // 4,
        )


def _role_marker(prompt: str) -> str:
    p = prompt.lower()
    if "vision_role: guardian" in p:
        return "guardian"
    if "vision_role: planner" in p:
        # The Manager wears three hats under one role marker; each needs a
        # differently-shaped response.
        if "reviewing a completed sub-task" in p:
            return "review"
        if "producing the final deliverable" in p:
            return "synthesis"
        return "planner"
    if "vision_role: reflection" in p:
        return "reflection"
    if "vision_role: intuition" in p:
        return "intuition"
    return "worker"


def _mock_planner(prompt: str) -> str:
    goal = _extract_tagged(prompt, "GOAL") or "the objective"
    short = goal.strip().splitlines()[0][:120]
    plan = {
        "strategy": f"Decompose '{short}' into research, build and synthesis phases.",
        "tasks": [
            {
                "id": "t1",
                "title": f"Research background for: {short}",
                "description": f"Gather the key facts, constraints and prior art needed for: {short}",
                "agent": "research",
                "depends_on": [],
            },
            {
                "id": "t2",
                "title": "Produce the core artifact",
                "description": f"Using the research, build the main deliverable for: {short}",
                "agent": "coder",
                "depends_on": ["t1"],
            },
            {
                "id": "t3",
                "title": "Write up the result",
                "description": "Summarise the outcome into a clear, well-structured final answer.",
                "agent": "writer",
                "depends_on": ["t2"],
            },
        ],
    }
    return json.dumps(plan, indent=2)


def _mock_guardian(prompt: str) -> str:
    action = (_extract_tagged(prompt, "ACTION") or prompt).lower()
    red_flags = [
        "rm -rf /",
        "mkfs",
        "dd if=",
        ":(){",
        "shutdown",
        "/etc/shadow",
        "id_rsa",
        "ddos",
        "exploit",
        "keylog",
        "ransom",
        "botnet",
        "sql injection",
        "brute force",
    ]
    hit = next((f for f in red_flags if f in action), None)
    if hit:
        verdict = {
            "decision": "deny",
            "confidence": 0.95,
            "violated_principles": ["P1_no_harm", "P3_system_integrity"],
            "reason": f"Action contains the destructive/malicious pattern '{hit}'.",
            "safe_alternative": "Describe the intended outcome and use a reversible, "
            "workspace-scoped operation instead.",
        }
    else:
        verdict = {
            "decision": "allow",
            "confidence": 0.82,
            "violated_principles": [],
            "reason": "No constitutional violation detected in the proposed action.",
            "safe_alternative": None,
        }
    return json.dumps(verdict)


def _mock_reflection(prompt: str) -> str:
    return json.dumps(
        {
            "summary": "Task completed via plan-execute-review; decomposition held up well.",
            "what_worked": ["Dependency-ordered execution", "Skill context injection"],
            "what_failed": ["Initial task titles were vaguer than ideal"],
            "user_preferences": [
                {
                    "observation": "User favours concise, structured deliverables.",
                    "confidence": 0.6,
                }
            ],
            "behavioral_adjustments": [
                "Lead with the artifact, keep narration short.",
            ],
            "lesson": "Front-load concrete deliverables; keep commentary minimal.",
        }
    )


def _mock_intuition(prompt: str) -> str:
    return json.dumps(
        {
            "should_speak": True,
            "confidence": 0.62,
            "suggestion": "I have a related skill in my YouTube Skill DB — want me to apply it?",
            "rationale": "Recent context overlaps with an acquired skill.",
        }
    )


def _mock_worker(prompt: str) -> str:
    task = _extract_tagged(prompt, "TASK") or "the assigned task"
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return (
        f"## Result for: {task.strip().splitlines()[0][:120]}\n\n"
        "Completed using the offline mock provider. Set `VISION_LLM_PROVIDER=openai` "
        "(plus `OPENAI_API_KEY`) for real reasoning.\n\n"
        f"- Deterministic run id: `{digest}`\n"
        "- Inputs from upstream tasks were incorporated.\n"
    )


def _mock_review(prompt: str) -> str:
    return json.dumps(
        {"accept": True, "quality": 0.85, "feedback": "", "reassign_to": None}
    )


def _mock_synthesis(prompt: str) -> str:
    goal = _extract_tagged(prompt, "GOAL") or "the objective"
    work = _extract_tagged(prompt, "COMPLETED_WORK") or ""
    sections = re.findall(r"^## (.+?)$", work, re.MULTILINE)
    lines = [
        f"# {goal.strip().splitlines()[0][:120]}",
        "",
        "Completed by Vision AI in Goal Mode. Sub-task outputs merged below.",
        "",
        "## What was done",
    ]
    lines += [f"- {s}" for s in sections] or ["- (no sub-task output)"]
    lines += [
        "",
        "## Notes",
        "This deliverable was produced by the offline mock provider, so the prose "
        "is illustrative rather than researched. Configure a real LLM provider "
        "(`VISION_LLM_PROVIDER=openai`) for substantive output.",
    ]
    return "\n".join(lines)


_MOCK_HANDLERS = {
    "planner": _mock_planner,
    "review": _mock_review,
    "synthesis": _mock_synthesis,
    "guardian": _mock_guardian,
    "reflection": _mock_reflection,
    "intuition": _mock_intuition,
    "worker": _mock_worker,
}


def _extract_tagged(prompt: str, tag: str) -> str | None:
    """Extract <TAG ...>body</TAG>, tolerating attributes on the opening tag."""
    m = re.search(rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", prompt, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


async def _post_json(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST with bounded exponential backoff on 429/5xx."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.post(path, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise LLMError(f"{resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, LLMError) as exc:
            last = exc
            if attempt == 2:
                break
            await asyncio.sleep(1.5 * (2**attempt))
    raise LLMError(f"LLM request to {path} failed after 3 attempts: {last}")


_PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
}


class LLMClient:
    """Role-aware façade over the configured provider."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        resolved = self.settings.effective_llm_provider()
        if resolved != self.settings.llm_provider:
            logger.warning(
                "LLM provider '%s' has no credentials — falling back to 'mock'.",
                self.settings.llm_provider,
            )
        self.provider_name = resolved
        self.provider = _PROVIDERS[resolved](self.settings)

    @property
    def is_mock(self) -> bool:
        return self.provider_name == "mock"

    async def complete(
        self,
        messages: list[Message],
        *,
        role: str = "worker",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = self.settings.model_for(role)
        return await self.provider.complete(
            messages,
            model=model,
            temperature=self.settings.llm_temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.settings.llm_max_tokens,
        )

    async def complete_json(
        self,
        messages: list[Message],
        *,
        role: str = "worker",
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 1,
    ) -> Any:
        """Complete and parse JSON, re-prompting once on malformed output."""
        attempt_messages = list(messages)
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            resp = await self.complete(
                attempt_messages, role=role, temperature=temperature, max_tokens=max_tokens
            )
            try:
                return extract_json(resp.text)
            except ValueError as exc:
                last_err = exc
                logger.warning("Malformed JSON from %s (attempt %d)", role, attempt + 1)
                attempt_messages = list(messages) + [
                    Message("assistant", resp.text[:1500]),
                    Message(
                        "user",
                        "That was not valid JSON. Reply with ONLY a single valid JSON "
                        "object. No prose, no markdown fences.",
                    ),
                ]
        raise LLMError(f"Could not obtain valid JSON for role '{role}': {last_err}")

    async def aclose(self) -> None:
        await self.provider.aclose()
