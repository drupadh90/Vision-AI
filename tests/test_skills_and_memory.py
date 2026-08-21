"""YouTube ingestion parsing, Skill Activation, and Digital Twin memory."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.embeddings import HashEmbedder
from app.core.llm import LLMClient, extract_json
from app.memory.digital_twin import (
    DigitalTwinMemory,
    MemoryItem,
    MemoryKind,
    ReflectionLoop,
)
from app.skills.skill_store import SkillLibrary
from app.skills.youtube_ingest import (
    IngestError,
    Transcript,
    TranscriptSegment,
    _parse_json3,
    _parse_vtt,
    chunk_transcript,
    extract_video_id,
    format_timestamp,
)

DAVINCI = (
    "Welcome to this DaVinci Resolve tutorial. First import your footage into the "
    "media pool. Use the blade tool, shortcut B, to cut clips at the playhead. To "
    "color grade, switch to the Color page and use the primary color wheels: lift "
    "for shadows, gamma for midtones, gain for highlights. Add a serial node with "
    "Alt-S so grading stays non-destructive. Deliver with H.264 at 20 megabits."
)
PANDAS = (
    "In this pandas tutorial we load a CSV with read_csv, clean missing values with "
    "fillna, group rows using groupby on the region column, aggregate revenue with "
    "sum, and plot a bar chart with matplotlib pyplot to visualise revenue by region."
)


# ---------------------------------------------------------------- ingestion

@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ],
)
def test_video_id_extraction(url: str) -> None:
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_invalid_url_raises() -> None:
    with pytest.raises(IngestError):
        extract_video_id("https://example.com/not-a-video")


def test_vtt_parsing_dedupes_rolling_captions() -> None:
    vtt = """WEBVTT
Kind: captions

00:00:00.120 --> 00:00:03.500
hey everyone welcome back

00:00:03.500 --> 00:00:07.000
hey everyone welcome back
today we are editing

00:00:07.000 --> 00:00:11.240
today we are editing
with <c>davinci</c> resolve
"""
    segments = _parse_vtt(vtt)
    joined = " ".join(s.text for s in segments)
    assert joined.count("welcome back") == 1, "rolling caption duplication not removed"
    assert "<c>" not in joined, "inline caption tags not stripped"
    assert segments[0].start == pytest.approx(0.12, abs=0.01)


def test_json3_parsing() -> None:
    payload = (
        '{"events":[{"tStartMs":0,"dDurationMs":2000,'
        '"segs":[{"utf8":"hello "},{"utf8":"world"}]}]}'
    )
    segments = _parse_json3(payload)
    assert segments[0].text == "hello world"


def test_chunking_preserves_timestamps_and_overlap() -> None:
    segments = [
        TranscriptSegment(text=f"step number {i} of the tutorial.", start=i * 3.0, end=i * 3.0 + 3)
        for i in range(40)
    ]
    chunks = chunk_transcript(Transcript(text="", segments=segments), max_chars=300, overlap=80)
    assert len(chunks) > 1
    assert chunks[0]["end"] > chunks[0]["start"]
    assert chunks[1]["start"] < chunks[0]["end"], "no context overlap between chunks"


def test_plain_text_chunking() -> None:
    text = " ".join(f"Sentence {i} is here." for i in range(120))
    chunks = chunk_transcript(Transcript(text=text), max_chars=300, overlap=60)
    assert len(chunks) > 1
    assert all(c["text"] for c in chunks)


def test_timestamp_formatting() -> None:
    assert format_timestamp(75) == "1:15"
    assert format_timestamp(3725) == "1:02:05"


# --------------------------------------------------------------- embeddings

@pytest.mark.asyncio
async def test_hash_embeddings_are_deterministic_and_normalised() -> None:
    e = HashEmbedder()
    a, b = await e.embed(["color grading tutorial", "color grading tutorial"])
    assert a == b
    assert abs(sum(x * x for x in a) ** 0.5 - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_embeddings_separate_topics() -> None:
    e = HashEmbedder()
    doc, close, far = await e.embed(
        [DAVINCI, "how do I color grade with the color wheels?", "how to bake sourdough bread"]
    )
    sim_close = sum(x * y for x, y in zip(doc, close))
    sim_far = sum(x * y for x, y in zip(doc, far))
    assert sim_close > sim_far + 0.1, "relevant and irrelevant queries are not separated"


# -------------------------------------------------------------------- skills

@pytest.fixture()
def settings(tmp_path) -> Settings:
    s = Settings(
        VISION_LLM_PROVIDER="mock",
        VISION_CHROMA_PATH=str(tmp_path / "chroma"),
        VISION_WORKSPACE_DIR=str(tmp_path / "ws"),
        VISION_AUDIO_CACHE=str(tmp_path / "audio"),
    )
    s.ensure_dirs()
    return s


@pytest.mark.asyncio
async def test_skill_activation_retrieves_right_skill(settings: Settings) -> None:
    lib = SkillLibrary(settings)
    await lib.add_text_skill("DaVinci Resolve editing", DAVINCI)
    await lib.add_text_skill("Pandas CSV analysis", PANDAS)

    prompt, matches = await lib.activate("how do I color grade my footage?")
    assert matches, "no skill activated for a clearly relevant query"
    assert "DaVinci" in matches[0].title
    assert "Activated skills" in prompt
    assert "youtube.com/watch" in prompt  # timestamped citation

    prompt2, matches2 = await lib.activate("group revenue by region and plot a bar chart")
    assert matches2 and "Pandas" in matches2[0].title


@pytest.mark.asyncio
async def test_irrelevant_query_activates_nothing(settings: Settings) -> None:
    lib = SkillLibrary(settings)
    await lib.add_text_skill("DaVinci Resolve editing", DAVINCI)
    prompt, matches = await lib.activate("what is the capital of France?")
    assert matches == []
    assert prompt == ""


@pytest.mark.asyncio
async def test_skill_listing_and_forgetting(settings: Settings) -> None:
    lib = SkillLibrary(settings)
    report = await lib.add_text_skill("Temporary skill", PANDAS)
    assert len(await lib.list_skills()) == 1
    await lib.forget(report.skill_id)
    assert await lib.list_skills() == []


# -------------------------------------------------------------- digital twin

@pytest.mark.asyncio
async def test_memory_roundtrip(settings: Settings) -> None:
    mem = DigitalTwinMemory(settings)
    await mem.remember(
        MemoryItem(content="User prefers dark mode interfaces", kind=MemoryKind.PREFERENCE)
    )
    found = await mem.recall("what theme does the user like?", k=3)
    assert any("dark mode" in m["content"] for m in found)


@pytest.mark.asyncio
async def test_preference_reinforcement_merges_not_duplicates(settings: Settings) -> None:
    mem = DigitalTwinMemory(settings)
    obs = "User prefers concise bullet-point summaries over long prose"
    first = await mem.reinforce_preference(obs, 0.5, user_id="u1")
    again = await mem.reinforce_preference(obs, 0.7, user_id="u1")
    assert first == again, "repeat observation created a duplicate memory"

    prefs = await mem.recall(obs, k=5, kinds=[MemoryKind.PREFERENCE], user_id="u1")
    assert len({p["id"] for p in prefs}) == 1
    assert prefs[0]["confidence"] > 0.5, "confidence did not strengthen with evidence"


@pytest.mark.asyncio
async def test_twin_context_renders_prompt_block(settings: Settings) -> None:
    mem = DigitalTwinMemory(settings)
    await mem.remember(
        MemoryItem(content="User writes Python, not Java", kind=MemoryKind.PREFERENCE,
                   confidence=0.9)
    )
    ctx = await mem.twin_context("write me a script")
    assert "Digital Twin context" in ctx
    assert "Python" in ctx


@pytest.mark.asyncio
async def test_twin_context_empty_when_no_memories(settings: Settings) -> None:
    assert await DigitalTwinMemory(settings).twin_context("anything") == ""


@pytest.mark.asyncio
async def test_reflection_writes_lessons_and_preferences(settings: Settings) -> None:
    mem = DigitalTwinMemory(settings)
    loop = ReflectionLoop(mem, LLMClient(settings))
    result = await loop.reflect(task="Build a chart", outcome="Chart delivered.", success=True)
    assert result["stored"] >= 2
    assert result.get("lesson")
    stats = await mem.stats()
    assert stats["by_kind"]["lesson"] >= 1
    assert stats["by_kind"]["episode"] >= 1


@pytest.mark.asyncio
async def test_memory_isolated_per_user(settings: Settings) -> None:
    mem = DigitalTwinMemory(settings)
    await mem.remember(MemoryItem(content="Alice likes tabs", kind=MemoryKind.PREFERENCE,
                                  user_id="alice"))
    bob = await mem.recall("indentation preference", k=5, user_id="bob")
    assert all("Alice" not in m["content"] for m in bob)


# ----------------------------------------------------------------- json utils

def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```\nDone.') == {"a": 2}
    assert extract_json('Here you go: {"a": {"b": [1,2]}} cheers') == {"a": {"b": [1, 2]}}
    assert extract_json('text {"s": "a } brace"} more') == {"s": "a } brace"}


def test_extract_json_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        extract_json("no json at all here")
