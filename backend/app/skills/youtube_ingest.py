"""Module 2 (part 1) — YouTube ingestion pipeline.

    URL -> metadata -> transcript -> chunks -> Skill Vector DB

Transcript strategies, in order of cost:

1. `captions`    — pull YouTube's own subtitles with yt-dlp. Free, no ffmpeg,
                   no API key, and instant. Tried first in `auto` mode.
2. `whisper_api` — download bestaudio, then transcribe with OpenAI Whisper.
                   Costs money and needs ffmpeg, but works on any video and is
                   far more accurate on jargon-heavy content.

Long audio is split on duration so each part stays under Whisper's 25 MB limit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings

logger = logging.getLogger("vision.youtube")

_YT_ID_RE = re.compile(
    r"(?:v=|/shorts/|/embed/|youtu\.be/|/live/)([A-Za-z0-9_-]{11})"
)
WHISPER_LIMIT_BYTES = 24 * 1024 * 1024


class IngestError(RuntimeError):
    pass


@dataclass
class VideoMetadata:
    video_id: str
    title: str
    channel: str = ""
    duration: int = 0
    url: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class TranscriptSegment:
    text: str
    start: float = 0.0
    end: float = 0.0


@dataclass
class Transcript:
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    method: str = "unknown"
    language: str = "en"


def extract_video_id(url: str) -> str:
    m = _YT_ID_RE.search(url)
    if m:
        return m.group(1)
    bare = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", bare):
        return bare
    raise IngestError(f"Could not extract a YouTube video id from: {url!r}")


def normalize_url(url: str) -> str:
    return f"https://www.youtube.com/watch?v={extract_video_id(url)}"


# ---------------------------------------------------------------------------
# Caption parsing
# ---------------------------------------------------------------------------

def _parse_vtt(content: str) -> list[TranscriptSegment]:
    """Parse WebVTT, de-duplicating YouTube's rolling auto-caption repeats."""
    segments: list[TranscriptSegment] = []
    ts_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
    )
    cur_start = cur_end = 0.0
    buf: list[str] = []
    last_text = ""

    def flush() -> None:
        nonlocal buf, last_text
        if not buf:
            return
        text = " ".join(buf).strip()
        text = re.sub(r"<[^>]+>", "", text)          # inline karaoke tags
        text = re.sub(r"\s+", " ", text).strip()
        # Auto-captions repeat the previous line as context; drop that overlap.
        if text and text != last_text:
            if last_text and text.startswith(last_text):
                text = text[len(last_text) :].strip()
            if text:
                segments.append(TranscriptSegment(text=text, start=cur_start, end=cur_end))
                last_text = text
        buf = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        m = ts_re.search(line)
        if m:
            flush()
            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
            cur_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            cur_end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        elif line.isdigit():
            continue
        else:
            buf.append(line)
    flush()
    return segments


def _parse_json3(content: str) -> list[TranscriptSegment]:
    data = json.loads(content)
    out: list[TranscriptSegment] = []
    for event in data.get("events", []):
        segs = event.get("segs") or []
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue
        start = event.get("tStartMs", 0) / 1000.0
        dur = event.get("dDurationMs", 0) / 1000.0
        out.append(TranscriptSegment(text=text, start=start, end=start + dur))
    return out


class YouTubeIngestor:
    """Fetch metadata + transcript for a video."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache = self.settings.audio_cache
        self.cache.mkdir(parents=True, exist_ok=True)

    # -- metadata --------------------------------------------------------
    def _fetch_metadata_sync(self, url: str) -> VideoMetadata:
        import yt_dlp

        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return VideoMetadata(
            video_id=info.get("id", extract_video_id(url)),
            title=info.get("title", "Untitled"),
            channel=info.get("uploader") or info.get("channel", ""),
            duration=int(info.get("duration") or 0),
            url=url,
            description=(info.get("description") or "")[:2000],
            tags=list(info.get("tags") or [])[:20],
        )

    async def fetch_metadata(self, url: str) -> VideoMetadata:
        return await asyncio.to_thread(self._fetch_metadata_sync, normalize_url(url))

    # -- strategy 1: captions -------------------------------------------
    def _fetch_captions_sync(self, url: str, video_id: str) -> Transcript | None:
        import yt_dlp

        outdir = self.cache / video_id
        outdir.mkdir(parents=True, exist_ok=True)
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB", "en-orig"],
            "subtitlesformat": "json3/vtt/best",
            "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as exc:
            logger.warning("caption download failed: %s", exc)
            return None

        files = sorted(outdir.glob(f"{video_id}*"))
        for path in files:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            segments: list[TranscriptSegment] = []
            if path.suffix == ".json3" or content.lstrip().startswith("{"):
                try:
                    segments = _parse_json3(content)
                except (json.JSONDecodeError, KeyError):
                    segments = []
            if not segments and ("-->" in content):
                segments = _parse_vtt(content)
            if segments:
                text = " ".join(s.text for s in segments).strip()
                if len(text) > 80:
                    return Transcript(text=text, segments=segments, method="captions")
        return None

    async def fetch_captions(self, url: str, video_id: str) -> Transcript | None:
        return await asyncio.to_thread(self._fetch_captions_sync, url, video_id)

    # -- strategy 2: audio + whisper -------------------------------------
    def _download_audio_sync(self, url: str, video_id: str) -> Path:
        import yt_dlp

        outdir = self.cache / video_id
        outdir.mkdir(parents=True, exist_ok=True)
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": str(outdir / "audio.%(ext)s"),
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "64"}
            ],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        candidates = sorted(
            (p for p in outdir.glob("audio.*") if p.suffix != ".part"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not candidates:
            raise IngestError("Audio download produced no file (is ffmpeg installed?).")
        return candidates[0]

    async def download_audio(self, url: str, video_id: str) -> Path:
        return await asyncio.to_thread(self._download_audio_sync, normalize_url(url), video_id)

    async def transcribe_with_whisper(self, audio: Path) -> Transcript:
        if not self.settings.openai_api_key:
            raise IngestError("Whisper transcription requires OPENAI_API_KEY.")
        parts = await self._split_audio(audio)
        texts: list[str] = []
        segments: list[TranscriptSegment] = []
        offset = 0.0
        for part in parts:
            data = await self._whisper_call(part)
            texts.append(data.get("text", ""))
            for seg in data.get("segments") or []:
                segments.append(
                    TranscriptSegment(
                        text=seg.get("text", "").strip(),
                        start=float(seg.get("start", 0)) + offset,
                        end=float(seg.get("end", 0)) + offset,
                    )
                )
            offset += float(data.get("duration", 0) or 0)
        return Transcript(
            text=" ".join(t.strip() for t in texts).strip(),
            segments=segments,
            method="whisper_api",
        )

    async def _whisper_call(self, audio: Path) -> dict[str, Any]:
        import httpx

        base = (self.settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        async with httpx.AsyncClient(timeout=600.0) as client:
            with audio.open("rb") as fh:
                resp = await client.post(
                    f"{base}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                    files={"file": (audio.name, fh, "application/octet-stream")},
                    data={
                        "model": self.settings.whisper_model,
                        "response_format": "verbose_json",
                    },
                )
        if resp.status_code >= 400:
            raise IngestError(f"Whisper API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    async def _split_audio(self, audio: Path) -> list[Path]:
        """Split oversized audio into ~10 min chunks so Whisper accepts it."""
        if audio.stat().st_size <= WHISPER_LIMIT_BYTES:
            return [audio]
        outdir = audio.parent / "parts"
        outdir.mkdir(exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(audio),
            "-f", "segment", "-segment_time", "600", "-c", "copy",
            str(outdir / "part_%03d.mp3"),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        parts = sorted(outdir.glob("part_*.mp3"))
        if proc.returncode != 0 or not parts:
            raise IngestError(f"ffmpeg split failed: {err.decode('utf-8', 'replace')[:300]}")
        logger.info("split oversized audio into %d parts", len(parts))
        return parts

    # -- orchestration ---------------------------------------------------
    async def get_transcript(self, url: str, video_id: str) -> Transcript:
        mode = self.settings.transcriber
        if mode in ("captions", "auto"):
            captions = await self.fetch_captions(normalize_url(url), video_id)
            if captions:
                logger.info("transcript via captions (%d chars)", len(captions.text))
                return captions
            if mode == "captions":
                raise IngestError(
                    "No captions available. Set VISION_TRANSCRIBER=whisper_api "
                    "(needs OPENAI_API_KEY + ffmpeg)."
                )
            logger.info("no captions; falling back to Whisper")
        audio = await self.download_audio(url, video_id)
        return await self.transcribe_with_whisper(audio)

    def cleanup(self, video_id: str) -> None:
        import shutil

        shutil.rmtree(self.cache / video_id, ignore_errors=True)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(
    transcript: Transcript, *, max_chars: int = 1200, overlap: int = 200
) -> list[dict[str, Any]]:
    """Chunk on sentence boundaries, preserving timestamps for citation.

    Timestamps matter: they let Vision AI say *"at 04:12 the tutorial does X"*
    and deep-link the user back to the exact moment.
    """
    if transcript.segments:
        return _chunk_segments(transcript.segments, max_chars, overlap)
    return _chunk_plain(transcript.text, max_chars, overlap)


def _chunk_segments(
    segments: list[TranscriptSegment], max_chars: int, overlap: int
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    buf: list[TranscriptSegment] = []
    size = 0
    for seg in segments:
        if size + len(seg.text) > max_chars and buf:
            chunks.append(_emit(buf))
            keep, kept = [], 0
            for s in reversed(buf):  # carry tail forward for context overlap
                if kept >= overlap:
                    break
                keep.insert(0, s)
                kept += len(s.text)
            buf, size = keep, kept
        buf.append(seg)
        size += len(seg.text) + 1
    if buf:
        chunks.append(_emit(buf))
    return chunks


def _emit(buf: list[TranscriptSegment]) -> dict[str, Any]:
    return {
        "text": " ".join(s.text for s in buf).strip(),
        "start": buf[0].start,
        "end": buf[-1].end,
    }


def _chunk_plain(text: str, max_chars: int, overlap: int) -> list[dict[str, Any]]:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    chunks: list[dict[str, Any]] = []
    buf: list[str] = []
    size = 0
    for sent in sentences:
        if size + len(sent) > max_chars and buf:
            chunks.append({"text": " ".join(buf).strip(), "start": 0.0, "end": 0.0})
            tail, kept = [], 0
            for s in reversed(buf):
                if kept >= overlap:
                    break
                tail.insert(0, s)
                kept += len(s)
            buf, size = tail, kept
        buf.append(sent)
        size += len(sent) + 1
    if buf:
        chunks.append({"text": " ".join(buf).strip(), "start": 0.0, "end": 0.0})
    return [c for c in chunks if c["text"]]


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
