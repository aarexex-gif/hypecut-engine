"""HYPECUT live highlight intelligence primitives.

Consumes short media windows produced by an authorised live ingest source.
This module deliberately does not fetch or rip YouTube URLs; the ingest layer
must supply a media file/segment that HYPECUT is permitted to process.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np

log = logging.getLogger("hypecut-live-intelligence")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TEXT_MODEL = os.getenv("HYPECUT_HIGHLIGHT_MODEL", "gpt-4o-mini")


@dataclass
class HighlightCandidate:
    start_seconds: float
    end_seconds: float
    score: float
    category: str
    hook: str
    title: str
    caption: str
    transcript: str
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or "media command failed")[-2000:])
    return result


def audio_features(media: Path) -> dict[str, float]:
    """Compute coarse loudness/energy features as signals, never as the sole rank."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(media), "-vn", "-ac", "1",
         "-ar", "8000", "-f", "f32le", "-"],
        capture_output=True, timeout=180,
    )
    if proc.returncode or not proc.stdout:
        return {"rms": 0.0, "peak": 0.0, "dynamic_range": 0.0}
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if not len(samples):
        return {"rms": 0.0, "peak": 0.0, "dynamic_range": 0.0}
    windows = [samples[i:i+8000] for i in range(0, len(samples), 8000) if len(samples[i:i+8000])]
    rms = np.asarray([float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) for w in windows])
    return {
        "rms": float(np.mean(rms)),
        "peak": float(np.max(rms)),
        "dynamic_range": float(np.percentile(rms, 95) - np.percentile(rms, 25)),
    }


def scene_change_count(media: Path) -> int:
    """Count rough shot changes; used as a context signal, not a quality verdict."""
    result = _run([
        "ffmpeg", "-hide_banner", "-i", str(media), "-vf",
        "select='gt(scene,0.32)',showinfo", "-an", "-f", "null", "-"
    ], timeout=180)
    return len(re.findall(r"showinfo", result.stderr))


async def transcribe_window(media: Path) -> str:
    if not OPENAI_API_KEY:
        return ""
    audio = media.with_suffix(".mp3")
    _run(["ffmpeg", "-y", "-i", str(media), "-vn", "-ac", "1",
          "-b:a", "64k", str(audio)], timeout=180)
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with audio.open("rb") as stream:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    data={"model": "whisper-1", "response_format": "text"},
                    files={"file": (audio.name, stream, "audio/mpeg")},
                )
            response.raise_for_status()
            return response.text.strip()
    finally:
        audio.unlink(missing_ok=True)


async def _post_with_retries(client: httpx.AsyncClient, url: str, *, attempts: int = 3, **kwargs):
    """Retry transient model API failures without retrying invalid requests."""
    for attempt in range(max(1, attempts)):
        try:
            response = await client.post(url, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < max(1, attempts):
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        delay = min(20.0, max(0.5, float(retry_after)))
                    except (TypeError, ValueError):
                        delay = min(8.0, 2.0 ** attempt)
                    await __import__("asyncio").sleep(delay)
                    continue
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt + 1 >= max(1, attempts):
                raise
            await __import__("asyncio").sleep(min(8.0, 2.0 ** attempt))
    raise RuntimeError("Model API request exhausted retry attempts")


def _keyframe_data_urls(media: Path, duration: float) -> list[str]:
    """Extract three representative visual frames from a short analysis window."""
    urls: list[str] = []
    sample_points = [duration * 0.2, duration * 0.5, duration * 0.8]
    for index, second in enumerate(sample_points):
        frame = media.with_name(f"{media.stem}-frame-{index}.jpg")
        try:
            _run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{max(0.0, second):.3f}", "-i", str(media),
                "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5", str(frame),
            ], timeout=60)
            if frame.exists() and frame.stat().st_size:
                encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
                urls.append(f"data:image/jpeg;base64,{encoded}")
        except Exception:
            log.exception("Could not extract visual review frame from %s", media.name)
        finally:
            frame.unlink(missing_ok=True)
    return urls


def normalise_facecam_box(value: Any) -> tuple[float, float, float, float] | None:
    """Validate a model-detected facecam box in normalised x,y,width,height coordinates."""
    if not isinstance(value, dict):
        return None
    try:
        x, y, width, height = (float(value[k]) for k in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (x, y, width, height)):
        return None
    x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
    width, height = min(width, 1.0 - x), min(height, 1.0 - y)
    if width < 0.08 or height < 0.08 or width > 0.85 or height > 0.85:
        return None
    return (round(x, 4), round(y, 4), round(width, 4), round(height, 4))


async def detect_facecam_region(media: Path) -> tuple[float, float, float, float] | None:
    """Ask vision to locate a persistent streamer facecam panel; fail safely if absent."""
    if not OPENAI_API_KEY:
        return None
    duration_result = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(media)
    ])
    duration = max(0.0, float(duration_result.stdout.strip() or 0))
    if duration < 2:
        return None
    frames = await __import__("asyncio").to_thread(_keyframe_data_urls, media, duration)
    if not frames:
        return None
    content = [{
        "type": "text",
        "text": (
            "Find the streamer webcam/facecam panel, if one is clearly visible, in these "
            "chronological frames from the same gaming video. Return JSON only: "
            "{\"found\":true,\"x\":0.0,\"y\":0.0,\"width\":0.2,\"height\":0.3}. "
            "Coordinates are normalised 0..1 relative to the full source frame, with x,y "
            "at the top-left. Prefer the persistent facecam panel, not an in-game character. "
            "If absent or uncertain return {\"found\":false}."
        )
    }]
    content.extend({"type": "image_url", "image_url": {"url": url, "detail": "high"}} for url in frames)
    async with httpx.AsyncClient(timeout=90) as client:
        response = await _post_with_retries(
            client, "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": TEXT_MODEL, "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You locate visible webcam panels precisely. Never guess."},
                    {"role": "user", "content": content},
                ],
            },
        )
    result = json.loads(response.json()["choices"][0]["message"]["content"])
    return normalise_facecam_box(result) if result.get("found") is True else None


async def rank_window(media: Path, window_start: float = 0.0) -> HighlightCandidate | None:
    """Rank a 30–60s live window using transcript semantics plus AV signals."""
    duration_result = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(media)
    ])
    duration = float(duration_result.stdout.strip() or 0)
    if duration < 20:
        return None
    transcript = await transcribe_window(media)
    if not transcript or len(transcript.split()) < 8:
        return None
    features = await __import__("asyncio").to_thread(audio_features, media)
    scenes = await __import__("asyncio").to_thread(scene_change_count, media)
    if not OPENAI_API_KEY:
        return None
    frame_urls = await __import__("asyncio").to_thread(_keyframe_data_urls, media, duration)
    prompt = {
        "transcript": transcript[:12000],
        "audio_features": features,
        "scene_change_count": scenes,
        "visual_frames_attached": len(frame_urls),
        "duration_seconds": duration,
        "task": (
            "Evaluate this livestream excerpt for a compelling 30-60 second social short. "
            "Use BOTH the transcript and attached chronological video frames. Verify that "
            "the visible action/context supports the spoken words; do not infer unseen events. "
            "Score genuine viewer value: humour, surprise, strong opinion, emotional reaction, "
            "useful insight, impressive action, conflict with payoff, or memorable story. "
            "Penalise missing context, repetition, silence, rambling and unclear speech. "
            "Do not mistake loudness for quality. Return strict JSON with score 0-100, "
            "category, hook (max 12 words), title (max 70 chars), caption (max 180 chars), "
            "reasons (3 concise strings), and keep (boolean)."
        ),
    }
    user_content = [{"type": "text", "text": json.dumps(prompt)}]
    user_content.extend(
        {"type": "image_url", "image_url": {"url": url, "detail": "low"}}
        for url in frame_urls
    )
    async with httpx.AsyncClient(timeout=90) as client:
        response = await _post_with_retries(
            client,
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": TEXT_MODEL,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a strict audiovisual short-form video highlight editor. Never invent events absent from the transcript or attached frames."},
                    {"role": "user", "content": user_content},
                ],
            },
        )
        data = response.json()
    raw = data["choices"][0]["message"]["content"]
    verdict = json.loads(raw)
    score = max(0.0, min(100.0, float(verdict.get("score", 0))))
    if not verdict.get("keep") or score < 72:
        return None
    return HighlightCandidate(
        start_seconds=round(window_start, 3),
        end_seconds=round(window_start + min(60.0, duration), 3),
        score=round(score, 1),
        category=str(verdict.get("category") or "highlight")[:40],
        hook=str(verdict.get("hook") or "")[:120],
        title=str(verdict.get("title") or "HYPECUT highlight")[:180],
        caption=str(verdict.get("caption") or "")[:300],
        transcript=transcript[:12000],
        reasons=[str(x)[:160] for x in (verdict.get("reasons") or [])[:3]],
    )
