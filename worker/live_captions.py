"""Timestamped Whisper captions for accepted HYPECUT live highlights."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def _seconds(value: float) -> str:
    total = max(0, int(round(value * 1000)))
    hours, rem = divmod(total, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


async def transcribe_srt(media: Path, destination: Path) -> bool:
    if not OPENAI_API_KEY:
        return False
    audio = destination.with_suffix(".mp3")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
         "-vn", "-ac", "1", "-b:a", "64k", str(audio)],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode:
        raise RuntimeError((proc.stderr or "Audio extraction failed")[-1200:])
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with audio.open("rb") as stream:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    data={"model": "whisper-1", "response_format": "verbose_json"},
                    files={"file": (audio.name, stream, "audio/mpeg")},
                )
            response.raise_for_status()
        segments = response.json().get("segments") or []
        lines = []
        for index, segment in enumerate(segments, 1):
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            start = float(segment.get("start") or 0)
            end = max(start + 0.2, float(segment.get("end") or start + 0.5))
            lines.extend([str(index), f"{_seconds(start)} --> {_seconds(end)}", text, ""])
        if not lines:
            return False
        destination.write_text("\n".join(lines), encoding="utf-8")
        return True
    finally:
        audio.unlink(missing_ok=True)
