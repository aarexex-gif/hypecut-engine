"""Continuous HLS live-session monitor and semantic clip renderer.

Only consumes HLS media published to HYPECUT's authorised MediaMTX ingest.
It does not scrape or download arbitrary YouTube URLs.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable
from urllib.parse import quote

import httpx

from live_intelligence import rank_window
from live_captions import transcribe_srt

log = logging.getLogger("hypecut-live-pipeline")
HLS_BASE = os.getenv("HYPECUT_HLS_BASE_URL", "").rstrip("/")
MIN_SCORE = float(os.getenv("HYPECUT_LIVE_MIN_SCORE", "72"))
WINDOW_SECONDS = max(30, min(60, int(os.getenv("HYPECUT_LIVE_WINDOW_SECONDS", "60"))))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def ffmpeg_capture(playlist: str, output: Path) -> None:
    # Read a bounded live HLS window and remux/transcode to a seekable local file.
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", playlist, "-t", str(WINDOW_SECONDS),
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", "fps=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(output),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=WINDOW_SECONDS + 150)
    if proc.returncode:
        raise RuntimeError((proc.stderr or "HLS capture failed")[-1800:])


def render_vertical(source: Path, output: Path, captions: Path | None = None) -> None:
    # Stable center crop fallback. Subject-aware face tracking is a separate
    # enhancement; this does not pretend a single program feed has other angles.
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,setsar=1"
    )
    if captions and captions.exists():
        escaped = str(captions).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        vf += f",subtitles='{escaped}':force_style='FontName=Arial,FontSize=18,Outline=2,Shadow=1,Alignment=2,MarginV=240'"
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
        "-vf", vf, "-c:v", "libx264", "-threads", "2",
        "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(output),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=300)
    if proc.returncode:
        raise RuntimeError((proc.stderr or "Vertical rendering failed")[-1800:])


async def process_live_sessions(
    client: httpx.AsyncClient,
    rest_get: Callable[..., Awaitable[Any]],
    rest_patch: Callable[..., Awaitable[Any]],
    rest_post: Callable[..., Awaitable[Any]],
    upload_file: Callable[..., Awaitable[None]],
    worker_id: str,
) -> int:
    """Process one bounded window from each eligible session; return work count."""
    if not HLS_BASE:
        return 0
    sessions = await rest_get(client, "stream_sessions", {
        "select": "id,user_id,source_id,status,started_at,ingest_metadata",
        "status": "in.(connecting,live)",
        "order": "created_at.asc",
        "limit": "10",
    })
    processed = 0
    for session in sessions or []:
        metadata = session.get("ingest_metadata") or {}
        ingest_path = metadata.get("ingest_path")
        if not ingest_path or not str(ingest_path).startswith("hypecut/"):
            continue
        playlist = f"{HLS_BASE}/{quote(str(ingest_path), safe='/')}/index.m3u8"
        try:
            probe = await client.get(playlist, timeout=10)
            if probe.status_code != 200 or "#EXTM3U" not in probe.text:
                continue
            if session["status"] == "connecting":
                await rest_patch(client, "stream_sessions", {"id": f"eq.{session['id']}"},
                    {"status": "live", "started_at": session.get("started_at") or utcnow(),
                     "last_heartbeat_at": utcnow(), "updated_at": utcnow()})
            else:
                await rest_patch(client, "stream_sessions", {"id": f"eq.{session['id']}"},
                    {"last_heartbeat_at": utcnow(), "updated_at": utcnow()})

            # Re-check rights before each analysis window; revoked consent stops processing.
            sources = await rest_get(client, "sources", {
                "select": "id,user_id,status,consent_id,title",
                "id": f"eq.{session['source_id']}",
                "user_id": f"eq.{session['user_id']}", "limit": "1",
            })
            if not sources or sources[0].get("status") not in ("authorised", "monitoring", "pending"):
                await rest_patch(client, "stream_sessions", {"id": f"eq.{session['id']}"},
                    {"status": "paused", "updated_at": utcnow()})
                continue
            consent_id = sources[0].get("consent_id")
            consents = await rest_get(client, "creator_consents", {
                "select": "id,rights_granted,revoked_at",
                "id": f"eq.{consent_id}" if consent_id else "eq.00000000-0000-0000-0000-000000000000",
                "user_id": f"eq.{session['user_id']}", "limit": "1",
            })
            if not consents or not consents[0].get("rights_granted") or consents[0].get("revoked_at"):
                await rest_patch(client, "stream_sessions", {"id": f"eq.{session['id']}"},
                    {"status": "paused", "updated_at": utcnow()})
                continue

            with tempfile.TemporaryDirectory(prefix="hypecut-live-") as tmp:
                work = Path(tmp)
                captured = work / "window.mp4"
                await asyncio.to_thread(ffmpeg_capture, playlist, captured)
                if not captured.exists() or captured.stat().st_size < 12000:
                    continue
                candidate = await rank_window(captured, 0.0)
                processed += 1
                if candidate is None or candidate.score < MIN_SCORE:
                    await rest_post(client, "analysis_events", {
                        "user_id": session["user_id"], "stream_session_id": session["id"],
                        "source_id": session["source_id"], "event_type": "system",
                        "occurred_at": utcnow(), "score": 0,
                        "payload": {"kind": "window_rejected", "reason": "No semantic highlight passed quality threshold",
                                    "worker_id": worker_id},
                    })
                    continue

                fingerprint = hashlib.sha256(candidate.transcript.lower().encode("utf-8")).hexdigest()[:24]
                latest = await rest_get(client, "stream_sessions", {
                    "select": "ingest_metadata", "id": f"eq.{session['id']}", "limit": "1",
                })
                latest_meta = (latest[0].get("ingest_metadata") or {}) if latest else metadata
                if latest_meta.get("last_candidate_fingerprint") == fingerprint:
                    continue

                output = work / "highlight.mp4"
                srt_path = work / "captions.srt"
                has_captions = await transcribe_srt(captured, srt_path)
                await asyncio.to_thread(render_vertical, captured, output, srt_path if has_captions else None)
                if not output.exists() or output.stat().st_size < 12000:
                    raise RuntimeError("Rendered live highlight is empty")
                storage_path = f"{session['user_id']}/live/{session['id']}/{fingerprint}.mp4"
                await upload_file(client, "hypecut-clips", storage_path, output, "video/mp4")
                reasons = "; ".join(candidate.reasons[:3])
                await rest_post(client, "clips", {
                    "user_id": session["user_id"], "source_id": session["source_id"],
                    "title": candidate.title or candidate.hook or "HYPECUT live highlight",
                    "description": f"{candidate.caption}\n\n{reasons}".strip()[:1500],
                    "storage_path": storage_path,
                    "duration_seconds": round(candidate.end_seconds - candidate.start_seconds, 2),
                    "aspect_ratio": "9:16", "captions_enabled": has_captions,
                    "hype_score": candidate.score,
                    "detection_reason": f"Semantic live highlight: {candidate.category}. {reasons}"[:1000],
                    "status": "ready",
                })
                await rest_post(client, "analysis_events", {
                    "user_id": session["user_id"], "stream_session_id": session["id"],
                    "source_id": session["source_id"], "event_type": "moment_candidate",
                    "occurred_at": utcnow(), "offset_seconds": None, "score": candidate.score,
                    "payload": {"title": candidate.title, "hook": candidate.hook,
                                "category": candidate.category, "reasons": candidate.reasons,
                                "transcript": candidate.transcript[:4000],
                                "fingerprint": fingerprint, "worker_id": worker_id},
                })
                latest_meta["last_candidate_fingerprint"] = fingerprint
                latest_meta["last_candidate_at"] = utcnow()
                await rest_patch(client, "stream_sessions", {"id": f"eq.{session['id']}"},
                    {"ingest_metadata": latest_meta, "last_heartbeat_at": utcnow(),
                     "updated_at": utcnow()})
                log.info("Created semantic live clip session=%s score=%s", session["id"], candidate.score)
        except Exception:
            log.exception("Live processing failed for session %s", session.get("id"))
    return processed
