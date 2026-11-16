from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("hypecut-worker")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "5"))
WORKER_ID = os.getenv("RAILWAY_REPLICA_ID", str(uuid.uuid4()))
INPUT_BUCKET = "hypecut-inputs"
OUTPUT_BUCKET = "hypecut-clips"


class WorkerError(Exception):
    pass


def headers() -> dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise WorkerError("Supabase server credentials are not configured")
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


async def rest_get(client: httpx.AsyncClient, table: str, params: dict[str, str]) -> Any:
    response = await client.get(f"{SUPABASE_URL}/rest/v1/{table}", params=params, headers=headers())
    response.raise_for_status()
    return response.json()


async def rest_patch(client: httpx.AsyncClient, table: str, params: dict[str, str], payload: dict[str, Any]) -> None:
    response = await client.patch(
        f"{SUPABASE_URL}/rest/v1/{table}", params=params,
        headers={**headers(), "Prefer": "return=minimal"}, json=payload,
    )
    response.raise_for_status()


async def rest_post(client: httpx.AsyncClient, table: str, payload: dict[str, Any]) -> Any:
    response = await client.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**headers(), "Prefer": "return=representation"}, json=payload,
    )
    response.raise_for_status()
    return response.json()


async def claim_job(client: httpx.AsyncClient) -> dict[str, Any] | None:
    response = await client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/claim_hypecut_clip_job",
        headers=headers(),
        json={"p_worker_id": WORKER_ID},
    )
    response.raise_for_status()
    rows = response.json()
    return rows[0] if rows else None


async def signed_download(client: httpx.AsyncClient, path: str) -> str:
    response = await client.post(
        f"{SUPABASE_URL}/storage/v1/object/sign/{INPUT_BUCKET}/{path}",
        headers=headers(),
        json={"expiresIn": 3600},
    )
    response.raise_for_status()
    signed = response.json().get("signedURL")
    if not signed:
        raise WorkerError("Storage did not return a signed input URL")
    if signed.startswith("/"):
        return f"{SUPABASE_URL}/storage/v1{signed}"
    return signed


async def upload_file(client: httpx.AsyncClient, bucket: str, path: str, file_path: Path, content_type: str) -> None:
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    with file_path.open("rb") as stream:
        response = await client.post(
            url,
            headers={**headers(), "Content-Type": content_type, "x-upsert": "true"},
            content=stream,
        )
    response.raise_for_status()


async def set_progress(client: httpx.AsyncClient, job_id: str, status: str, progress: int) -> None:
    await rest_patch(
        client, "clip_jobs", {"id": f"eq.{job_id}"},
        {"status": status, "progress_percent": progress, "worker_locked_at": "now()",
         "worker_id": WORKER_ID, "updated_at": "now()"},
    )


def run_ffmpeg(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if proc.returncode:
        detail = (proc.stderr or "FFmpeg produced no diagnostic output")[-3000:]
        log.error("FFmpeg exited with return code %s; command=%s; stderr_tail=%s",
                  proc.returncode, " ".join(args), detail)
        raise WorkerError(f"FFmpeg exited with return code {proc.returncode}: {detail}")
    return proc.stderr


def media_duration(path: Path) -> float:
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True, timeout=30)
    return max(0.0, float(output.strip()))


def choose_highlight(path: Path, duration: float, clip_length: int) -> tuple[float, float, float]:
    # Decode mono 8 kHz audio and score one-second RMS windows. If the source
    # has no audio, use a safe middle segment rather than inventing a score.
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
        capture_output=True, timeout=900,
    )
    if proc.returncode or not proc.stdout:
        start = max(0.0, (duration - clip_length) / 2)
        return start, min(duration, start + clip_length), 0.0

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    sample_rate = 8000
    window = sample_rate
    count = len(samples) // window
    if count < 1:
        start = max(0.0, (duration - clip_length) / 2)
        return start, min(duration, start + clip_length), 0.0
    rms = np.array([
        float(np.sqrt(np.mean(np.square(samples[i*window:(i+1)*window], dtype=np.float64))))
        for i in range(count)
    ])
    span = min(clip_length, max(1, int(duration)))
    width = max(1, min(span, len(rms)))
    # A moving average favours sustained energetic sections over one-sample spikes.
    scores = np.convolve(rms, np.ones(width) / width, mode="valid")
    best = int(np.argmax(scores))
    start = min(float(best), max(0.0, duration - clip_length))
    end = min(duration, start + clip_length)
    baseline = float(np.percentile(rms, 75)) if len(rms) else 0.0
    peak = float(np.max(scores)) if len(scores) else 0.0
    score = 0.0 if baseline <= 1e-8 else min(100.0, max(0.0, 50.0 + 25.0 * (peak / baseline - 1.0)))
    return start, end, round(score, 1)


async def transcribe(client: httpx.AsyncClient, media_path: Path, work: Path) -> Path | None:
    if not OPENAI_API_KEY:
        return None
    audio = work / "audio.mp3"
    run_ffmpeg(["ffmpeg", "-y", "-ss", "0", "-i", str(media_path), "-vn", "-ac", "1", "-b:a", "64k", str(audio)])
    with audio.open("rb") as stream:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            data={"model": "whisper-1", "response_format": "srt"},
            files={"file": ("audio.mp3", stream, "audio/mpeg")},
            timeout=600,
        )
    if response.is_error:
        log.warning("Transcription unavailable: %s", response.status_code)
        return None
    srt = work / "captions.srt"
    srt.write_text(response.text, encoding="utf-8")
    return srt


async def process_job(client: httpx.AsyncClient, job: dict[str, Any]) -> None:
    job_id = job["id"]
    user_id = job["user_id"]
    try:
        sources = await rest_get(client, "sources", {
            "select": "id,platform,source_url,metadata,title",
            "id": f"eq.{job['source_id']}", "user_id": f"eq.{user_id}", "limit": "1",
        })
        if not sources:
            raise WorkerError("Creator source was not found")
        source = sources[0]
        if source["platform"] != "upload":
            raise WorkerError("This worker currently accepts creator-uploaded media only; YouTube ingestion is not enabled")

        metadata = source.get("metadata") or {}
        storage_path = metadata.get("input_storage_path")
        if not storage_path or not storage_path.startswith(f"{user_id}/"):
            raise WorkerError("Creator input storage path is missing or invalid")

        with tempfile.TemporaryDirectory(prefix="hypecut-") as tmp:
            work = Path(tmp)
            media = work / "source-media"
            await set_progress(client, job_id, "downloading", 5)
            signed = await signed_download(client, storage_path)
            async with client.stream("GET", signed, timeout=900) as response:
                response.raise_for_status()
                with media.open("wb") as out:
                    async for chunk in response.aiter_bytes():
                        out.write(chunk)
            if not media.exists() or media.stat().st_size < 1024:
                raise WorkerError("Uploaded source file is empty or invalid")

            duration = media_duration(media)
            if duration < 30:
                raise WorkerError("Source video must be at least 30 seconds")
            clip_length = min(60, max(30, int(job.get("target_duration_seconds") or 45)))
            await set_progress(client, job_id, "analysing", 25)
            start, end, score = await asyncio.to_thread(choose_highlight, media, duration, clip_length)

            await set_progress(client, job_id, "rendering", 55)
            captions = await transcribe(client, media, work)
            output = work / "clip.mp4"
            args = [
                "ffmpeg", "-y",
                "-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1",
                "-ss", f"{start:.3f}", "-i", str(media),
                "-t", f"{end-start:.3f}", "-map", "0:v:0", "-map", "0:a?",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            ]
            if captions:
                escaped = str(captions).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
                args[-1] += f",subtitles='{escaped}'"
            args += ["-c:v", "libx264", "-threads", "2", "-x264-params", "threads=2",
                     "-preset", "veryfast", "-crf", "23",
                     "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output)]
            await asyncio.to_thread(run_ffmpeg, args)
            if not output.exists() or output.stat().st_size < 1024:
                raise WorkerError("FFmpeg did not produce a valid clip")

            await set_progress(client, job_id, "uploading", 85)
            output_path = f"{user_id}/{job_id}.mp4"
            await upload_file(client, OUTPUT_BUCKET, output_path, output, "video/mp4")
            clip = {
                "user_id": user_id, "source_id": source["id"], "job_id": job_id,
                "title": f"{(source.get('title') or 'HYPECUT highlight')[:180]} — highlight",
                "description": "Automatically selected using an audio-energy heuristic. Not a semantic AI highlight score.",
                "storage_path": output_path, "duration_seconds": round(end-start, 2),
                "aspect_ratio": "9:16", "captions_enabled": bool(captions),
                "hype_score": score, "detection_reason": "Highest sustained audio-energy window",
                "status": "ready",
            }
            await rest_post(client, "clips", clip)
            await rest_patch(client, "clip_jobs", {"id": f"eq.{job_id}"}, {
                "status": "completed", "progress_percent": 100,
                "start_seconds": start, "end_seconds": end,
                "hype_score": score, "detection_reason": "Highest sustained audio-energy window",
                "completed_at": datetime.now(timezone.utc).isoformat(), "worker_locked_at": None, "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            log.info("Completed clip job %s", job_id)
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        try:
            await rest_patch(client, "clip_jobs", {"id": f"eq.{job_id}"}, {
                "status": "failed", "error_code": "worker_processing_failed",
                "error_message": str(exc)[:1000], "worker_locked_at": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            log.exception("Could not record failure for job %s", job_id)


async def main() -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        log.info("HYPECUT media worker process started; id=%s", WORKER_ID)
        while True:
            if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
                log.error("Worker is idle: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured in Railway")
                await asyncio.sleep(60)
                continue
            try:
                job = await claim_job(client)
                if job:
                    await process_job(client, job)
                else:
                    await asyncio.sleep(POLL_SECONDS)
            except Exception:
                log.exception("Worker polling error")
                await asyncio.sleep(max(POLL_SECONDS, 10))


if __name__ == "__main__":
    asyncio.run(main())
