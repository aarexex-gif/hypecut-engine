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

from live_pipeline import process_live_sessions
from live_intelligence import detect_facecam_region, rank_window

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
    # httpx.AsyncClient requires async-compatible request content; a normal
    # file object is a synchronous stream and raises at send time. Read the
    # rendered MP4 off the event loop, then send its bytes asynchronously.
    payload = await asyncio.to_thread(file_path.read_bytes)
    response = await client.post(
        url,
        headers={**headers(), "Content-Type": content_type, "x-upsert": "true"},
        content=payload,
        timeout=900,
    )
    response.raise_for_status()


async def cleanup_failed_job_outputs(client: httpx.AsyncClient, job_id: str, user_id: str, paths: list[str]) -> None:
    """Remove this failed attempt's visible rows and uploaded objects."""
    # Hide rows first so a storage-delete failure cannot leave broken,
    # downloadable-looking clips in the user's library.
    response = await client.delete(
        f"{SUPABASE_URL}/rest/v1/clips",
        params={"job_id": f"eq.{job_id}", "user_id": f"eq.{user_id}"},
        headers=headers(),
    )
    response.raise_for_status()
    if paths:
        response = await client.request(
            "POST", f"{SUPABASE_URL}/storage/v1/object/{OUTPUT_BUCKET}",
            headers=headers(), json={"prefixes": paths},
        )
        response.raise_for_status()


async def set_progress(client: httpx.AsyncClient, job_id: str, status: str, progress: int) -> None:
    await rest_patch(
        client, "clip_jobs", {"id": f"eq.{job_id}"},
        {"status": status, "progress_percent": progress,
         "worker_locked_at": datetime.now(timezone.utc).isoformat(),
         "worker_id": WORKER_ID, "updated_at": datetime.now(timezone.utc).isoformat()},
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


def motion_zoom_expression(path: Path, duration: float) -> str:
    """Derive restrained zoom pulses from actual frame-to-frame visual activity."""
    if duration < 2:
        return "1+0.08*(0.5-0.5*cos(2*PI*on/240))"
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf",
         "fps=2,scale=160:90:flags=area,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=180,
    )
    frame_size = 160 * 90
    if proc.returncode or len(proc.stdout) < frame_size * 4:
        return "1+0.08*(0.5-0.5*cos(2*PI*on/240))"
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    frames = frames[:(len(frames) // frame_size) * frame_size].reshape(-1, 90, 160)
    activity = np.mean(np.abs(np.diff(frames.astype(np.float32), axis=0)), axis=(1, 2))
    if len(activity) < 3 or float(np.max(activity)) < 1.5:
        return "1"
    smooth = np.convolve(activity, np.ones(3) / 3, mode="same")
    threshold = max(float(np.percentile(smooth, 70)), float(np.max(smooth)) * 0.38)
    peaks = sorted((i for i in range(1, len(smooth)-1)
        if smooth[i] >= threshold and smooth[i] >= smooth[i-1] and smooth[i] >= smooth[i+1]),
        key=lambda i: smooth[i], reverse=True)
    selected = []
    for peak in peaks:
        if all(abs(peak-other) >= 5 for other in selected):
            selected.append(peak)
        if len(selected) == 4:
            break
    if not selected:
        return "1"
    maximum = max(float(np.max(smooth)), 1e-6)
    pulses = [f"{min(0.14, 0.06 + 0.08*float(smooth[p]/maximum)):.4f}*max(0,1-abs(on-{int(round((p+0.5)*15))})/45)" for p in selected]
    return "min(1.16,1+" + "+".join(pulses) + ")"


def build_vertical_filter(
    captions_path: str | None = None,
    facecam_box: tuple[float, float, float, float] | None = None,
    zoom_expression: str | None = None,
) -> str:
    """Compose a portrait gaming short and optionally lift the detected facecam into a clear inset."""
    split = "[0:v]split=3[bgsrc][fgsrc][camsrc];" if facecam_box else "[0:v]split=2[bgsrc][fgsrc];"
    if facecam_box:
        x, _, width, _ = facecam_box
        right_edge = x + width
        if x < 0.4 - 1e-9:
            crop_x = f"iw*{right_edge:.4f}+(iw*(1-{right_edge:.4f})-ih*9/16)/2"
        elif right_edge > 0.6 + 1e-9:
            crop_x = f"(iw*{x:.4f}-ih*9/16)/2"
        else:
            # A centrally placed webcam must not remain baked into the gameplay
            # crop as well as appearing in the separate inset. Choose the side
            # with more horizontal room and keep the crop outside its box.
            left_room = x
            right_room = 1.0 - right_edge
            if right_room >= left_room:
                crop_x = f"iw*{right_edge:.4f}"
            else:
                crop_x = f"iw*{x:.4f}-ih*9/16"
    else:
        crop_x = "(iw-ih*9/16)/2"
    zoom_expression = zoom_expression or "1+0.08*(0.5-0.5*cos(2*PI*on/240))"
    graph = (
        split
        + "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:10[bg];"
        f"[fgsrc]crop=w=ih*9/16:h=ih:x={crop_x}:y=0,"
        "scale=1080:1920:flags=lanczos,"
        f"zoompan=z='{zoom_expression}':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        "d=1:s=1080x1920:fps=30[fg];"
        "[bg][fg]overlay=0:0,setsar=1[base]"
    )
    if facecam_box:
        x, y, width, height = facecam_box
        graph += (
            f";[camsrc]crop=w=iw*{width:.4f}:h=ih*{height:.4f}:"
            f"x=iw*{x:.4f}:y=ih*{y:.4f},"
            "scale=420:-2:flags=lanczos,"
            "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=3,setsar=1[cam];"
            "[base][cam]overlay=x=W-w-36:y=72:eval=init:shortest=1"
        )
    else:
        graph += ";[base]null"
    if captions_path:
        escaped = captions_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        graph += (
            f",subtitles='{escaped}':force_style="
            "'FontName=Arial,FontSize=48,Bold=1,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00101010,BorderStyle=1,Outline=4,Shadow=1,"
            "Alignment=2,MarginV=210'"
        )
    return graph + "[vout]"


def validate_rendered_clip(path: Path, max_av_offset: float = 0.15, max_stream_duration_delta: float = 0.75) -> dict[str, Any]:
    """Reject malformed renders and measurable audio/video timestamp drift."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode:
        raise WorkerError("Could not inspect rendered clip")
    try:
        metadata = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise WorkerError("Rendered clip metadata is invalid") from exc
    streams = metadata.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video:
        raise WorkerError("Rendered clip has no video stream")
    if int(video.get("width") or 0) != 1080 or int(video.get("height") or 0) != 1920:
        raise WorkerError("Rendered clip is not 1080x1920 vertical output")
    if audio:
        video_start = float(video.get("start_time") or 0)
        audio_start = float(audio.get("start_time") or 0)
        if abs(video_start - audio_start) > max_av_offset:
            raise WorkerError("Rendered clip failed audio/video synchronization check")
        video_duration = float(video.get("duration") or 0)
        audio_duration = float(audio.get("duration") or 0)
        if video_duration > 0 and audio_duration > 0 and abs(video_duration - audio_duration) > max_stream_duration_delta:
            raise WorkerError("Rendered clip audio/video durations are misaligned")
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "has_audio": audio is not None,
        "av_start_offset": abs(float(video.get("start_time") or 0) - float(audio.get("start_time") or 0)) if audio else None,
    }


def choose_highlights(path: Path, duration: float, clip_length: int, max_clips: int = 3) -> list[tuple[float, float, float]]:
    """Return high-energy, non-overlapping candidate windows from an authorised upload."""
    # Decode in bounded five-minute chunks. A six-hour recording therefore
    # never requires hundreds of megabytes of decoded PCM in one allocation.
    sample_rate = 8000
    chunk_seconds = 300
    rms_parts: list[np.ndarray] = []
    for offset in range(0, max(1, int(np.ceil(duration))), chunk_seconds):
        chunk_duration = min(chunk_seconds, duration - offset)
        if chunk_duration <= 0:
            break
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{offset:.3f}", "-i", str(path),
             "-t", f"{chunk_duration:.3f}", "-vn", "-ac", "1", "-ar", str(sample_rate),
             "-f", "f32le", "-"],
            capture_output=True, timeout=900,
        )
        if proc.returncode:
            raise WorkerError(f"Audio analysis failed in source segment at {offset}s")
        if not proc.stdout:
            continue
        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        count = len(samples) // sample_rate
        if count:
            complete = samples[:count * sample_rate].reshape(count, sample_rate)
            chunk_rms = np.sqrt(np.mean(np.square(complete, dtype=np.float64), axis=1))
            rms_parts.append(chunk_rms.astype(np.float32))
    if not rms_parts:
        start = max(0.0, (duration - clip_length) / 2)
        return [(start, min(duration, start + clip_length), 0.0)]
    rms = np.concatenate(rms_parts)
    width = max(1, min(clip_length, len(rms)))
    scores = np.convolve(rms, np.ones(width) / width, mode="valid")
    baseline = float(np.percentile(rms, 75)) if len(rms) else 0.0
    ranked = np.argsort(scores)[::-1]
    chosen: list[tuple[float, float, float]] = []
    # Keep candidates distinct without spacing them so far apart that a
    # typical three-minute upload cannot yield a three-clip pack.
    min_gap = max(10, clip_length * 0.2)
    for candidate in ranked:
        start = min(float(candidate), max(0.0, duration - clip_length))
        end = min(duration, start + clip_length)
        if any(not (end + min_gap <= a or start >= b + min_gap) for a, b, _ in chosen):
            continue
        peak = float(scores[candidate])
        score = 0.0 if baseline <= 1e-8 else min(100.0, max(0.0, 50.0 + 25.0 * (peak / baseline - 1.0)))
        chosen.append((start, end, round(score, 1)))
        if len(chosen) >= max_clips:
            break
    if not chosen:
        start = max(0.0, (duration - clip_length) / 2)
        chosen.append((start, min(duration, start + clip_length), 0.0))
    return sorted(chosen, key=lambda item: item[0])


def choose_highlight(path: Path, duration: float, clip_length: int) -> tuple[float, float, float]:
    """Backward-compatible single-best helper."""
    return max(choose_highlights(path, duration, clip_length, 1), key=lambda item: item[2])


def semantic_window_offsets(duration: float, window_seconds: int = 60, step_seconds: int = 45) -> list[int]:
    """Return overlapping windows covering the complete source timeline."""
    if duration < 20 or window_seconds < 1 or step_seconds < 1:
        return []
    last_start = max(0, int(duration) - window_seconds)
    offsets = list(range(0, last_start + 1, step_seconds))
    if not offsets or offsets[-1] != last_start:
        offsets.append(last_start)
    return offsets


async def choose_semantic_highlight(media_path: Path, duration: float, work: Path, on_progress=None):
    """Review every overlapping source window and return the strongest semantic moment."""
    if not OPENAI_API_KEY or duration < 30:
        return None
    offsets = semantic_window_offsets(duration)
    # Do not sample only the beginning/middle/end of long recordings.
    # Every 45-second start position is reviewed, including a final window
    # aligned to the end so the entire source timeline is covered.
    best = None
    for index, offset in enumerate(offsets):
        excerpt = work / f"semantic-window-{index}.mp4"
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{offset:.3f}", "-i", str(media_path), "-t", "60",
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
            "-c:a", "aac", "-b:a", "96k", str(excerpt),
        ]
        try:
            await asyncio.to_thread(run_ffmpeg, args)
            candidate = await rank_window(excerpt, float(offset))
            if candidate and (best is None or candidate.score > best.score):
                best = candidate
        finally:
            excerpt.unlink(missing_ok=True)
        if on_progress and ((index + 1) % 5 == 0 or index + 1 == len(offsets)):
            await on_progress(index + 1, len(offsets))
    return best


def validate_srt_captions(content: str, clip_duration: float) -> bool:
    """Validate numbered SRT cues, timestamp fields, ordering, and clip bounds."""
    timestamp = re.compile(r"^(\d{2,}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2,}):(\d{2}):(\d{2}),(\d{3})(?:\s+.*)?$")

    def seconds(hours: str, minutes: str, whole: str, millis: str) -> float | None:
        if int(minutes) > 59 or int(whole) > 59:
            return None
        return int(hours) * 3600 + int(minutes) * 60 + int(whole) + int(millis) / 1000

    blocks = [block.strip() for block in re.split(r"\n\s*\n", content.strip()) if block.strip()]
    if not blocks or clip_duration <= 0:
        return False
    previous_start = -1.0
    seen_indices: set[int] = set()
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].strip().isdigit():
            return False
        cue_index = int(lines[0].strip())
        if cue_index < 1 or cue_index in seen_indices:
            return False
        seen_indices.add(cue_index)
        match = timestamp.match(lines[1].strip())
        if not match:
            return False
        values = match.groups()
        start = seconds(*values[:4])
        end = seconds(*values[4:])
        if start is None or end is None:
            return False
        if start < 0 or end <= start or end > clip_duration + 0.25 or start < previous_start:
            return False
        if not any(line.strip() for line in lines[2:]):
            return False
        previous_start = start
    return True


def _ass_timestamp(seconds: float) -> str:
    value = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(value, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{centiseconds:02d}"


def word_timestamps_to_ass(payload: dict[str, Any], clip_duration: float) -> str | None:
    """Create short karaoke-style caption phrases from Whisper word timestamps."""
    raw_words = payload.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        return None
    words: list[tuple[float, float, str]] = []
    for item in raw_words:
        if not isinstance(item, dict):
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
            word = str(item.get("word") or "").strip()
        except (KeyError, TypeError, ValueError):
            continue
        word = re.sub(r"[{}\\\\]", "", word)
        if not word or start < 0 or end <= start or end > clip_duration + 0.25:
            continue
        words.append((start, end, word))
    if not words:
        return None
    words.sort(key=lambda item: item[0])
    groups: list[list[tuple[float, float, str]]] = []
    for word in words:
        if (
            not groups
            or len(groups[-1]) >= 5
            or word[0] - groups[-1][-1][1] > 0.65
            or word[0] - groups[-1][0][0] > 2.8
        ):
            groups.append([word])
        else:
            groups[-1].append(word)
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,62,&H00FFFFFF,&H0000D7FF,&H00101010,&H80000000,-1,0,1,5,2,2,80,80,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for group in groups:
        start, end = group[0][0], group[-1][1]
        parts = []
        for word_start, word_end, word in group:
            duration_cs = max(1, int(round((word_end - word_start) * 100)))
            parts.append(f"{{\\k{duration_cs}}}{word}")
        events.append(
            f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},"
            f"Default,,0,0,0,,{' '.join(parts)}"
        )
    return header + "\n".join(events) + "\n"


async def transcribe(
    client: httpx.AsyncClient,
    media_path: Path,
    work: Path,
    start_seconds: float = 0,
    duration_seconds: float | None = None,
) -> Path | None:
    if not OPENAI_API_KEY:
        return None
    suffix = str(int(start_seconds * 1000))
    audio = work / f"audio-{suffix}.mp3"
    args = ["ffmpeg", "-y", "-ss", f"{start_seconds:.3f}", "-i", str(media_path)]
    if duration_seconds is not None:
        args += ["-t", f"{duration_seconds:.3f}"]
    args += ["-vn", "-ac", "1", "-b:a", "64k", str(audio)]
    await asyncio.to_thread(run_ffmpeg, args)
    with audio.open("rb") as stream:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            data={
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
            files={"file": (audio.name, stream, "audio/mpeg")},
            timeout=600,
        )
    if response.is_error:
        log.warning("Transcription unavailable: %s", response.status_code)
        return None
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    ass_content = word_timestamps_to_ass(payload, duration_seconds or 0)
    if ass_content:
        captions = work / f"captions-{suffix}.ass"
        captions.write_text(ass_content, encoding="utf-8")
        return captions
    # Keep a conservative segment-caption fallback if word timestamps are absent.
    with audio.open("rb") as fallback_stream:
        fallback = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            data={"model": "whisper-1", "response_format": "srt"},
            files={"file": (audio.name, fallback_stream, "audio/mpeg")},
            timeout=600,
        )
    if fallback.is_error or not validate_srt_captions(fallback.text, duration_seconds or 0):
        log.warning("Transcription returned no valid word or segment captions")
        return None
    srt = work / f"captions-{suffix}.srt"
    srt.write_text(fallback.text, encoding="utf-8")
    return srt

async def process_job(client: httpx.AsyncClient, job: dict[str, Any]) -> None:
    job_id = job["id"]
    user_id = job["user_id"]
    uploaded_paths: list[str] = []
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

            async def report_semantic_progress(done: int, total: int) -> None:
                progress = 25 + int(25 * done / max(1, total))
                await set_progress(client, job_id, "analysing", progress)

            semantic = await choose_semantic_highlight(
                media, duration, work, on_progress=report_semantic_progress
            )
            if semantic:
                candidates = [(semantic.start_seconds, semantic.end_seconds, semantic.score)]
                semantic_title = semantic.title
                semantic_description = (semantic.caption + "\n\n" + "; ".join(semantic.reasons)).strip()[:1500]
                semantic_reason = f"Semantic AI highlight: {semantic.category}. " + "; ".join(semantic.reasons)
                fallback = await asyncio.to_thread(choose_highlights, media, duration, clip_length, 3)
                for candidate in fallback:
                    if all(candidate[1] <= existing[0] or candidate[0] >= existing[1] for existing in candidates):
                        candidates.append(candidate)
                    if len(candidates) >= 3:
                        break
            else:
                candidates = await asyncio.to_thread(choose_highlights, media, duration, clip_length, 3)
                semantic_title = None
                semantic_description = "Automatically selected using sustained audio-energy scoring."
                semantic_reason = "High sustained audio-energy window (fallback)"

            # Detect the webcam panel once per source. If vision cannot locate it,
            # retain the safe full-frame portrait composition without an inset.
            try:
                facecam_box = await detect_facecam_region(media)
            except Exception:
                log.exception("Facecam detection failed; using full-frame fallback composition")
                facecam_box = None
            log.info("Facecam composition detected: %s", bool(facecam_box))

            completed_clips = []
            await set_progress(client, job_id, "rendering", 55)
            for index, (start, end, score) in enumerate(candidates, start=1):
                captions = await transcribe(client, media, work, start, end - start)
                clip_source = work / f"motion-source-{index}.mp4"
                await asyncio.to_thread(run_ffmpeg, [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{start:.3f}", "-i", str(media), "-t", f"{end-start:.3f}",
                    "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                    "-threads", "2", str(clip_source),
                ])
                zoom_expression = await asyncio.to_thread(motion_zoom_expression, clip_source, end-start)
                clip_source.unlink(missing_ok=True)
                output = work / f"clip-{index}.mp4"
                args = [
                    "ffmpeg", "-y",
                    "-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1",
                    "-ss", f"{start:.3f}", "-i", str(media),
                    "-t", f"{end-start:.3f}", "-filter_complex", build_vertical_filter(str(captions) if captions else None, facecam_box, zoom_expression),
                    "-map", "[vout]", "-map", "0:a?",
                ]
                args += ["-c:v", "libx264", "-threads", "2", "-x264-params", "threads=2",
                         "-preset", "veryfast", "-crf", "23",
                         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output)]
                await asyncio.to_thread(run_ffmpeg, args)
                if not output.exists() or output.stat().st_size < 1024:
                    raise WorkerError(f"FFmpeg did not produce clip {index}")
                qc = await asyncio.to_thread(validate_rendered_clip, output)
                log.info("Clip %s passed render QC: %s", index, qc)
                await set_progress(client, job_id, "uploading", 70 + int(20 * index / len(candidates)))
                output_path = f"{user_id}/{job_id}/{index}.mp4"
                await upload_file(client, OUTPUT_BUCKET, output_path, output, "video/mp4")
                uploaded_paths.append(output_path)
                clip = {
                    "user_id": user_id, "source_id": source["id"], "job_id": job_id,
                    "title": (semantic_title if index == 1 and semantic_title else f"{(source.get('title') or 'HYPECUT highlight')[:150]} — Clip {index}"),
                    "description": semantic_description if index == 1 and semantic else f"Automatically selected highlight {index} from this source.",
                    "storage_path": output_path, "duration_seconds": round(end-start, 2),
                    "aspect_ratio": "9:16", "captions_enabled": bool(captions),
                    "hype_score": score, "detection_reason": (semantic_reason if index == 1 and semantic else f"{semantic_reason}; candidate {index}")[:1000],
                    "status": "ready",
                }
                created = await rest_post(client, "clips", clip)
                completed_clips.extend(created if isinstance(created, list) else [created])

            first = candidates[0]
            await rest_patch(client, "clip_jobs", {"id": f"eq.{job_id}"}, {
                "status": "completed", "progress_percent": 100,
                "start_seconds": first[0], "end_seconds": first[1],
                "hype_score": max(item[2] for item in candidates),
                "detection_reason": f"Generated {len(completed_clips)} vertical clips: {semantic_reason}"[:1000],
                "completed_at": datetime.now(timezone.utc).isoformat(), "worker_locked_at": None, "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            log.info("Completed clip job %s with %s clips", job_id, len(completed_clips))
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        try:
            await cleanup_failed_job_outputs(client, job_id, user_id, uploaded_paths)
        except Exception:
            log.exception("Could not clean partial outputs for failed job %s", job_id)
        try:
            await rest_patch(client, "clip_jobs", {"id": f"eq.{job_id}"}, {
                "status": "failed", "error_code": "worker_processing_failed",
                "error_message": str(exc)[:1000], "worker_locked_at": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            log.exception("Could not record failure for job %s", job_id)


async def live_session_loop(client: httpx.AsyncClient) -> None:
    while True:
        try:
            if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
                await process_live_sessions(client, rest_get, rest_patch, rest_post, upload_file, WORKER_ID)
        except Exception:
            log.exception("Live-session monitor cycle failed")
        await asyncio.sleep(max(3, POLL_SECONDS))


async def main() -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        log.info("HYPECUT media worker process started; id=%s", WORKER_ID)
        live_task = None
        while True:
            if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
                log.error("Worker is idle: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured in Railway")
                await asyncio.sleep(60)
                continue
            if live_task is None:
                live_task = asyncio.create_task(live_session_loop(client))
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
