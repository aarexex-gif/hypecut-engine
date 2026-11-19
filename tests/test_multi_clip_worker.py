"""Regression tests for multi-clip selection in the authorised-upload worker."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import worker
import live_intelligence


def test_choose_highlights_returns_three_distinct_windows(monkeypatch):
    sample_rate = 8000
    samples = np.full(180 * sample_rate, 0.05, dtype=np.float32)
    for start in (0, 65, 130):
        samples[start * sample_rate : (start + 45) * sample_rate] = 0.8

    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=samples.tobytes(),
        ),
    )

    clips = worker.choose_highlights(Path("synthetic.mp4"), 180, 45, max_clips=3)

    assert len(clips) == 3
    assert all(0 <= start < end <= 180 for start, end, _ in clips)
    assert all(44 <= end - start <= 45 for start, end, _ in clips)
    assert all(clips[i + 1][0] >= clips[i][1] + 9 for i in range(2))


def test_choose_highlights_uses_safe_fallback_without_audio(monkeypatch):
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b""),
    )

    clips = worker.choose_highlights(Path("silent.mp4"), 90, 30, max_clips=3)

    assert len(clips) == 1
    assert clips[0] == (30.0, 60.0, 0.0)


def test_semantic_windows_cover_entire_six_hour_recording():
    duration = 6 * 60 * 60
    offsets = worker.semantic_window_offsets(duration)

    assert len(offsets) > 12
    assert offsets[0] == 0
    assert offsets[-1] == duration - 60
    assert all(b - a <= 45 for a, b in zip(offsets, offsets[1:]))
    assert offsets[-1] + 60 == duration


def test_audio_analysis_decodes_long_recording_in_bounded_chunks(monkeypatch):
    sample_rate = 8000
    calls = []

    def fake_ffmpeg(*args, **kwargs):
        command = args[0]
        offset = float(command[command.index("-ss") + 1])
        chunk_duration = float(command[command.index("-t") + 1])
        calls.append(offset)
        count = int(chunk_duration)
        samples = np.full(count * sample_rate, 0.05, dtype=np.float32)
        if offset <= 510 < offset + chunk_duration:
            local_start = int(510 - offset)
            local_end = min(count, local_start + 45)
            samples[local_start * sample_rate:local_end * sample_rate] = 0.8
        return SimpleNamespace(returncode=0, stdout=samples.tobytes())

    monkeypatch.setattr(worker.subprocess, "run", fake_ffmpeg)
    clips = worker.choose_highlights(Path("six-hour.mp4"), 610, 45, max_clips=3)

    assert calls == [0.0, 300.0, 600.0]
    assert any(start >= 500 for start, _, _ in clips)



def test_semantic_window_attaches_visual_frames_and_cleans_temp_files(monkeypatch, tmp_path):
    media = tmp_path / "window.mp4"

    def fake_run(command, timeout=300):
        Path(command[-1]).write_bytes(b"fake-jpeg-frame")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(live_intelligence, "_run", fake_run)
    urls = live_intelligence._keyframe_data_urls(media, 60)

    assert len(urls) == 3
    assert all(url.startswith("data:image/jpeg;base64,") for url in urls)
    assert not list(tmp_path.glob("*-frame-*.jpg"))



def test_long_semantic_review_reports_incremental_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "OPENAI_API_KEY", "test-key")

    def fake_render(args):
        Path(args[-1]).write_bytes(b"window")
        return ""

    async def fake_rank_window(media, window_start=0.0):
        return None

    monkeypatch.setattr(worker, "run_ffmpeg", fake_render)
    monkeypatch.setattr(worker, "rank_window", fake_rank_window)
    updates = []

    async def on_progress(done, total):
        updates.append((done, total))

    worker.asyncio.run(
        worker.choose_semantic_highlight(
            Path("long-recording.mp4"), 330, tmp_path, on_progress=on_progress
        )
    )

    assert updates == [(5, 7), (7, 7)]



def test_render_qc_accepts_synced_1080x1920_clip(monkeypatch, tmp_path):
    metadata = {
        "streams": [
            {"codec_type": "video", "width": 1080, "height": 1920, "start_time": "0.000", "duration": "45.000"},
            {"codec_type": "audio", "start_time": "0.020", "duration": "44.990"},
        ]
    }
    monkeypatch.setattr(
        worker.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=__import__("json").dumps(metadata)),
    )

    qc = worker.validate_rendered_clip(tmp_path / "clip.mp4")

    assert qc["width"] == 1080
    assert qc["height"] == 1920
    assert qc["has_audio"] is True
    assert qc["av_start_offset"] == 0.02


def test_render_qc_rejects_audio_video_offset(monkeypatch, tmp_path):
    metadata = {
        "streams": [
            {"codec_type": "video", "width": 1080, "height": 1920, "start_time": "0.000", "duration": "45.000"},
            {"codec_type": "audio", "start_time": "0.300", "duration": "44.990"},
        ]
    }
    monkeypatch.setattr(
        worker.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=__import__("json").dumps(metadata)),
    )

    try:
        worker.validate_rendered_clip(tmp_path / "bad-sync.mp4")
        assert False, "Expected an audio/video synchronization failure"
    except worker.WorkerError as exc:
        assert "synchronization" in str(exc)


def test_render_qc_rejects_non_vertical_output(monkeypatch, tmp_path):
    metadata = {
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "start_time": "0.000", "duration": "45.000"},
        ]
    }
    monkeypatch.setattr(
        worker.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=__import__("json").dumps(metadata)),
    )

    try:
        worker.validate_rendered_clip(tmp_path / "landscape.mp4")
        assert False, "Expected a vertical output validation failure"
    except worker.WorkerError as exc:
        assert "1080x1920" in str(exc)



def test_model_api_retries_rate_limit_then_succeeds(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self.headers = {"Retry-After": "0.5"}
            self.payload = payload or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(429)
            return FakeResponse(200, {"ok": True})

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(worker.asyncio, "sleep", no_sleep)
    client = FakeClient()
    response = worker.asyncio.run(
        live_intelligence._post_with_retries(client, "https://example.test", attempts=3)
    )

    assert response.json() == {"ok": True}
    assert client.calls == 2


def test_model_api_does_not_retry_client_errors():
    class FakeResponse:
        status_code = 400
        headers = {}

        def raise_for_status(self):
            raise RuntimeError("HTTP 400")

    class FakeClient:
        calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            return FakeResponse()

    client = FakeClient()
    try:
        worker.asyncio.run(
            live_intelligence._post_with_retries(client, "https://example.test", attempts=3)
        )
        assert False, "Expected client error to propagate"
    except RuntimeError as exc:
        assert "HTTP 400" in str(exc)

    assert client.calls == 1


def test_failed_job_cleanup_removes_uploaded_objects_and_clip_rows(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        async def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return FakeResponse()

        async def delete(self, url, **kwargs):
            calls.append(("DELETE", url, kwargs))
            return FakeResponse()

    monkeypatch.setattr(worker, "SUPABASE_URL", "https://project.example")
    monkeypatch.setattr(worker, "OUTPUT_BUCKET", "hypecut-clips")
    monkeypatch.setattr(worker, "headers", lambda: {"Authorization": "Bearer test"})

    worker.asyncio.run(
        worker.cleanup_failed_job_outputs(
            FakeClient(), "job-123", "user-456",
            ["user-456/job-123/1.mp4", "user-456/job-123/2.mp4"],
        )
    )

    assert calls[0][0] == "DELETE"
    assert calls[0][1].endswith("/rest/v1/clips")
    assert calls[0][2]["params"] == {
        "job_id": "eq.job-123", "user_id": "eq.user-456"
    }
    assert calls[1][0] == "POST"
    assert calls[1][1].endswith("/storage/v1/object/hypecut-clips")
    assert calls[1][2]["json"]["prefixes"] == [
        "user-456/job-123/1.mp4", "user-456/job-123/2.mp4"
    ]


def test_failed_job_cleanup_skips_storage_request_when_no_uploads(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        async def request(self, *args, **kwargs):
            calls.append(("request", args, kwargs))
            return FakeResponse()

        async def delete(self, *args, **kwargs):
            calls.append(("delete", args, kwargs))
            return FakeResponse()

    monkeypatch.setattr(worker, "SUPABASE_URL", "https://project.example")
    monkeypatch.setattr(worker, "headers", lambda: {})

    worker.asyncio.run(
        worker.cleanup_failed_job_outputs(FakeClient(), "job-123", "user-456", [])
    )

    assert len(calls) == 1
    assert calls[0][0] == "delete"



def test_caption_validator_accepts_in_range_srt_cues():
    srt = """1
00:00:00,200 --> 00:00:01,400
Hello there.

2
00:00:01,500 --> 00:00:03,000
Welcome to HYPECUT.
"""
    assert worker.validate_srt_captions(srt, 3.0)


def test_caption_validator_rejects_cues_beyond_clip_duration():
    srt = """1
00:00:04,000 --> 00:00:05,000
Late caption.
"""
    assert not worker.validate_srt_captions(srt, 4.5)


def test_caption_validator_rejects_malformed_or_empty_cues():
    assert not worker.validate_srt_captions("not an srt file", 30)
    assert not worker.validate_srt_captions("1\\n00:00:01,000 --> 00:00:01,000\\nEmpty duration", 30)
    assert not worker.validate_srt_captions("", 30)


def test_caption_validator_rejects_non_monotonic_cue_order():
    srt = """1
00:00:05,000 --> 00:00:06,000
Later.

2
00:00:02,000 --> 00:00:03,000
Earlier.
"""
    assert not worker.validate_srt_captions(srt, 10)


def test_caption_validator_rejects_invalid_minute_or_second_fields():
    invalid_minute = "1\\n00:61:00,000 --> 00:61:01,000\\nBad timestamp"
    invalid_second = "1\\n00:00:60,000 --> 00:01:01,000\\nBad timestamp"
    assert not worker.validate_srt_captions(invalid_minute, 90)
    assert not worker.validate_srt_captions(invalid_second, 90)


def test_caption_validator_requires_unique_numeric_cue_indices():
    duplicate = """1
00:00:00,000 --> 00:00:01,000
First.

1
00:00:01,100 --> 00:00:02,000
Duplicate index.
"""
    malformed_index = """cue
00:00:00,000 --> 00:00:01,000
No numeric index.
"""
    assert not worker.validate_srt_captions(duplicate, 3)
    assert not worker.validate_srt_captions(malformed_index, 3)



def test_vertical_filter_preserves_full_landscape_frame_and_blurs_fill():
    graph = worker.build_vertical_filter()

    assert "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase" in graph
    assert "boxblur=20:10" in graph
    assert "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease" in graph
    assert "[bg][fg]overlay=(W-w)/2:(H-h)/2" in graph
    assert graph.endswith("[vout]")


def test_vertical_filter_adds_captions_to_composited_output():
    graph = worker.build_vertical_filter("/tmp/captions.srt")

    assert "subtitles='/tmp/captions.srt'" in graph
    assert graph.endswith("[vout]")
