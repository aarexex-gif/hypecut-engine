"""Regression tests for multi-clip selection in the authorised-upload worker."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import worker


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
