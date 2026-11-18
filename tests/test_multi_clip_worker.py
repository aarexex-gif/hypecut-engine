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
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b""),
    )

    clips = worker.choose_highlights(Path("silent.mp4"), 90, 30, max_clips=3)

    assert len(clips) == 1
    assert clips[0] == (30.0, 60.0, 0.0)
