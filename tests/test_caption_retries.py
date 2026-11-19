"""Tests for resilient, truthful caption transcription."""
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
import worker


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _prepare_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "OPENAI_API_KEY", "test-key")

    def fake_ffmpeg(args):
        Path(args[-1]).write_bytes(b"synthetic-audio")
        return ""

    monkeypatch.setattr(worker, "run_ffmpeg", fake_ffmpeg)
    media = tmp_path / "source.mp4"
    media.write_bytes(b"synthetic-video")
    return media


def test_transcribe_retries_via_shared_retry_helper_and_writes_karaoke(monkeypatch, tmp_path):
    media = _prepare_audio(monkeypatch, tmp_path)
    calls = []

    async def fake_retry(client, url, *, attempts, **kwargs):
        calls.append((url, attempts, kwargs))
        return FakeResponse({
            "words": [
                {"start": 0.2, "end": 0.5, "word": " Hello"},
                {"start": 0.6, "end": 0.9, "word": " world"},
            ]
        })

    monkeypatch.setattr(worker, "_post_with_retries", fake_retry)
    output = worker.asyncio.run(
        worker.transcribe(object(), media, tmp_path, 0, 5)
    )

    assert output is not None
    assert output.suffix == ".ass"
    assert r"{\\k30}Hello" in output.read_text(encoding="utf-8")
    assert len(calls) == 1
    assert calls[0][1] == 3
    assert calls[0][2]["files"]["file"][1] == b"synthetic-audio"
    assert not (tmp_path / "audio-0.mp3").exists()


def test_transcribe_returns_no_fake_captions_after_persistent_429(monkeypatch, tmp_path):
    media = _prepare_audio(monkeypatch, tmp_path)
    calls = []

    async def rate_limited(client, url, *, attempts, **kwargs):
        calls.append(attempts)
        request = httpx.Request("POST", url)
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    monkeypatch.setattr(worker, "_post_with_retries", rate_limited)
    output = worker.asyncio.run(
        worker.transcribe(object(), media, tmp_path, 0, 5)
    )

    assert output is None
    assert calls == [3]
    assert not (tmp_path / "audio-0.mp3").exists()


def test_transcribe_uses_srt_fallback_when_word_timestamps_missing(monkeypatch, tmp_path):
    media = _prepare_audio(monkeypatch, tmp_path)
    calls = []

    async def no_word_timestamps(client, url, *, attempts, **kwargs):
        calls.append(kwargs["data"]["response_format"])
        if len(calls) == 1:
            return FakeResponse({"segments": []})
        return FakeResponse(text="1\n00:00:00,200 --> 00:00:01,400\nHello there.\n")

    monkeypatch.setattr(worker, "_post_with_retries", no_word_timestamps)
    output = worker.asyncio.run(
        worker.transcribe(object(), media, tmp_path, 0, 5)
    )

    assert output is not None
    assert output.suffix == ".srt"
    assert calls == ["verbose_json", "srt"]
    assert "Hello there." in output.read_text(encoding="utf-8")
    assert not (tmp_path / "audio-0.mp3").exists()
