"""Unit tests for HYPECUT's YouTube live-agent parsing and state logic."""
import pytest
from fastapi import HTTPException

from app.main import (
    chat_activity_level,
    parse_youtube_video_id,
    youtube_live_state,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abcdefghijk", "abcdefghijk"),
        ("https://youtu.be/abcdefghijk?t=30", "abcdefghijk"),
        ("https://youtube.com/live/abcdefghijk?feature=share", "abcdefghijk"),
        ("https://m.youtube.com/embed/abcdefghijk", "abcdefghijk"),
        ("https://www.youtube.com/shorts/abcdefghijk", "abcdefghijk"),
        ("youtube.com/watch?v=abcdefghijk", "abcdefghijk"),
    ],
)
def test_parse_youtube_video_id_accepts_supported_urls(url, expected):
    assert parse_youtube_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://example.com/watch?v=abcdefghijk",
        "https://youtube.com.evil.example/watch?v=abcdefghijk",
        "https://youtube.com/watch?v=too-short",
        "https://youtu.be/",
    ],
)
def test_parse_youtube_video_id_rejects_invalid_urls(url):
    with pytest.raises(HTTPException) as error:
        parse_youtube_video_id(url)
    assert error.value.status_code == 400


def test_youtube_live_state_transitions():
    assert youtube_live_state(
        {"liveBroadcastContent": "live"},
        {"actualStartTime": "2026-11-17T10:00:00Z"},
    ) == "live"
    assert youtube_live_state(
        {"liveBroadcastContent": "live"},
        {"actualStartTime": "2026-11-17T10:00:00Z", "actualEndTime": "2026-11-17T11:00:00Z"},
    ) == "ended_or_not_live"
    assert youtube_live_state({"liveBroadcastContent": "upcoming"}, {}) == "upcoming"
    assert youtube_live_state({"liveBroadcastContent": "none"}, {}) == "ended_or_not_live"


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, "quiet"), (1, "normal"), (7, "normal"), (8, "elevated"), (19, "elevated"), (20, "surge"), (200, "surge")],
)
def test_chat_activity_level_thresholds(count, expected):
    assert chat_activity_level(count) == expected
