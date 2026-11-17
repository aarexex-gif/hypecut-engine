"""API-level tests for YouTube live-agent endpoints using mocked providers."""
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(monkeypatch):
    async def fake_auth(_token):
        return {"id": "user-123"}

    async def fake_access(_user_id):
        return "youtube-access-token"

    monkeypatch.setattr(main, "bearer_token", lambda _authorization: "test-token")
    monkeypatch.setattr(main, "supabase_auth_user", fake_auth)
    monkeypatch.setattr(main, "youtube_access_token", fake_access)
    return TestClient(main.app)


def test_inspect_endpoint_reports_live_metadata_without_media_capture(client, monkeypatch):
    async def fake_youtube(path, _access, _params):
        assert path == "videos"
        return {"items": [{
            "id": "abcdefghijk",
            "snippet": {
                "title": "Test stream",
                "channelId": "channel-1",
                "channelTitle": "Creator",
                "liveBroadcastContent": "live",
                "thumbnails": {"high": {"url": "https://example.test/thumb.jpg"}},
            },
            "liveStreamingDetails": {"actualStartTime": "2026-11-17T10:00:00Z", "concurrentViewers": "123"},
            "status": {"privacyStatus": "public"},
        }]}

    monkeypatch.setattr(main, "youtube_api_get", fake_youtube)
    response = client.post(
        "/v1/youtube/agent/inspect",
        headers={"Authorization": "Bearer test-token"},
        json={"url": "https://youtu.be/abcdefghijk"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["broadcast_state"] == "live"
    assert body["agent_status"] == "monitoring_metadata"
    assert body["concurrent_viewers"] == "123"
    assert body["video_capture_enabled"] is False
    assert body["media_access"] == "not_available_via_youtube_data_api"


def test_chat_pulse_returns_aggregate_metrics_only(client, monkeypatch):
    async def fake_youtube(path, _access, params):
        if path == "liveBroadcasts":
            return {"items": [{"id": "broadcast-1", "snippet": {"liveChatId": "chat-1"}}]}
        assert path == "liveChatMessages"
        assert params["part"] == "id,snippet"
        return {
            "items": [{"snippet": {"type": "textMessageEvent"}} for _ in range(9)]
                    + [{"snippet": {"type": "superChatEvent"}}],
            "nextPageToken": "next-page",
            "pollingIntervalMillis": 6000,
        }

    monkeypatch.setattr(main, "youtube_api_get", fake_youtube)
    response = client.get(
        "/v1/youtube/agent/chat-pulse?broadcast_id=broadcast-1",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["message_count"] == 10
    assert body["paid_or_membership_event_count"] == 1
    assert body["activity_level"] == "elevated"
    assert body["next_page_token"] == "next-page"
    assert "messages" not in body
    assert "display_message" not in response.text


def test_timeline_filters_transcripts_and_worker_details(client, monkeypatch):
    async def fake_billing(method, table, params=None, payload=None):
        if table == "stream_sessions":
            return [{"id": "session-1", "source_id": "source-1", "status": "live"}]
        if table == "analysis_events":
            return [{
                "id": "event-1",
                "event_type": "moment_candidate",
                "occurred_at": "2026-11-17T10:00:00Z",
                "offset_seconds": None,
                "score": 88,
                "payload": {
                    "title": "Great moment",
                    "hook": "Watch this",
                    "transcript": "private transcript",
                    "worker_id": "internal-worker",
                },
            }]
        raise AssertionError(f"Unexpected table: {table}")

    monkeypatch.setattr(main, "billing_rest", fake_billing)
    response = client.get(
        "/v1/streams/session-1/timeline",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    payload = body["events"][0]["payload"]
    assert payload == {"title": "Great moment", "hook": "Watch this"}
    assert "private transcript" not in response.text
    assert "internal-worker" not in response.text


def test_chat_pulse_supports_consent_checked_public_broadcast(client, monkeypatch):
    async def fake_billing(method, table, *, params=None, payload=None):
        if table == "sources":
            return [{
                "id": "source-1",
                "platform": "youtube",
                "source_type": "live",
                "external_id": "abcdefghijk",
                "source_url": "https://www.youtube.com/watch?v=abcdefghijk",
                "consent_id": "consent-1",
                "status": "authorised",
            }]
        if table == "creator_consents":
            return [{"rights_granted": True, "revoked_at": None}]
        raise AssertionError(f"Unexpected table: {table}")

    async def fake_youtube(path, _access, params):
        if path == "liveBroadcasts":
            return {"items": []}
        if path == "videos":
            return {"items": [{
                "id": "abcdefghijk",
                "liveStreamingDetails": {"activeLiveChatId": "public-chat-1"},
            }]}
        if path == "liveChatMessages":
            assert params["liveChatId"] == "public-chat-1"
            return {
                "items": [{"snippet": {"type": "textMessageEvent"}} for _ in range(4)],
                "pollingIntervalMillis": 5000,
            }
        raise AssertionError(f"Unexpected YouTube resource: {path}")

    monkeypatch.setattr(main, "billing_rest", fake_billing)
    monkeypatch.setattr(main, "youtube_api_get", fake_youtube)
    response = client.get(
        "/v1/youtube/agent/chat-pulse?broadcast_id=abcdefghijk&source_id=source-1",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chat_access_mode"] == "consented_public_broadcast"
    assert body["message_count"] == 4
    assert "display_message" not in response.text
