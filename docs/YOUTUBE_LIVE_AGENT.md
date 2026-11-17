# YouTube Live Agent — metadata and chat telemetry

## Current implementation

### Inspect a YouTube URL

Authenticated endpoint: `POST /v1/youtube/agent/inspect`

Request body:

```json
{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}
```

Uses the connected account's YouTube OAuth token and official YouTube Data API `videos.list` resource. Returns title, channel, thumbnail, live/upcoming/ended state, broadcast timestamps and concurrent viewers when available.

### Read chat activity pulse

Authenticated endpoint: `GET /v1/youtube/agent/chat-pulse?broadcast_id=...`

Optional query parameters:
- `page_token`: continue from the API's next page token.
- `stream_session_id`: attach an aggregate event to an existing HYPECUT session after re-checking source ownership and unrevoked creator consent.

The endpoint only accepts an active broadcast owned by the connected YouTube account. It returns counts, paid/membership event count, polling interval, next page token and a coarse activity level (quiet/normal/elevated/surge). It does not return or store chat text, usernames, channel IDs or author details.

## Important product boundary

These endpoints are metadata and chat telemetry, not a video-watching or clipping agent. YouTube Data API metadata/chat resources do not supply the audiovisual stream. The endpoints intentionally do not scrape the player, extract playback URLs, record browser pixels, or download media. `video_capture_enabled` remains false until a permitted audiovisual access path exists.

Creator permission is necessary but is not, by itself, a technical API that supplies video bytes or overrides platform restrictions.

## Next engineering milestones

1. Add a persisted monitoring job and scheduler to re-check live status with quota-aware backoff.
2. Convert chat pulses into timestamped candidate markers and ranker features.
3. Define a pluggable `LiveMediaSource` interface. Keep the existing MediaMTX/HLS implementation as one adapter.
4. Enable a YouTube media adapter only after a documented, platform-approved audiovisual access method and retention rights are obtained.
5. Feed timestamped media windows plus engagement features into the existing highlight ranker; render clips only from media the adapter is authorised to provide.
6. Add integration tests for YouTube API responses, quota handling, consent revocation, and end-to-end clip creation.

## Response state semantics

- `monitoring_metadata`: official metadata currently reports a live broadcast.
- `waiting_for_broadcast`: metadata reports upcoming.
- `not_live`: ended, archived, or not currently live.
- Chat activity levels: `quiet` (0), `normal` (1–7), `elevated` (8–19), `surge` (20+).
- `media_access`: currently `not_available_via_youtube_data_api` for this adapter.
