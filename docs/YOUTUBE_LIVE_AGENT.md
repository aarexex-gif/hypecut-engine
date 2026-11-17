# YouTube Live Agent — metadata monitor

## Current implementation

The authenticated endpoint `POST /v1/youtube/agent/inspect` accepts a public YouTube URL and uses the connected account's YouTube OAuth token with the official YouTube Data API `videos.list` resource.

Example request:

```http
POST /v1/youtube/agent/inspect
Authorization: Bearer <supabase-access-token>
Content-Type: application/json

{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}
```

The endpoint returns the video ID, title, channel, thumbnail, live/upcoming/ended state, start/end timestamps when available, concurrent viewer count when exposed, and an explicit media-access status.

Accepted URL forms: youtube.com/watch?v=..., youtu.be/..., youtube.com/live/..., youtube.com/embed/..., youtube.com/shorts/...

## Important product boundary

This is a metadata inspection agent, not a video-watching or clipping agent. YouTube Data API metadata does not contain the audiovisual stream. The endpoint intentionally does not scrape the player, extract playback URLs, record browser pixels, or download media. It returns `video_capture_enabled: false` until a permitted audiovisual access path exists.

Creator permission is necessary but is not, by itself, a technical API that supplies video bytes or overrides platform restrictions.

## Next engineering milestones

1. Add a persisted monitoring job and scheduler to re-check live status with quota-aware backoff.
2. Add authorised live-chat polling where the connected account has access; store aggregate engagement signals and minimise viewer data.
3. Define a pluggable `LiveMediaSource` interface. Keep the existing MediaMTX/HLS implementation as one adapter.
4. Enable a YouTube media adapter only after a documented, platform-approved audiovisual access method and retention rights are obtained.
5. Feed timestamped media windows plus engagement features into the existing highlight ranker; render clips only from media the adapter is authorised to provide.
6. Add integration tests for URL parsing, live-state transitions, API errors, consent revocation, and end-to-end clip creation.

## Response state semantics

- `monitoring_metadata`: official metadata currently reports a live broadcast.
- `waiting_for_broadcast`: metadata reports upcoming.
- `not_live`: ended, archived, or not currently live.
- `media_access`: currently `not_available_via_youtube_data_api` for this adapter.
