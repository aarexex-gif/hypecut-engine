# HYPECUT Engine

**Watch live. Catch the hype. Create automatically.**

HYPECUT is an AI-assisted livestream and video clipping platform for creators. Its goal is to analyse creator-authorised sources, identify engaging moments, and produce vertical clips with captions and smart reframing.

## Repository contents

- `app/main.py` — FastAPI service foundation
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build
- `.env.example` — safe configuration template

## Current implementation status

Implemented:
- API health endpoint (`GET /health`)
- Readiness/configuration endpoint (`GET /ready`)
- Supabase access-token validation endpoint (`GET /v1/me`)
- Capability/status endpoint (`GET /v1/capabilities`)
- Container build configuration

Implemented in code:
- Authenticated signed-upload URL endpoint for creator-owned video files
- Private input bucket with per-user storage policies
- Creator consent records for uploaded sources
- Atomic queued-job claim RPC restricted to the server service role
- FFmpeg worker for uploaded media: audio-energy highlight selection, 9:16 rendering, private clip storage
- Optional OpenAI Whisper SRT captions when an OpenAI API key is configured
- Signed clip download URLs in the clips listing

Not implemented yet:
- YouTube live/video media ingestion (YouTube OAuth alone does not expose a general stream-download feed)
- Creator RTMP contribution / live capture
- Semantic multimodal AI highlight understanding
- Live chat reaction analysis and fused scene/audio scoring
- Full browser dashboard integration and end-to-end creator acceptance test

The API intentionally reports these capabilities as unavailable until they are implemented and tested. It does not fabricate processing status, analytics, or clips.

## Local run

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set the Supabase project URL and publishable/anon key.

3. Start the API:

   ```bash
   uvicorn app.main:app --reload --port 8080
   ```

4. Open `http://localhost:8080/docs` for the API documentation and `/health` for the health check.

On Windows PowerShell, activate the environment with `.venv\\Scripts\\Activate.ps1`.

## Deployment

The Dockerfile starts Uvicorn on port 8080. Configure environment variables in the hosting provider rather than committing secrets.

Required:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`

Optional:
- `CORS_ORIGINS` — comma-separated allowed frontend origins
- `ENVIRONMENT` — defaults to `development`

Never commit Supabase service-role keys, OAuth client secrets, or AI provider keys.

## Creator rights and platform compliance

Only process content the creator owns or is explicitly authorised to use. Integrations must use appropriate OAuth scopes and respect provider terms, API policies, and applicable law.


## YouTube OAuth setup

The API includes a Google OAuth authorization-code flow for the creator's own YouTube account. It requests read-only YouTube access, fetches the channel identity, encrypts access/refresh tokens with Fernet, and stores them in the private `provider_credentials` table using the server-only Supabase service key.

Set these server-side Railway variables before enabling the connection:

- `YOUTUBE_CLIENT_ID` and `YOUTUBE_CLIENT_SECRET` from a Google Cloud OAuth client
- `YOUTUBE_REDIRECT_URI` set to `https://hypecut-api-production.up.railway.app/oauth/youtube/callback` and registered exactly in Google Cloud
- `HYPECUT_OAUTH_STATE_SECRET` — a long random secret
- `HYPECUT_TOKEN_ENCRYPTION_KEY` — a Fernet URL-safe base64 key
- `SUPABASE_SERVICE_ROLE_KEY` — the project's server-side secret/service-role key

The client secret, service key, state secret and encryption key must never be placed in Wix/frontend code or committed to Git. The connect endpoint is `GET /v1/youtube/connect` and requires a Supabase bearer access token.

**Important platform limitation:** YouTube OAuth authorizes account access and read-only channel data. It does not provide a general-purpose video stream download endpoint. Actual live video ingestion requires a creator-provided stream feed/RTMP contribution or another permitted source integration. This repository does not scrape or download YouTube streams.

OAuth is not operational until the required Google and Supabase server secrets are configured and a creator grants consent.

## Creator upload workflow

1. Sign in and obtain a Supabase Auth access token.
2. Call `POST /v1/uploads/sign` with that bearer token and JSON containing `filename` and `content_type`.
3. Upload the file to the returned `upload_url` using HTTP `PUT` and the returned content type.
4. Register the uploaded media with `POST /v1/sources`, using `platform: "upload"`, `source_type: "video"`, `storage_path` from the upload response, and `rights_granted: true`.
5. Create a job with `POST /v1/jobs` and the returned source ID.
6. The worker claims the job, selects a sustained high-energy audio window, renders a 9:16 MP4 with FFmpeg, and stores it in the private output bucket. This is a signal-based baseline, not semantic AI highlight understanding.

The worker service requires `SUPABASE_SERVICE_ROLE_KEY` in Railway. Optional captions require `OPENAI_API_KEY`. The API's YouTube OAuth flow additionally requires `YOUTUBE_CLIENT_ID` and `YOUTUBE_CLIENT_SECRET`. Add secrets directly in Railway Variables; never paste them into GitHub or frontend code.
