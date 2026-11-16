from __future__ import annotations

import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel


SERVICE_NAME = "hypecut-engine"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
EDGE_FUNCTION_URL = f"{SUPABASE_URL}/functions/v1/hypecut-api" if SUPABASE_URL else ""
GOOGLE_CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv(
    "YOUTUBE_REDIRECT_URI",
    "https://hypecut-api-production.up.railway.app/oauth/youtube/callback",
)
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://headless-osivutkraak-aarexex-1407.wix-site-host.com",
).rstrip("/")
OAUTH_STATE_SECRET = os.getenv("HYPECUT_OAUTH_STATE_SECRET", "")
TOKEN_ENCRYPTION_KEY = os.getenv("HYPECUT_TOKEN_ENCRYPTION_KEY", "")
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    service: str
    configured: dict[str, bool]
    capabilities: dict[str, str]


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="HYPECUT Engine API",
    description=(
        "Backend for creator-authorised livestream and video clipping. "
        "Creator source and job records are connected to Supabase. YouTube OAuth "
        "can be enabled with server-side credentials. Video ingestion, AI analysis, "
        "and rendering remain unavailable until the media worker is deployed."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )


def bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer access token required")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Bearer access token required")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=503, detail="Supabase is not configured")
    return token


def oauth_state_serializer() -> URLSafeTimedSerializer:
    if not OAUTH_STATE_SECRET:
        raise HTTPException(status_code=503, detail="YouTube OAuth state secret is not configured")
    return URLSafeTimedSerializer(OAUTH_STATE_SECRET, salt="hypecut-youtube-oauth-v1")


def token_cipher() -> Fernet:
    if not TOKEN_ENCRYPTION_KEY:
        raise HTTPException(status_code=503, detail="Credential encryption is not configured")
    try:
        return Fernet(TOKEN_ENCRYPTION_KEY.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        raise HTTPException(status_code=503, detail="Credential encryption key is invalid")


def credential_store_ready() -> bool:
    return bool(SUPABASE_SERVICE_ROLE_KEY and TOKEN_ENCRYPTION_KEY)


async def edge_request(
    method: str,
    route: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> JSONResponse:
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.request(
                method,
                f"{EDGE_FUNCTION_URL}/{route}",
                headers=headers,
                json=payload if payload is not None else None,
            )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="HYPECUT data service is unavailable")
    try:
        body = response.json()
    except ValueError:
        body = {"error": "invalid_upstream_response"}
    return JSONResponse(status_code=response.status_code, content=body)


async def supabase_auth_user(token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"},
        )
    if response.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    if response.is_error:
        raise HTTPException(status_code=502, detail="Authentication provider unavailable")
    return response.json()


async def store_youtube_credentials(
    user_id: str,
    channel_id: str,
    access_token: str,
    refresh_token: str,
    expires_in: int,
    scopes: list[str],
) -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Secure credential storage is not configured")
    cipher = token_cipher()
    payload = {
        "user_id": user_id,
        "provider": "youtube",
        "external_account_id": channel_id,
        "encrypted_access_token": cipher.encrypt(access_token.encode()).decode(),
        "encrypted_refresh_token": cipher.encrypt(refresh_token.encode()).decode(),
        "expires_at": (__import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            + __import__("datetime").timedelta(seconds=max(0, expires_in))).isoformat(),
        "scopes": scopes,
    }
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/provider_credentials",
            params={"on_conflict": "user_id,provider,external_account_id"},
            headers=headers,
            json=payload,
        )
    if response.is_error:
        raise RuntimeError("Secure provider credential storage failed")


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, environment=ENVIRONMENT)


@app.get("/ready", response_model=ReadinessResponse)
async def ready() -> ReadinessResponse:
    configured = {
        "supabase_url": bool(SUPABASE_URL),
        "supabase_anon_key": bool(SUPABASE_ANON_KEY),
        "youtube_oauth_client": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
        "oauth_state_secret": bool(OAUTH_STATE_SECRET),
        "credential_encryption": bool(TOKEN_ENCRYPTION_KEY),
        "secure_credential_storage": bool(SUPABASE_SERVICE_ROLE_KEY),
        "creator_upload_storage": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
    }
    oauth_ready = all(
        configured[k] for k in (
            "youtube_oauth_client", "oauth_state_secret",
            "credential_encryption", "secure_credential_storage"
        )
    )
    return ReadinessResponse(
        status="ready" if configured["supabase_url"] and configured["supabase_anon_key"] else "needs_configuration",
        service=SERVICE_NAME,
        configured=configured,
        capabilities={
            "creator_auth_validation": "available" if configured["supabase_url"] and configured["supabase_anon_key"] else "not_configured",
            "source_registration": "connected_to_supabase_edge_api",
            "clip_job_queue": "connected_to_supabase_edge_api",
            "youtube_oauth": "configured_but_requires_user_consent" if oauth_ready else "awaiting_server_credentials",
            "youtube_live_video_ingestion": "not_implemented",
            "ai_moment_detection": "not_implemented",
            "vertical_clip_rendering": "worker_source_deployed_waiting_for_worker_credentials",
        },
    )


@app.get("/v1/me")
async def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = await supabase_auth_user(bearer_token(authorization))
    return {"id": user.get("id"), "email": user.get("email"), "authenticated": True}


@app.post("/v1/uploads/sign")
async def create_upload_url(request: Request, authorization: str | None = Header(default=None)):
    token = bearer_token(authorization)
    body = await request.json()
    filename = body.get("filename") if isinstance(body, dict) else None
    content_type = body.get("content_type") if isinstance(body, dict) else None
    allowed = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"}
    if not isinstance(filename, str) or not filename.strip() or content_type not in allowed:
        raise HTTPException(status_code=400, detail="A filename and supported video content_type are required")
    user = await supabase_auth_user(token)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename.rsplit("/", 1)[-1])[:120]
    path = f"{user['id']}/{uuid.uuid4()}-{safe_name}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{SUPABASE_URL}/storage/v1/object/upload/sign/hypecut-inputs/{path}",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"upsert": False},
        )
    if response.status_code in (401, 403):
        raise HTTPException(status_code=403, detail="Upload permission denied")
    if response.is_error:
        raise HTTPException(status_code=502, detail="Could not create secure upload URL")
    data = response.json()
    upload_token = data.get("token")
    if not upload_token:
        raise HTTPException(status_code=502, detail="Storage did not return an upload token")
    return {
        "bucket": "hypecut-inputs",
        "path": path,
        "token": upload_token,
        "upload_url": f"{SUPABASE_URL}/storage/v1/object/upload/sign/hypecut-inputs/{path}?token={upload_token}",
        "method": "PUT",
        "content_type": content_type,
        "max_size_bytes": 1073741824,
        "next_step": "Upload the file, then register it with POST /v1/sources using platform=upload and this storage_path.",
    }


@app.get("/v1/sources")
async def list_sources(authorization: str | None = Header(default=None)):
    return await edge_request("GET", "sources", bearer_token(authorization))


@app.post("/v1/sources")
async def create_source(request: Request, authorization: str | None = Header(default=None)):
    token = bearer_token(authorization)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    return await edge_request("POST", "sources", token, body)


@app.get("/v1/clips")
async def list_clips(authorization: str | None = Header(default=None)):
    return await edge_request("GET", "clips", bearer_token(authorization))


@app.post("/v1/jobs")
async def create_clip_job(request: Request, authorization: str | None = Header(default=None)):
    token = bearer_token(authorization)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    return await edge_request("POST", "jobs", token, body)


@app.get("/v1/youtube/connect")
@app.post("/v1/youtube/connect")
async def youtube_connect(request: Request, authorization: str | None = Header(default=None)):
    token = bearer_token(authorization)
    user = await supabase_auth_user(token)
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=503, detail="YouTube OAuth client credentials are not configured")
    if not OAUTH_STATE_SECRET:
        raise HTTPException(status_code=503, detail="YouTube OAuth state secret is not configured")
    state = oauth_state_serializer().dumps({"user_id": user["id"]})
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(YOUTUBE_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    authorization_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    if request.method == "POST":
        return {"authorization_url": authorization_url}
    return RedirectResponse(authorization_url, status_code=302)


@app.get("/oauth/youtube/callback")
async def youtube_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    def finish(result: str) -> RedirectResponse:
        return RedirectResponse(f"{FRONTEND_URL}/dashboard.html?youtube={result}", status_code=303)

    if error:
        return finish("cancelled")
    if not code or not state:
        return finish("failed")
    try:
        identity = oauth_state_serializer().loads(state, max_age=600)
    except (BadSignature, SignatureExpired):
        return finish("invalid_state")
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return finish("not_configured")
    if not credential_store_ready():
        return finish("secure_storage_not_configured")

    async with httpx.AsyncClient(timeout=20.0) as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if token_response.is_error:
            return finish("token_exchange_failed")
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        if not access_token or not refresh_token:
            return finish("refresh_consent_required")
        channel_response = await client.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "id,snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if channel_response.is_error:
            return finish("channel_lookup_failed")
        channels = channel_response.json().get("items", [])
        if not channels:
            return finish("no_channel_found")
        channel = channels[0]

    try:
        await store_youtube_credentials(
            user_id=identity["user_id"],
            channel_id=channel["id"],
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(token_data.get("expires_in", 3600)),
            scopes=YOUTUBE_SCOPES,
        )
    except Exception:
        return finish("credential_storage_failed")
    return finish("connected")


@app.get("/v1/capabilities")
async def capabilities() -> dict[str, Any]:
    return {
        "product": SERVICE_NAME,
        "modes": ["live", "video"],
        "clip_target_seconds": {"minimum": 30, "maximum": 60},
        "implemented": [
            "health", "readiness", "supabase_access_token_validation",
            "creator_owned_source_registration_proxy", "owner_scoped_source_and_clip_listing",
            "consent_checked_clip_job_queueing", "signed_creator_video_upload_urls",
            "youtube_oauth_authorization_flow", "encrypted_youtube_token_persistence_when_secrets_are_configured",
            "ffmpeg_worker_source", "audio_energy_highlight_selection", "vertical_9_by_16_rendering",
            "private_clip_storage", "signed_clip_download_urls",
            "optional_openai_whisper_caption_generation",
        ],
        "not_yet_implemented": [
            "youtube_live_video_ingestion", "live_rtmp_ingestion",
            "semantic_multimodal_ai_highlight_detection", "live_chat_reaction_analysis",
            "scene_change_and_audio_fusion_scoring", "real_time_stream_monitoring",
        ],
        "rights_policy": "Only creator-owned or explicitly authorised content may be processed.",
    }
