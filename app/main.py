from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


SERVICE_NAME = "hypecut-engine"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


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
        "Backend foundation for creator-authorised livestream and video clipping. "
        "Media ingestion, AI analysis, and rendering remain disabled until their "
        "providers and worker are implemented and verified."
    ),
    version="0.1.0",
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


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        environment=ENVIRONMENT,
    )


@app.get("/ready", response_model=ReadinessResponse)
async def ready() -> ReadinessResponse:
    configured = {
        "supabase_url": bool(SUPABASE_URL),
        "supabase_anon_key": bool(SUPABASE_ANON_KEY),
    }
    return ReadinessResponse(
        status="ready" if all(configured.values()) else "needs_configuration",
        service=SERVICE_NAME,
        configured=configured,
        capabilities={
            "creator_auth_validation": (
                "available" if all(configured.values()) else "not_configured"
            ),
            "youtube_live_ingestion": "not_implemented",
            "ai_moment_detection": "not_implemented",
            "vertical_clip_rendering": "not_implemented",
            "clip_storage_delivery": "not_implemented",
        },
    )


@app.get("/v1/me")
async def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Validate a Supabase access token without exposing service-role credentials."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer access token required")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Bearer access token required")

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {token}",
            },
        )

    if response.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    if response.is_error:
        raise HTTPException(status_code=502, detail="Authentication provider unavailable")

    user = response.json()
    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "authenticated": True,
    }


@app.get("/v1/capabilities")
async def capabilities() -> dict[str, Any]:
    return {
        "product": SERVICE_NAME,
        "modes": ["live", "video"],
        "clip_target_seconds": {"minimum": 30, "maximum": 60},
        "implemented": ["health", "readiness", "supabase_access_token_validation"],
        "not_yet_implemented": [
            "youtube_oauth",
            "live_stream_ingestion",
            "video_download_or_ingestion",
            "audio_video_chat_analysis",
            "moment_scoring",
            "vertical_reframing",
            "caption_generation",
            "clip_rendering",
            "clip_delivery",
        ],
        "rights_policy": "Only creator-owned or explicitly authorised content may be processed.",
    }
