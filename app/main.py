from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


SERVICE_NAME = "hypecut-engine"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
EDGE_FUNCTION_URL = f"{SUPABASE_URL}/functions/v1/hypecut-api" if SUPABASE_URL else ""


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
        "Source registration and job queueing are connected to the HYPECUT "
        "Supabase API. Media ingestion, AI analysis, and rendering are not yet deployed."
    ),
    version="0.2.0",
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


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, environment=ENVIRONMENT)


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
            "creator_auth_validation": "available" if all(configured.values()) else "not_configured",
            "source_registration": "connected_to_supabase_edge_api" if all(configured.values()) else "not_configured",
            "clip_job_queue": "connected_to_supabase_edge_api" if all(configured.values()) else "not_configured",
            "youtube_live_ingestion": "not_implemented",
            "ai_moment_detection": "not_implemented",
            "vertical_clip_rendering": "not_implemented",
        },
    )


@app.get("/v1/me")
async def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = bearer_token(authorization)
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
    return {"id": user.get("id"), "email": user.get("email"), "authenticated": True}


@app.get("/v1/sources")
async def list_sources(authorization: str | None = Header(default=None)):
    return await edge_request("GET", "sources", bearer_token(authorization))


@app.post("/v1/sources")
async def create_source(
    request: Request,
    authorization: str | None = Header(default=None),
):
    token = bearer_token(authorization)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    return await edge_request("POST", "sources", token, body)


@app.get("/v1/clips")
async def list_clips(authorization: str | None = Header(default=None)):
    return await edge_request("GET", "clips", bearer_token(authorization))


@app.post("/v1/jobs")
async def create_clip_job(
    request: Request,
    authorization: str | None = Header(default=None),
):
    token = bearer_token(authorization)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    return await edge_request("POST", "jobs", token, body)


@app.get("/v1/capabilities")
async def capabilities() -> dict[str, Any]:
    return {
        "product": SERVICE_NAME,
        "modes": ["live", "video"],
        "clip_target_seconds": {"minimum": 30, "maximum": 60},
        "implemented": [
            "health",
            "readiness",
            "supabase_access_token_validation",
            "creator_owned_source_registration_proxy",
            "owner_scoped_source_and_clip_listing",
            "consent_checked_clip_job_queueing",
        ],
        "not_yet_implemented": [
            "youtube_oauth",
            "live_stream_ingestion",
            "video_media_ingestion",
            "audio_video_chat_analysis",
            "moment_scoring",
            "vertical_reframing",
            "caption_generation",
            "clip_rendering",
            "clip_delivery",
        ],
        "rights_policy": "Only creator-owned or explicitly authorised content may be processed.",
    }
