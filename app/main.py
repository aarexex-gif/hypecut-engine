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
# Supabase Free projects have a 50 MB project-wide upload ceiling.
# Raise this server-side value only after the project plan/global limit is changed.
MAX_UPLOAD_BYTES = int(os.getenv("HYPECUT_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
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
# The dashboard uses bearer-token authentication (not cross-site cookies).
# Allow Wix-hosted frontend origins and preflight headers used by mobile browsers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["https://headless-osivutkraak-aarexex-1407.wix-site-host.com"],
    allow_origin_regex=r"https://[a-zA-Z0-9-]+\\.wix-site-host\\.com",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "max_size_bytes": MAX_UPLOAD_BYTES,
        "max_size_label": f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB",
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
    user = await supabase_auth_user(token)
    request_key = body.get("client_request_id") or str(uuid.uuid4())
    if not isinstance(request_key, str) or not request_key or len(request_key) > 120:
        raise HTTPException(status_code=400, detail="Invalid client_request_id")
    await apply_billing_credit(user["id"], -1, "usage", f"clip-job:{user['id']}:{request_key}",
                               None, "HYPECUT clip processing")
    result = await edge_request("POST", "jobs", token, body)
    if result.status_code >= 400:
        await apply_billing_credit(user["id"], 1, "refund", f"clip-job-refund:{user['id']}:{request_key}",
                                   None, "Refund for rejected clip job")
    return result


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



# Billing: INR launch catalogue. Prices/allowances are configuration defaults, not a
# claim that checkout is active; Razorpay credentials must be configured first.
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
CREDIT_PACKS = {
    "starter": {"credits": 25, "amount_paise": 9900, "label": "25 processing credits"},
    "creator": {"credits": 100, "amount_paise": 29900, "label": "100 processing credits"},
    "studio": {"credits": 250, "amount_paise": 59900, "label": "250 processing credits"},
}
SUBSCRIPTION_PLANS = {
    "creator": {"amount_paise": 49900, "monthly_credits": 150, "label": "Creator"},
    "pro": {"amount_paise": 99900, "monthly_credits": 400, "label": "Pro Creator"},
    "agency": {"amount_paise": 249900, "monthly_credits": 1200, "label": "Agency"},
}


async def billing_rest(method: str, table: str, *, params: dict[str, str] | None = None,
                       payload: Any = None) -> Any:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Billing database is not configured")
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.request(method, f"{SUPABASE_URL}/rest/v1/{table}",
                                        params=params, headers=headers, json=payload)
    if response.is_error:
        raise HTTPException(status_code=502, detail="Billing database operation failed")
    return response.json() if response.content else None


async def apply_billing_credit(user_id: str, delta: int, entry_type: str,
                               idempotency_key: str, reference: str | None = None,
                               description: str | None = None) -> None:
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "p_user_id": user_id, "p_delta": delta, "p_entry_type": entry_type,
        "p_idempotency_key": idempotency_key, "p_provider_reference": reference,
        "p_description": description, "p_metadata": {},
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{SUPABASE_URL}/rest/v1/rpc/apply_credit_delta",
                                     headers=headers, json=payload)
    if response.is_error:
        if response.status_code == 400 and "Insufficient credits" in response.text:
            raise HTTPException(status_code=402, detail="Not enough processing credits")
        raise HTTPException(status_code=502, detail="Could not update credit balance")


def require_razorpay() -> None:
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=503, detail="Payments are not enabled yet")


@app.get("/v1/billing/catalog")
async def billing_catalog():
    return {
        "currency": "INR",
        "market": "India",
        "subscriptions": SUBSCRIPTION_PLANS,
        "credit_packs": CREDIT_PACKS,
        "checkout_enabled": bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET),
        "subscription_checkout_ready": bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET and all(os.getenv(f"RAZORPAY_PLAN_{code.upper()}") for code in SUBSCRIPTION_PLANS)),
        "note": "Prices and allowances are proposed launch settings and may be adjusted before public sale.",
    }


@app.get("/v1/billing/me")
async def billing_me(authorization: str | None = Header(default=None)):
    user = await supabase_auth_user(bearer_token(authorization))
    uid = user["id"]
    wallets = await billing_rest("GET", "billing_wallets",
                                 params={"user_id": f"eq.{uid}", "select": "balance,plan_code,period_credits,period_ends_at"})
    orders = await billing_rest("GET", "billing_orders",
                                params={"user_id": f"eq.{uid}", "select": "pack_code,amount_paise,currency,credits,status,created_at,paid_at", "order": "created_at.desc", "limit": "10"})
    subscriptions = await billing_rest("GET", "billing_subscriptions",
                                       params={"user_id": f"eq.{uid}", "select": "plan_code,status,current_start,current_end,cancel_at_cycle_end", "order": "created_at.desc", "limit": "1"})
    wallet = wallets[0] if wallets else {"balance": 0, "plan_code": "free", "period_credits": 0}
    return {"wallet": wallet, "recent_orders": orders or [],
            "subscription": subscriptions[0] if subscriptions else None}


@app.post("/v1/billing/orders")
async def create_credit_order(request: Request, authorization: str | None = Header(default=None)):
    require_razorpay()
    user = await supabase_auth_user(bearer_token(authorization))
    body = await request.json()
    pack_code = body.get("pack_code") if isinstance(body, dict) else None
    pack = CREDIT_PACKS.get(pack_code)
    if not pack:
        raise HTTPException(status_code=400, detail="Choose a valid credit pack")
    receipt = f"hc_{uuid.uuid4().hex[:24]}"
    order_payload = {
        "amount": pack["amount_paise"], "currency": "INR", "receipt": receipt,
        "notes": {"user_id": user["id"], "pack_code": pack_code, "credits": str(pack["credits"])},
    }
    async with httpx.AsyncClient(timeout=20.0, auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)) as client:
        response = await client.post("https://api.razorpay.com/v1/orders", json=order_payload)
    if response.is_error:
        raise HTTPException(status_code=502, detail="Payment provider could not create an order")
    provider_order = response.json()
    await billing_rest("POST", "billing_orders", payload={
        "user_id": user["id"], "provider_order_id": provider_order["id"],
        "pack_code": pack_code, "amount_paise": pack["amount_paise"],
        "currency": "INR", "credits": pack["credits"], "status": "created",
    })
    return {"order_id": provider_order["id"], "amount": pack["amount_paise"],
            "currency": "INR", "key_id": RAZORPAY_KEY_ID,
            "credits": pack["credits"], "pack_code": pack_code}


@app.post("/v1/billing/orders/verify")
async def verify_credit_order(request: Request, authorization: str | None = Header(default=None)):
    require_razorpay()
    user = await supabase_auth_user(bearer_token(authorization))
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    order_id, payment_id, signature = body.get("razorpay_order_id"), body.get("razorpay_payment_id"), body.get("razorpay_signature")
    if not all(isinstance(v, str) and v for v in (order_id, payment_id, signature)):
        raise HTTPException(status_code=400, detail="Payment verification details are required")
    expected = hmac.new(RAZORPAY_KEY_SECRET.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Payment signature is invalid")
    rows = await billing_rest("GET", "billing_orders", params={
        "provider_order_id": f"eq.{order_id}", "user_id": f"eq.{user['id']}",
        "select": "id,credits,status,amount_paise", "limit": "1",
    })
    if not rows:
        raise HTTPException(status_code=404, detail="Payment order not found")
    order = rows[0]
    if order["status"] != "paid":
        # Confirm capture with Razorpay; a valid checkout signature alone is not a capture.
        async with httpx.AsyncClient(timeout=20.0, auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)) as client:
            payment_response = await client.get(f"https://api.razorpay.com/v1/payments/{payment_id}")
        if payment_response.is_error:
            raise HTTPException(status_code=502, detail="Could not confirm payment status")
        payment = payment_response.json()
        if payment.get("order_id") != order_id or payment.get("status") != "captured" or payment.get("amount") != order["amount_paise"]:
            raise HTTPException(status_code=409, detail="Payment has not been captured for this order")
        await apply_billing_credit(user["id"], order["credits"], "credit_purchase",
                                   f"purchase:{order_id}", payment_id, "HYPECUT credit pack")
        await billing_rest("PATCH", "billing_orders",
                           params={"provider_order_id": f"eq.{order_id}", "user_id": f"eq.{user['id']}"},
                           payload={"status": "paid", "provider_payment_id": payment_id, "paid_at": datetime.now(timezone.utc).isoformat()})
    return {"verified": True, "credits_added": order["credits"]}



@app.post("/v1/billing/subscriptions")
async def create_billing_subscription(request: Request, authorization: str | None = Header(default=None)):
    require_razorpay()
    user = await supabase_auth_user(bearer_token(authorization))
    body = await request.json()
    plan_code = body.get("plan_code") if isinstance(body, dict) else None
    plan = SUBSCRIPTION_PLANS.get(plan_code)
    plan_id = os.getenv(f"RAZORPAY_PLAN_{str(plan_code).upper()}", "")
    if not plan or not plan_id:
        raise HTTPException(status_code=503, detail="This subscription plan is not configured for checkout yet")
    subscription_payload = {
        "plan_id": plan_id,
        "total_count": 120,
        "quantity": 1,
        "customer_notify": 1,
        "notes": {"user_id": user["id"], "plan_code": plan_code},
    }
    async with httpx.AsyncClient(timeout=20.0, auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)) as client:
        response = await client.post("https://api.razorpay.com/v1/subscriptions", json=subscription_payload)
    if response.is_error:
        raise HTTPException(status_code=502, detail="Payment provider could not create a subscription")
    provider_sub = response.json()
    await billing_rest("POST", "billing_subscriptions", payload={
        "user_id": user["id"], "provider_subscription_id": provider_sub["id"],
        "provider_plan_id": plan_id, "plan_code": plan_code, "status": provider_sub.get("status", "created"),
    })
    return {"subscription_id": provider_sub["id"], "short_url": provider_sub.get("short_url"),
            "plan_code": plan_code, "amount_paise": plan["amount_paise"],
            "monthly_credits": plan["monthly_credits"]}


@app.post("/v1/billing/webhook")
async def razorpay_billing_webhook(request: Request):
    if not RAZORPAY_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Payment webhook is not configured")
    raw = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    expected = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    if not signature or not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    event = await request.json()
    event_id = request.headers.get("x-razorpay-event-id")
    if not event_id:
        raise HTTPException(status_code=400, detail="Webhook event ID is required")
    existing = await billing_rest("GET", "billing_webhook_events",
                                  params={"event_id": f"eq.{event_id}", "select": "event_id", "limit": "1"})
    if existing:
        return {"received": True, "duplicate": True, "event_id": event_id}
    await billing_rest("POST", "billing_webhook_events",
                       payload={"event_id": event_id, "event_type": event.get("event", "unknown"), "payload": event})
    event_type = event.get("event")
    subscription = (event.get("payload") or {}).get("subscription", {}).get("entity", {})
    payment = (event.get("payload") or {}).get("payment", {}).get("entity", {})
    if event_type in ("subscription.activated", "subscription.charged") and subscription.get("id"):
        rows = await billing_rest("GET", "billing_subscriptions", params={
            "provider_subscription_id": f"eq.{subscription['id']}",
            "select": "user_id,plan_code,status", "limit": "1",
        })
        if rows:
            sub = rows[0]
            plan = SUBSCRIPTION_PLANS[sub["plan_code"]]
            if event_type == "subscription.charged" and payment.get("id"):
                await apply_billing_credit(sub["user_id"], plan["monthly_credits"], "monthly_grant",
                                           f"subscription-charge:{payment['id']}", payment["id"],
                                           f"HYPECUT {sub['plan_code']} monthly credits")
            await billing_rest("PATCH", "billing_subscriptions",
                               params={"provider_subscription_id": f"eq.{subscription['id']}"},
                               payload={"status": subscription.get("status", "active"),
                                        "current_start": datetime.fromtimestamp(subscription.get("current_start", 0), timezone.utc).isoformat() if subscription.get("current_start") else None,
                                        "current_end": datetime.fromtimestamp(subscription.get("current_end", 0), timezone.utc).isoformat() if subscription.get("current_end") else None})
            await billing_rest("PATCH", "billing_wallets",
                               params={"user_id": f"eq.{sub['user_id']}"},
                               payload={"plan_code": sub["plan_code"], "period_credits": plan["monthly_credits"],
                                        "period_ends_at": datetime.fromtimestamp(subscription.get("current_end", 0), timezone.utc).isoformat() if subscription.get("current_end") else None})
    elif event_type in ("subscription.cancelled", "subscription.halted") and subscription.get("id"):
        await billing_rest("PATCH", "billing_subscriptions",
                           params={"provider_subscription_id": f"eq.{subscription['id']}"},
                           payload={"status": subscription.get("status", "cancelled")})
    return {"received": True, "event_id": event_id, "event_type": event_type}

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
