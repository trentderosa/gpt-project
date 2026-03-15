import logging
import os
import re
import time
import threading
import base64
import mimetypes
import json
from collections import defaultdict, deque
from io import BytesIO
from datetime import datetime
from uuid import uuid4

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, AuthenticationError, BadRequestError, RateLimitError
from pydantic import BaseModel, Field
import stripe

from .core.chat_service import ChatService
from .core.config import DEFAULT_MODEL, KNOWLEDGE_DIR, WEB_DIR, WEB_SEARCH_PROVIDER
from .core.llm_wrapper import LLMWrapper
from .core.live_data_store import LiveDataStore
from .core.retriever import load_knowledge_chunks
from .core.search_tool import (
    BraveSearchTool,
    CompositeWebSearchTool,
    DisabledWebSearchTool,
    DuckDuckGoSearchTool,
    WikipediaSearchTool,
)
from .core.storage import ChatStorage
from .jobs.updater import run_once


app = FastAPI(title="Cortex Engine API (Created by Trent DeRosa)", version="0.1.0")
logger = logging.getLogger("trent_gpt_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

allowed_origins_raw = (os.getenv("ALLOWED_ORIGINS") or "").strip()
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

storage = ChatStorage()
live_store = LiveDataStore()
chunks = load_knowledge_chunks(KNOWLEDGE_DIR)
llm = LLMWrapper(model=DEFAULT_MODEL)
if WEB_SEARCH_PROVIDER in {"duckduckgo", "multi", "auto"}:
    providers = [DuckDuckGoSearchTool(), WikipediaSearchTool()]
    brave_key = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
    if brave_key:
        providers.insert(0, BraveSearchTool(brave_key))
    search_tool = CompositeWebSearchTool(providers=providers)
else:
    search_tool = DisabledWebSearchTool()
chat_service = ChatService(llm=llm, chunks=chunks, web_search_tool=search_tool)
embedded_scheduler: BackgroundScheduler | None = None
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_TEXT_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_FILENAME_LEN = 180
MAX_MESSAGE_CHARS = 6000
BLOCKED_UPLOAD_EXTENSIONS = {
    ".exe",
    ".dll",
    ".bat",
    ".cmd",
    ".ps1",
    ".sh",
    ".msi",
    ".com",
    ".scr",
    ".jar",
    ".vbs",
}
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
CHAT_RATE_LIMIT_PER_MIN = int(os.getenv("CHAT_RATE_LIMIT_PER_MIN", "60"))
UPLOAD_RATE_LIMIT_PER_MIN = int(os.getenv("UPLOAD_RATE_LIMIT_PER_MIN", "20"))
AUTH_RATE_LIMIT_PER_MIN = int(os.getenv("AUTH_RATE_LIMIT_PER_MIN", "20"))
AUTH_EMAIL_RATE_LIMIT_PER_15MIN = int(os.getenv("AUTH_EMAIL_RATE_LIMIT_PER_15MIN", "12"))
FREE_INPUTS_PER_HOUR = int(os.getenv("FREE_INPUTS_PER_HOUR", "30"))
FREE_IMAGES_PER_HOUR = int(os.getenv("FREE_IMAGES_PER_HOUR", "1"))
PASSWORD_RESET_TOKEN_TTL_MINUTES = int(os.getenv("PASSWORD_RESET_TOKEN_TTL_MINUTES", "30"))
PASSWORD_RESET_DEV_MODE = (os.getenv("PASSWORD_RESET_DEV_MODE", "false").strip().lower() == "true")
SESSION_TTL_DAYS = int(os.getenv("SESSION_TTL_DAYS", "180"))
SESSION_SLIDING_RENEWAL = os.getenv("SESSION_SLIDING_RENEWAL", "true").strip().lower() == "true"
SESSION_ABSOLUTE_MAX_DAYS = int(os.getenv("SESSION_ABSOLUTE_MAX_DAYS", "365"))
CREATOR_BOOTSTRAP_SECRET = (os.getenv("CREATOR_BOOTSTRAP_SECRET") or "").strip()
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "cortex_session").strip()
_cookie_secure_env = os.getenv("COOKIE_SECURE")
COOKIE_SECURE = False
PLAN_CONFIG = {
    "free": {"price_usd_month": 0, "inputs_per_hour": 30, "images_per_hour": 1, "unlimited": False},
    "pro5": {"price_usd_month": 5, "inputs_per_hour": 100, "images_per_hour": 3, "unlimited": False},
    "pro10": {"price_usd_month": 10, "inputs_per_hour": None, "images_per_hour": None, "unlimited": True},
    "creator": {"price_usd_month": 0, "inputs_per_hour": None, "images_per_hour": None, "unlimited": True},
}
STRIPE_SECRET_KEY = (os.getenv("STRIPE_SECRET_KEY") or "").strip()
STRIPE_WEBHOOK_SECRET = (os.getenv("STRIPE_WEBHOOK_SECRET") or "").strip()
APP_BASE_URL = (os.getenv("APP_BASE_URL") or "http://127.0.0.1:8000").strip().rstrip("/")
COOKIE_SECURE = (
    _cookie_secure_env.strip().lower() == "true"
    if _cookie_secure_env is not None
    else APP_BASE_URL.startswith("https://")
)
STRIPE_PRICE_IDS = {
    "pro5": (os.getenv("STRIPE_PRICE_PRO5_ID") or "").strip(),
    "pro10": (os.getenv("STRIPE_PRICE_PRO10_ID") or "").strip(),
}
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = max(limit, 1)
        self.window_seconds = max(window_seconds, 1)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            queue = self._events[key]
            cutoff = now - self.window_seconds
            while queue and queue[0] < cutoff:
                queue.popleft()
            if len(queue) >= self.limit:
                return False
            queue.append(now)
            return True


chat_rate_limiter = SlidingWindowRateLimiter(CHAT_RATE_LIMIT_PER_MIN, window_seconds=60)
upload_rate_limiter = SlidingWindowRateLimiter(UPLOAD_RATE_LIMIT_PER_MIN, window_seconds=60)
auth_rate_limiter = SlidingWindowRateLimiter(AUTH_RATE_LIMIT_PER_MIN, window_seconds=60)
auth_email_rate_limiter = SlidingWindowRateLimiter(AUTH_EMAIL_RATE_LIMIT_PER_15MIN, window_seconds=900)
anon_input_hour_limiter = SlidingWindowRateLimiter(FREE_INPUTS_PER_HOUR, window_seconds=3600)
anon_image_hour_limiter = SlidingWindowRateLimiter(FREE_IMAGES_PER_HOUR, window_seconds=3600)


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _safe_filename(name: str | None) -> str:
    base = (name or "uploaded_file").strip() or "uploaded_file"
    base = re.sub(r"[^\w.\- ]+", "_", base)
    return base[:MAX_FILENAME_LEN] or "uploaded_file"


def _is_blocked_upload(filename: str | None, content_type: str | None) -> bool:
    safe_name = _safe_filename(filename).lower()
    _, ext = os.path.splitext(safe_name)
    if ext in BLOCKED_UPLOAD_EXTENSIONS:
        return True
    ctype = (content_type or "").strip().lower()
    if ctype in {
        "application/x-msdownload",
        "application/x-msdos-program",
        "application/x-dosexec",
        "application/x-sh",
        "application/x-bat",
        "application/x-powershell",
    }:
        return True
    return False


def _sanitize_chat_message(raw: str) -> str:
    text = (raw or "").strip()
    text = CONTROL_CHAR_PATTERN.sub("", text)
    if not text:
        return ""
    return text[:MAX_MESSAGE_CHARS]


def _sanitize_extracted_text(text: str) -> str:
    cleaned = (text or "").replace("\x00", "").strip()
    if not cleaned:
        return "No readable text could be extracted from this file."
    return cleaned[:6000]


def _extract_image_prompt(raw_message: str) -> str | None:
    text = raw_message.strip()
    lower = text.lower()

    if lower.startswith("/image ") or lower.startswith("/img "):
        return text.split(" ", 1)[1].strip()

    direct_prefixes = [
        "create an image of ",
        "create a picture of ",
        "generate an image of ",
        "make an image of ",
        "make me an image of ",
        "make me a picture of ",
        "draw me ",
        "draw ",
    ]
    for prefix in direct_prefixes:
        if lower.startswith(prefix):
            return text[len(prefix):].strip()

    # Flexible natural-language image requests.
    patterns = [
        # can you make me a picture of a dragon
        re.compile(
            r"(?:can you|could you|please|pls)?\s*(?:create|generate|make|draw|design)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo|art|wallpaper|logo)\s*(?:of\s+)?(.+)",
            re.IGNORECASE,
        ),
        # i want an image of a dragon
        re.compile(
            r"(?:i want|i need)\s+(?:an?\s+)?(?:image|picture|photo|art|wallpaper|logo)\s*(?:of\s+)?(.+)",
            re.IGNORECASE,
        ),
        # show me an image of a dragon
        re.compile(
            r"(?:show|give)\s+me\s+(?:an?\s+)?(?:image|picture|photo)\s*(?:of\s+)?(.+)",
            re.IGNORECASE,
        ),
        # make a dragon image / create dragon art
        re.compile(
            r"(?:create|generate|make|draw|design)\s+(.+?)\s+(?:image|picture|photo|art|wallpaper|logo)\b",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            prompt = match.group(1).strip(" .,!?:;")
            if prompt:
                return prompt

    # Final broad fallback: if user mentions an image/picture/photo with "of ...",
    # treat it as an image request.
    fallback = re.search(r"(?:image|picture|photo|art|wallpaper|logo)\s+of\s+(.+)", text, re.IGNORECASE)
    if fallback:
        prompt = fallback.group(1).strip(" .,!?:;")
        if prompt:
            return prompt

    # Also catch common typo "pciture".
    typo_fallback = re.search(r"(?:pciture)\s+of\s+(.+)", text, re.IGNORECASE)
    if typo_fallback:
        prompt = typo_fallback.group(1).strip(" .,!?:;")
        if prompt:
            return prompt

    return None


class CreateConversationResponse(BaseModel):
    conversation_id: str


SUPPORTED_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = None
    use_web_search: bool = True
    user_timezone: str | None = None
    user_utc_offset_minutes: int | None = None
    user_location_label: str | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    retrieved_sources: list[dict]
    web_results: list[dict]
    generated_images: list[str] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: list[dict]


class ConversationListResponse(BaseModel):
    conversations: list[dict]


class LiveDataResponse(BaseModel):
    items: list[dict]


class UploadResponse(BaseModel):
    conversation_id: str
    file_id: int
    filename: str
    media_type: str
    extracted_preview: str


class AuthRequest(BaseModel):
    email: str = Field(min_length=5, max_length=160)
    password: str = Field(min_length=8, max_length=128)
    creator_bootstrap_secret: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: dict


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=5, max_length=160)


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    new_password: str = Field(min_length=8, max_length=128)


class ResetPasswordByEmailRequest(BaseModel):
    email: str = Field(min_length=5, max_length=160)
    new_password: str = Field(min_length=8, max_length=128)


class MeResponse(BaseModel):
    user: dict
    usage_last_hour: int
    free_inputs_per_hour: int
    limits: dict


class PlanUpdateRequest(BaseModel):
    email: str = Field(min_length=5, max_length=160)
    plan: str = Field(min_length=4, max_length=20)


class BillingCheckoutRequest(BaseModel):
    plan: str = Field(min_length=4, max_length=20)


class BillingCheckoutResponse(BaseModel):
    checkout_url: str


class BillingPortalResponse(BaseModel):
    portal_url: str


class MemoryForgetRequest(BaseModel):
    keys: list[str] = Field(default_factory=list)


def _build_uploaded_file_context(uploaded_files: list[dict]) -> str:
    if not uploaded_files:
        return ""
    lines = ["Uploaded file context (use this if relevant):"]
    for file_row in uploaded_files[:8]:
        lines.append(
            f"- {file_row['filename']} [{file_row['media_type']}]: {file_row['extracted_text'][:700]}"
        )
    return "\n".join(lines) + "\n\n"


def _token_from_request(request: Request) -> str | None:
    auth = (request.headers.get("Authorization") or "").strip()
    if not auth.lower().startswith("bearer "):
        cookie_token = (request.cookies.get(SESSION_COOKIE_NAME) or "").strip()
        return cookie_token or None
    token = auth[7:].strip()
    return token or None


def _current_user(request: Request) -> dict | None:
    token = _token_from_request(request)
    if not token:
        return None
    user = storage.get_user_by_token(token, max_age_days=SESSION_ABSOLUTE_MAX_DAYS)
    if user and SESSION_SLIDING_RENEWAL:
        storage.touch_session(token, ttl_days=SESSION_TTL_DAYS)
    return user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=SESSION_TTL_DAYS * 24 * 3600,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def _effective_plan(user: dict | None) -> str:
    if not user:
        return "free"
    email = str(user.get("email", "")).strip().lower()
    creator_email = (os.getenv("CREATOR_EMAIL") or "").strip().lower()
    if creator_email and email == creator_email:
        return "creator"
    plan = str(user.get("plan", "free")).strip().lower() or "free"
    return plan if plan in PLAN_CONFIG else "free"


def _plan_limits(plan: str) -> dict:
    return PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])


def _require_stripe_config() -> None:
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Billing is not configured.")


def _price_id_for_plan(plan: str) -> str:
    price_id = STRIPE_PRICE_IDS.get(plan, "")
    if not price_id:
        raise HTTPException(status_code=400, detail=f"No Stripe price configured for plan '{plan}'.")
    return price_id


def _plan_from_price_id(price_id: str) -> str:
    for plan, configured in STRIPE_PRICE_IDS.items():
        if configured and configured == price_id:
            return plan
    return "free"


def _conversation_access_check(conversation_id: str, user: dict | None) -> None:
    if not storage.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    owner = storage.conversation_owner(conversation_id)
    if owner is None and user is None:
        return
    if owner is None and user is not None:
        raise HTTPException(status_code=403, detail="Forbidden for this conversation.")
    if user is None or int(user["id"]) != int(owner):
        raise HTTPException(status_code=403, detail="Forbidden for this conversation.")


def _merge_profiles(primary: dict, secondary: dict) -> dict:
    out = dict(primary or {})
    other = dict(secondary or {})
    if other.get("name"):
        out["name"] = other["name"]
    facts = []
    for src in (out.get("facts", []), other.get("facts", [])):
        for f in src if isinstance(src, list) else []:
            if f not in facts:
                facts.append(f)
    if facts:
        out["facts"] = facts[-40:]
    if other.get("style"):
        out["style"] = other["style"]
    return out


def _extract_uploaded_content(file: UploadFile, payload: bytes) -> str:
    detected_type = file.content_type or ""
    if (not detected_type) or detected_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(file.filename or "")
        if guessed:
            detected_type = guessed
    media_type = (detected_type or "application/octet-stream").lower()
    if len(payload) > MAX_UPLOAD_BYTES:
        return "File is too large to process safely."

    if media_type.startswith("text/") or media_type in {
        "application/json",
        "application/xml",
        "text/csv",
    }:
        if len(payload) > MAX_TEXT_UPLOAD_BYTES:
            return "Text file is too large to process safely. Limit is 2MB for text formats."
        text = payload.decode("utf-8", errors="ignore").strip()
        if not text:
            return "File uploaded but text content was empty."
        return _sanitize_extracted_text(text)

    if media_type == "application/pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            return "PDF uploaded, but PDF parser dependency is missing."
        try:
            reader = PdfReader(BytesIO(payload))
            page_text = []
            for page in reader.pages:
                page_text.append((page.extract_text() or "").strip())
            text = "\n".join(t for t in page_text if t).strip()
            if text:
                return _sanitize_extracted_text(text)
            return "PDF uploaded, but no extractable text was found."
        except Exception:
            return "PDF uploaded, but text extraction failed."

    if media_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }:
        try:
            from docx import Document
        except Exception:
            return "Word document uploaded, but DOCX parser dependency is missing."
        try:
            doc = Document(BytesIO(payload))
            text = "\n".join((p.text or "").strip() for p in doc.paragraphs).strip()
            if text:
                return _sanitize_extracted_text(text)
            return "Word document uploaded, but no extractable text was found."
        except Exception:
            return "Word document uploaded, but text extraction failed."

    if media_type.startswith("image/"):
        if len(payload) > MAX_IMAGE_UPLOAD_BYTES:
            return "Image is too large to analyze safely. Limit is 8MB for image analysis."
        b64 = base64.b64encode(payload).decode("ascii")
        image_url = f"data:{media_type};base64,{b64}"
        analysis = llm.analyze_image(image_url)
        return _sanitize_extracted_text(analysis)

    return "File uploaded. Binary format is stored as attachment context but text extraction is limited for this type."


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = uuid4().hex[:12]
    start = time.perf_counter()
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(self), microphone=(), camera=()"
    response.headers["Cache-Control"] = "no-store"
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "n/a")
    logger.warning("validation_error request_id=%s errors=%s", request_id, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request payload.", "request_id": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "n/a")
    logger.exception("unhandled_error request_id=%s", request_id)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Please try again.",
            "request_id": request_id,
        },
    )


@app.on_event("startup")
def startup_live_updates() -> None:
    global embedded_scheduler
    enabled = os.getenv("LIVE_UPDATE_ENABLED", "true").strip().lower() == "true"
    run_in_api = os.getenv("RUN_UPDATER_IN_API", "true").strip().lower() == "true"
    if not (enabled and run_in_api):
        return

    interval = int(os.getenv("LIVE_UPDATE_INTERVAL_SECONDS", "300"))
    logger.info("embedded_updater_start interval_seconds=%s", interval)

    embedded_scheduler = BackgroundScheduler()
    embedded_scheduler.add_job(
        run_once,
        "interval",
        seconds=interval,
        args=[live_store],
        max_instances=1,
        next_run_time=datetime.now(),
    )
    embedded_scheduler.start()


@app.on_event("shutdown")
def shutdown_live_updates() -> None:
    global embedded_scheduler
    if embedded_scheduler:
        embedded_scheduler.shutdown(wait=False)
        embedded_scheduler = None


@app.get("/health")
def health() -> dict:
    live_updates_enabled = os.getenv("LIVE_UPDATE_ENABLED", "true").strip().lower() == "true"
    run_updater_in_api = os.getenv("RUN_UPDATER_IN_API", "true").strip().lower() == "true"
    return {
        "status": "ok",
        "knowledge_chunks": len(chunks),
        "web_search_provider": WEB_SEARCH_PROVIDER,
        "live_updates_enabled": live_updates_enabled,
        "run_updater_in_api": run_updater_in_api,
    }


@app.get("/")
def web_home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/privacy")
def privacy_page() -> FileResponse:
    return FileResponse(WEB_DIR / "privacy.html")


@app.get("/terms")
def terms_page() -> FileResponse:
    return FileResponse(WEB_DIR / "terms.html")


@app.post("/auth/register", response_model=AuthResponse)
def register(body: AuthRequest, request: Request, response: Response) -> AuthResponse:
    if not auth_rate_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="Too many auth attempts. Try again soon.")
    email_key = (body.email or "").strip().lower()
    if email_key and not auth_email_rate_limiter.allow(f"register:{email_key}"):
        raise HTTPException(status_code=429, detail="Too many attempts for this account/email.")
    creator_email = (os.getenv("CREATOR_EMAIL") or "").strip().lower()
    incoming_email = (body.email or "").strip().lower()
    if creator_email and incoming_email == creator_email:
        if not CREATOR_BOOTSTRAP_SECRET:
            raise HTTPException(status_code=503, detail="Creator bootstrap is not configured.")
        submitted = (body.creator_bootstrap_secret or "").strip()
        if submitted != CREATOR_BOOTSTRAP_SECRET:
            raise HTTPException(status_code=403, detail="Creator bootstrap secret is required.")
    try:
        user = storage.create_user(email=body.email, password=body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if creator_email and str(user.get("email", "")).strip().lower() == creator_email:
        storage.set_user_plan_by_id(int(user["id"]), "creator")
        user["plan"] = "creator"
    token = storage.create_session(user_id=int(user["id"]), ttl_days=SESSION_TTL_DAYS)
    _set_session_cookie(response, token)
    return AuthResponse(
        token=token,
        user={
            "id": user["id"],
            "email": user["email"],
            "plan": _effective_plan(user),
        },
    )


@app.post("/auth/login", response_model=AuthResponse)
def login(body: AuthRequest, request: Request, response: Response) -> AuthResponse:
    if not auth_rate_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="Too many auth attempts. Try again soon.")
    email_key = (body.email or "").strip().lower()
    if email_key and not auth_email_rate_limiter.allow(f"login:{email_key}"):
        raise HTTPException(status_code=429, detail="Too many attempts for this account/email.")
    user = storage.authenticate_user(email=body.email, password=body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = storage.create_session(user_id=int(user["id"]), ttl_days=SESSION_TTL_DAYS)
    _set_session_cookie(response, token)
    return AuthResponse(
        token=token,
        user={
            "id": user["id"],
            "email": user["email"],
            "plan": _effective_plan(user),
        },
    )


@app.post("/auth/change_password")
def change_password(body: ChangePasswordRequest, request: Request, response: Response) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    if body.current_password == body.new_password:
        raise HTTPException(status_code=400, detail="New password must be different from current password.")
    changed = storage.change_password(
        user_id=int(user["id"]),
        current_password=body.current_password,
        new_password=body.new_password,
    )
    if not changed:
        raise HTTPException(status_code=400, detail="Current password is incorrect or password update failed.")

    token = _token_from_request(request)
    if token:
        storage.delete_session(token)
    _clear_session_cookie(response)
    return {"status": "ok", "detail": "Password changed. Please log in again."}


@app.post("/auth/forgot_password")
def forgot_password(body: ForgotPasswordRequest, request: Request) -> dict:
    if not auth_rate_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="Too many auth attempts. Try again soon.")
    email_key = (body.email or "").strip().lower()
    if email_key and not auth_email_rate_limiter.allow(f"forgot:{email_key}"):
        raise HTTPException(status_code=429, detail="Too many attempts for this account/email.")
    token = storage.create_password_reset_token(
        email=body.email,
        ttl_minutes=PASSWORD_RESET_TOKEN_TTL_MINUTES,
    )
    if PASSWORD_RESET_DEV_MODE and token:
        return {
            "status": "ok",
            "detail": "Password reset token generated (dev mode).",
            "reset_token": token,
            "expires_in_minutes": PASSWORD_RESET_TOKEN_TTL_MINUTES,
        }
    return {
        "status": "ok",
        "detail": "If the account exists, a password reset instruction has been issued.",
    }


@app.post("/auth/reset_password")
def reset_password(body: ResetPasswordRequest) -> dict:
    changed = storage.reset_password_with_token(token=body.token, new_password=body.new_password)
    if not changed:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
    return {"status": "ok", "detail": "Password reset successful. Please log in."}


@app.post("/auth/reset_password_email")
def reset_password_email(body: ResetPasswordByEmailRequest, request: Request) -> dict:
    # Security guardrail: direct email-based password reset is only for controlled dev/testing.
    if not PASSWORD_RESET_DEV_MODE:
        raise HTTPException(
            status_code=403,
            detail="Email-based direct password reset is disabled. Use forgot/reset token flow.",
        )
    if not auth_rate_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="Too many auth attempts. Try again soon.")
    email_key = (body.email or "").strip().lower()
    if email_key and not auth_email_rate_limiter.allow(f"reset_email:{email_key}"):
        raise HTTPException(status_code=429, detail="Too many attempts for this account/email.")
    changed = storage.reset_password_by_email(email=body.email, new_password=body.new_password)
    if not changed:
        raise HTTPException(status_code=400, detail="Password reset failed for this email.")
    return {"status": "ok", "detail": "Password reset successful. Please log in."}


@app.post("/auth/logout")
def logout(request: Request, response: Response) -> dict:
    token = _token_from_request(request)
    if token:
        storage.delete_session(token)
    _clear_session_cookie(response)
    return {"status": "ok"}


@app.post("/auth/logout_all")
def logout_all(request: Request, response: Response) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    count = storage.delete_sessions_for_user(int(user["id"]))
    _clear_session_cookie(response)
    return {"status": "ok", "revoked_sessions": count}


@app.get("/auth/me", response_model=MeResponse)
def me(request: Request) -> MeResponse:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    plan = _effective_plan(user)
    limits = _plan_limits(plan)
    usage = storage.usage_count_last_hour(int(user["id"]))
    return MeResponse(
        user={
            "id": user["id"],
            "email": user["email"],
            "plan": plan,
        },
        usage_last_hour=usage,
        free_inputs_per_hour=FREE_INPUTS_PER_HOUR,
        limits=limits,
    )


@app.get("/plans")
def plans() -> dict:
    return {"plans": PLAN_CONFIG}


@app.get("/memory")
def get_memory(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    memory = storage.get_user_memory(int(user["id"]))
    return {"memory": memory}


@app.post("/memory/forget")
def forget_memory(body: MemoryForgetRequest, request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    memory = storage.get_user_memory(int(user["id"]))
    if not body.keys:
        memory = {}
    else:
        for key in body.keys:
            memory.pop((key or "").strip(), None)
    storage.upsert_user_memory(int(user["id"]), memory)
    return {"status": "ok", "memory": memory}


@app.post("/billing/checkout", response_model=BillingCheckoutResponse)
def billing_checkout(body: BillingCheckoutRequest, request: Request) -> BillingCheckoutResponse:
    _require_stripe_config()
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    target_plan = body.plan.strip().lower()
    if target_plan not in {"pro5", "pro10"}:
        raise HTTPException(status_code=400, detail="Invalid paid plan. Use pro5 or pro10.")

    price_id = _price_id_for_plan(target_plan)
    customer_id = (user.get("stripe_customer_id") or "").strip()
    if not customer_id:
        try:
            customer = stripe.Customer.create(email=user["email"], metadata={"user_id": str(user["id"])})
            customer_id = customer.id
            storage.set_user_billing_ids(int(user["id"]), stripe_customer_id=customer_id)
        except Exception as exc:
            logger.exception("stripe_customer_create_failed")
            raise HTTPException(status_code=503, detail="Could not initialize billing customer.") from exc

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{APP_BASE_URL}/?billing=success",
            cancel_url=f"{APP_BASE_URL}/?billing=cancel",
            metadata={"user_id": str(user["id"]), "target_plan": target_plan},
        )
    except Exception as exc:
        logger.exception("stripe_checkout_create_failed")
        raise HTTPException(status_code=503, detail="Could not create checkout session.") from exc

    return BillingCheckoutResponse(checkout_url=session.url)


@app.post("/billing/portal", response_model=BillingPortalResponse)
def billing_portal(request: Request) -> BillingPortalResponse:
    _require_stripe_config()
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    customer_id = (user.get("stripe_customer_id") or "").strip()
    if not customer_id:
        try:
            customer = stripe.Customer.create(email=user["email"], metadata={"user_id": str(user["id"])})
            customer_id = customer.id
            storage.set_user_billing_ids(int(user["id"]), stripe_customer_id=customer_id)
        except Exception as exc:
            logger.exception("stripe_customer_create_failed_for_portal")
            raise HTTPException(status_code=503, detail="Could not initialize billing customer.") from exc

    try:
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{APP_BASE_URL}/?billing=portal_return",
        )
    except Exception as exc:
        logger.exception("stripe_billing_portal_create_failed")
        raise HTTPException(status_code=503, detail="Could not create billing portal session.") from exc

    return BillingPortalResponse(portal_url=portal.url)


@app.post("/billing/webhook")
async def billing_webhook(request: Request) -> dict:
    _require_stripe_config()
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Stripe webhook secret not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=signature, secret=STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook signature.")

    event_type = event.get("type", "")
    data_obj = (event.get("data") or {}).get("object") or {}

    if event_type == "checkout.session.completed":
        metadata = data_obj.get("metadata") or {}
        user_id = metadata.get("user_id")
        customer = data_obj.get("customer")
        subscription = data_obj.get("subscription")
        if user_id:
            try:
                storage.set_user_billing_ids(
                    int(user_id),
                    stripe_customer_id=str(customer) if customer else None,
                    stripe_subscription_id=str(subscription) if subscription else None,
                )
            except Exception:
                logger.exception("billing_webhook_checkout_sync_failed")

    if event_type in {"customer.subscription.created", "customer.subscription.updated"}:
        customer = str(data_obj.get("customer") or "")
        subscription_id = str(data_obj.get("id") or "")
        status = str(data_obj.get("status") or "")
        items = ((data_obj.get("items") or {}).get("data") or [])
        price_id = ""
        if items:
            price_id = str(((items[0].get("price") or {}).get("id")) or "")

        plan = _plan_from_price_id(price_id)
        if status in {"active", "trialing"} and plan in {"pro5", "pro10"}:
            updated = storage.set_user_plan_by_stripe_customer(customer, plan)
            if not updated:
                logger.warning("billing_webhook_no_user_for_customer customer=%s", customer)
        elif status in {"canceled", "unpaid", "incomplete_expired"}:
            storage.set_user_plan_by_stripe_customer(customer, "free")

        # Best-effort subscription id sync for linked customer.
        if customer and subscription_id:
            try:
                storage.set_subscription_by_stripe_customer(customer, subscription_id)
            except Exception:
                logger.exception("billing_webhook_subscription_sync_failed")

    return {"received": True}


@app.post("/admin/users/plan")
def admin_set_user_plan(body: PlanUpdateRequest, request: Request) -> dict:
    user = _current_user(request)
    if not user or _effective_plan(user) != "creator":
        raise HTTPException(status_code=403, detail="Creator access required.")
    plan = body.plan.strip().lower()
    if plan not in PLAN_CONFIG:
        raise HTTPException(status_code=400, detail="Invalid plan. Use free, pro5, pro10, or creator.")
    updated = storage.set_user_plan_by_email(body.email, plan)
    if not updated:
        raise HTTPException(status_code=404, detail="User email not found.")
    return {"status": "ok", "email": body.email.strip().lower(), "plan": plan}


@app.get("/admin/stats")
def admin_stats(request: Request) -> dict:
    user = _current_user(request)
    if not user or _effective_plan(user) != "creator":
        raise HTTPException(status_code=403, detail="Creator access required.")
    return {"stats": storage.admin_stats()}


@app.post("/conversations", response_model=CreateConversationResponse)
def create_conversation(request: Request) -> CreateConversationResponse:
    user = _current_user(request)
    conversation_id = storage.create_conversation(user_id=int(user["id"]) if user else None)
    return CreateConversationResponse(conversation_id=conversation_id)


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
) -> UploadResponse:
    user = _current_user(request)
    if not upload_rate_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="Too many uploads. Please slow down and try again.")
    if _is_blocked_upload(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="This file type is not allowed for upload.")

    if conversation_id is None:
        conversation_id = storage.create_conversation(user_id=int(user["id"]) if user else None)
    else:
        _conversation_access_check(conversation_id, user)

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large. Max 10MB.")

    try:
        extracted = _extract_uploaded_content(file, payload)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key.") from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="OpenAI quota/rate limit reached.") from exc
    except APIConnectionError as exc:
        raise HTTPException(status_code=503, detail="Could not reach OpenAI API.") from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=f"OpenAI file analysis error: {exc.message}") from exc

    file_id = storage.add_uploaded_file(
        conversation_id=conversation_id,
        filename=_safe_filename(file.filename),
        media_type=file.content_type or "application/octet-stream",
        extracted_text=extracted,
    )
    return UploadResponse(
        conversation_id=conversation_id,
        file_id=file_id,
        filename=_safe_filename(file.filename),
        media_type=file.content_type or "application/octet-stream",
        extracted_preview=extracted[:240],
    )


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: str, request: Request) -> ConversationResponse:
    user = _current_user(request)
    _conversation_access_check(conversation_id, user)
    messages = storage.get_messages(conversation_id)
    return ConversationResponse(conversation_id=conversation_id, messages=messages)


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict:
    user = _current_user(request)
    if not storage.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    owner = storage.conversation_owner(conversation_id)
    if owner is not None and (user is None or int(user["id"]) != int(owner)):
        raise HTTPException(status_code=403, detail="Forbidden.")
    ok = storage.delete_conversation(conversation_id, user_id=int(user["id"]) if user else None)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found or already deleted.")
    return {"deleted": True, "conversation_id": conversation_id}


@app.get("/conversations", response_model=ConversationListResponse)
def list_conversations(request: Request, limit: int = 30) -> ConversationListResponse:
    user = _current_user(request)
    if user is None:
        return ConversationListResponse(conversations=[])
    conversations = storage.list_conversations(limit=limit, user_id=int(user["id"]))
    return ConversationListResponse(conversations=conversations)


@app.get("/live/stocks", response_model=LiveDataResponse)
def get_live_stocks(limit: int = 20) -> LiveDataResponse:
    return LiveDataResponse(items=live_store.get_latest("stock", limit=limit))


@app.get("/live/news", response_model=LiveDataResponse)
def get_live_news(limit: int = 20) -> LiveDataResponse:
    return LiveDataResponse(items=live_store.get_latest("news", limit=limit))


def _generate_title(user_msg: str, assistant_msg: str) -> str:
    try:
        msgs = [
            {"role": "system", "content": "Generate a short 4-6 word title for this conversation. Reply with ONLY the title, no punctuation or quotes."},
            {"role": "user", "content": user_msg[:300]},
            {"role": "assistant", "content": assistant_msg[:300]},
            {"role": "user", "content": "Title:"},
        ]
        return llm.chat(messages=msgs, max_tokens=20).strip().strip('"').strip("'")[:80]
    except Exception:
        return ""


@app.get("/models")
def list_models() -> dict:
    return {"models": sorted(SUPPORTED_MODELS), "default": DEFAULT_MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
    user = _current_user(request)
    client_key = _client_key(request)
    if not chat_rate_limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="Too many chat requests. Please wait a moment.")

    # Validate and resolve model selection.
    requested_model = (chat_request.model or "").strip()
    if requested_model and requested_model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{requested_model}'. Valid options: {sorted(SUPPORTED_MODELS)}")
    active_model = requested_model or DEFAULT_MODEL

    sanitized_message = _sanitize_chat_message(chat_request.message)
    if not sanitized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    conversation_id = chat_request.conversation_id
    if conversation_id is None:
        conversation_id = storage.create_conversation(user_id=int(user["id"]) if user else None)
    else:
        _conversation_access_check(conversation_id, user)

    if user:
        plan = _effective_plan(user)
        limits = _plan_limits(plan)
        usage = storage.usage_count_last_hour(int(user["id"]))
        input_limit = limits.get("inputs_per_hour")
        if input_limit is not None and usage >= int(input_limit):
            raise HTTPException(
                status_code=402,
                detail=f"{plan} plan limit reached ({input_limit} prompts/hour). Upgrade to continue.",
            )
        storage.record_usage_event(int(user["id"]), event_type="chat_input")
    else:
        if not anon_input_hour_limiter.allow(f"inputs:{client_key}"):
            raise HTTPException(
                status_code=402,
                detail=f"free plan limit reached ({FREE_INPUTS_PER_HOUR} prompts/hour). Sign in or upgrade to continue.",
            )

    history = storage.get_messages(conversation_id)
    conversation_profile = storage.get_user_profile(conversation_id)
    user_profile = storage.get_user_memory(int(user["id"])) if user else {}
    merged_profile = _merge_profiles(conversation_profile, user_profile)
    uploaded_files = storage.get_uploaded_files(conversation_id, limit=12)
    uploaded_file_context = _build_uploaded_file_context(uploaded_files)
    if user:
        _uk = storage.get_user_knowledge_content(int(user["id"]))
        if _uk:
            _kparts = ["[" + f["filename"] + "]" + " " + f["content"][:2000] for f in _uk[:5]]
            _kctx = "User's personal notes:" + " | ".join(_kparts)
            uploaded_file_context = (uploaded_file_context or "") + "\n\n" + _kctx
    raw_message = sanitized_message
    image_prompt = _extract_image_prompt(raw_message)

    if image_prompt is not None:
        prompt = image_prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="Provide an image prompt after /image.")
        if user:
            plan = _effective_plan(user)
            limits = _plan_limits(plan)
            image_limit = limits.get("images_per_hour")
            image_usage = storage.usage_count_last_hour(int(user["id"]), event_type="image_generation")
            if image_limit is not None and image_usage >= int(image_limit):
                raise HTTPException(
                    status_code=402,
                    detail=f"{plan} plan image limit reached ({image_limit}/hour). Upgrade to continue.",
                )
        else:
            if not anon_image_hour_limiter.allow(f"images:{client_key}"):
                raise HTTPException(
                    status_code=402,
                    detail=f"free plan image limit reached ({FREE_IMAGES_PER_HOUR}/hour). Sign in or upgrade to continue.",
                )
        try:
            image_data_url = llm.generate_image(prompt=prompt)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail="Invalid OpenAI API key.") from exc
        except RateLimitError as exc:
            raise HTTPException(status_code=429, detail="OpenAI quota/rate limit reached.") from exc
        except APIConnectionError as exc:
            raise HTTPException(status_code=503, detail="Could not reach OpenAI API.") from exc
        except BadRequestError as exc:
            raise HTTPException(status_code=400, detail=f"OpenAI image request error: {exc.message}") from exc

        answer = f"Generated image for: {prompt}"
        if user:
            storage.record_usage_event(int(user["id"]), event_type="image_generation")
        storage.add_message(conversation_id, "user", sanitized_message)
        storage.add_message(conversation_id, "assistant", answer)
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            retrieved_sources=[],
            web_results=[],
            generated_images=[image_data_url],
        )

    try:
        answer, hits, web_results, updated_profile = chat_service.ask(
            question=sanitized_message,
            history=history,
            use_web_search=chat_request.use_web_search,
            user_timezone=chat_request.user_timezone,
            user_utc_offset_minutes=chat_request.user_utc_offset_minutes,
            user_location_label=chat_request.user_location_label,
            user_profile=merged_profile,
            uploaded_file_context=uploaded_file_context,
            model=active_model,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key.") from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="OpenAI quota/rate limit reached.") from exc
    except APIConnectionError as exc:
        raise HTTPException(status_code=503, detail="Could not reach OpenAI API.") from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=f"OpenAI request error: {exc.message}") from exc
    except Exception as exc:
        logger.exception("chat_unexpected_failure error=%s", str(exc))
        raise HTTPException(status_code=503, detail="Temporary server issue. Please try again.") from exc

    storage.add_message(conversation_id, "user", sanitized_message)
    storage.add_message(conversation_id, "assistant", answer)
    storage.upsert_user_profile(conversation_id, updated_profile)
    if user:
        storage.upsert_user_memory(int(user["id"]), updated_profile)
    _msg_count = len(storage.get_messages(conversation_id, limit=3))
    if _msg_count == 2:
        _title = _generate_title(sanitized_message, answer)
        if _title:
            storage.set_conversation_title(conversation_id, _title)

    source_rows = [
        {"source": source, "score": round(score, 3), "preview": text[:180]}
        for score, source, text in hits
    ]
    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        retrieved_sources=source_rows,
        web_results=web_results,
        generated_images=[],
    )


def _stream_text_tokens(text: str):
    for token in re.findall(r"\S+\s*", text or ""):
        yield token


@app.post("/chat/stream")
def chat_stream(chat_request: ChatRequest, request: Request):
    user = _current_user(request)
    client_key = _client_key(request)
    if not chat_rate_limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="Too many chat requests. Please wait a moment.")
    requested_model = (chat_request.model or "").strip()
    if requested_model and requested_model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{requested_model}'.")
    active_model = requested_model or DEFAULT_MODEL
    sanitized_message = _sanitize_chat_message(chat_request.message)
    if not sanitized_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    conversation_id = chat_request.conversation_id
    if conversation_id is None:
        conversation_id = storage.create_conversation(user_id=int(user["id"]) if user else None)
    else:
        _conversation_access_check(conversation_id, user)
    if user:
        plan = _effective_plan(user)
        limits = _plan_limits(plan)
        usage = storage.usage_count_last_hour(int(user["id"]))
        input_limit = limits.get("inputs_per_hour")
        if input_limit is not None and usage >= int(input_limit):
            raise HTTPException(
                status_code=402,
                detail=f"{plan} plan limit reached ({input_limit} prompts/hour). Upgrade to continue.",
            )
        storage.record_usage_event(int(user["id"]), event_type="chat_input")
    else:
        if not anon_input_hour_limiter.allow(f"inputs:{client_key}"):
            raise HTTPException(
                status_code=402,
                detail=f"free plan limit reached ({FREE_INPUTS_PER_HOUR} prompts/hour). Sign in or upgrade to continue.",
            )
    history = storage.get_messages(conversation_id)
    conversation_profile = storage.get_user_profile(conversation_id)
    user_profile = storage.get_user_memory(int(user["id"])) if user else {}
    merged_profile = _merge_profiles(conversation_profile, user_profile)
    uploaded_files = storage.get_uploaded_files(conversation_id, limit=12)
    uploaded_file_context = _build_uploaded_file_context(uploaded_files)
    if user:
        _uk = storage.get_user_knowledge_content(int(user["id"]))
        if _uk:
            _kparts = ["[" + f["filename"] + "]" + " " + f["content"][:2000] for f in _uk[:5]]
            _kctx = "User's personal notes:" + " | ".join(_kparts)
            uploaded_file_context = (uploaded_file_context or "") + "\n\n" + _kctx
    image_prompt = _extract_image_prompt(sanitized_message)
    if image_prompt is not None:
        result = chat(chat_request, request)
        def image_gen():
            yield f"event: meta\ndata: {json.dumps({'conversation_id': result.conversation_id})}\n\n"
            payload = {
                "conversation_id": result.conversation_id,
                "answer": result.answer,
                "retrieved_sources": result.retrieved_sources,
                "web_results": result.web_results,
                "generated_images": result.generated_images,
            }
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"
        return StreamingResponse(image_gen(), media_type="text/event-stream")
    def event_gen():
        yield f"event: meta\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
        final_answer = ""
        final_hits = []
        final_web_results = []
        final_profile = {}
        try:
            for event_type, event_data in chat_service.ask_stream(
                question=sanitized_message,
                history=history,
                use_web_search=chat_request.use_web_search,
                user_timezone=chat_request.user_timezone,
                user_utc_offset_minutes=chat_request.user_utc_offset_minutes,
                user_location_label=chat_request.user_location_label,
                user_profile=merged_profile,
                uploaded_file_context=uploaded_file_context,
                model=active_model,
            ):
                if event_type == "token":
                    yield f"event: token\ndata: {json.dumps({'token': event_data})}\n\n"
                elif event_type == "replace":
                    yield f"event: replace\ndata: {json.dumps({'answer': event_data})}\n\n"
                elif event_type == "done":
                    final_answer = event_data["answer"]
                    final_hits = event_data["hits"]
                    final_web_results = event_data["web_results"]
                    final_profile = event_data["updated_profile"]
        except Exception as exc:
            logger.exception("chat_stream_failure error=%s", str(exc))
            yield f"event: error\ndata: {json.dumps({'detail': 'Temporary server issue. Please try again.'})}\n\n"
            return
        storage.add_message(conversation_id, "user", sanitized_message)
        storage.add_message(conversation_id, "assistant", final_answer)
        storage.upsert_user_profile(conversation_id, final_profile)
        if user:
            storage.upsert_user_memory(int(user["id"]), final_profile)
        _msg_count = len(storage.get_messages(conversation_id, limit=3))
        if _msg_count == 2:
            _title = _generate_title(sanitized_message, final_answer)
            if _title:
                storage.set_conversation_title(conversation_id, _title)
        source_rows = [
            {"source": source, "score": round(score, 3), "preview": text[:180]}
            for score, source, text in final_hits
        ]
        payload = {
            "conversation_id": conversation_id,
            "answer": final_answer,
            "retrieved_sources": source_rows,
            "web_results": final_web_results,
            "generated_images": [],
        }
        yield f"event: done\ndata: {json.dumps(payload)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/knowledge")
async def upload_knowledge(request: Request, file: UploadFile = File(...)) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required to upload personal notes.")
    if _is_blocked_upload(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="This file type is not allowed.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="File is empty.")
    extracted = _extract_uploaded_content(file, payload)
    file_id = storage.add_user_knowledge_file(int(user["id"]), _safe_filename(file.filename), extracted)
    return {"id": file_id, "filename": _safe_filename(file.filename)}


@app.get("/knowledge")
def list_knowledge(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required.")
    files = storage.list_user_knowledge_files(int(user["id"]))
    return {"files": files}


@app.delete("/knowledge/{file_id}")
def delete_knowledge(file_id: int, request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required.")
    ok = storage.delete_user_knowledge_file(file_id, int(user["id"]))
    if not ok:
        raise HTTPException(status_code=404, detail="File not found.")
    return {"deleted": True}
