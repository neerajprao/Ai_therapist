"""FastAPI application.

Run locally with:  uvicorn soulsync.api:app --port 5050
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import __version__, safety, tts
from .audio import AudioDecodeError, decode_audio
from .chat import ChatEngine
from .config import Settings, load_settings, setup_logging
from .emotion import EmotionDetector, build_detector
from .llm import LLM, build_llm
from .memory import MemoryStore, build_memory
from .stt import STT, STTError, build_stt

log = logging.getLogger(__name__)

USER_COOKIE = "soulsync_uid"
COOKIE_MAX_AGE = 365 * 24 * 3600
_SESSION_RE = re.compile(r"^[0-9a-f]{32}$")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str
    llm: str
    stt: str
    emotion_backend: str
    emotion_available: bool
    emotion2vec_enabled: bool
    memory_enabled: bool
    knowledge_enabled: bool
    tts_backend: str


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ForgetResponse(BaseModel):
    deleted_memories: int


class ErrorResponse(BaseModel):
    detail: str


def _sign(uid: str, secret: str) -> str:
    mac = hmac.new(secret.encode(), uid.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{uid}.{mac}"


def _verify(cookie: str | None, secret: str) -> str | None:
    if not cookie or "." not in cookie:
        return None
    uid, _, _ = cookie.partition(".")
    return uid if hmac.compare_digest(cookie, _sign(uid, secret)) else None


class RateLimiter:
    """Sliding one-minute window per user, in memory (single process)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLM | None = None,
    stt: STT | None = None,
    emotion: EmotionDetector | None = None,
    memory: MemoryStore | None = None,
    knowledge=None,
) -> FastAPI:
    """Build the app. Tests pass fakes for llm/stt/emotion/memory."""
    settings = settings or load_settings()
    setup_logging(settings.log_level)

    llm = llm or build_llm(settings)
    stt = stt or build_stt(settings)
    emotion = emotion or build_detector(settings)
    if memory is None:
        memory = build_memory(settings)
    if knowledge is None and settings.knowledge_enabled:
        from .knowledge import build_knowledge

        knowledge = build_knowledge(settings)
    engine = ChatEngine(llm, memory, max_turns=settings.history_max_turns, knowledge=knowledge)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app = FastAPI(title="SoulSync", version=__version__)
    app.state.settings = settings
    app.state.engine = engine

    log.info(
        "SoulSync %s | mode=%s llm=%s stt=%s emotion=%s (available=%s) memory=%s tts=%s",
        __version__, settings.mode, llm.name, stt.name, settings.emotion_backend,
        emotion.available, memory is not None, settings.tts_backend,
    )

    def current_user(request: Request) -> str | None:
        return _verify(request.cookies.get(USER_COOKIE), settings.secret_key)

    def require_user(request: Request) -> str:
        uid = current_user(request)
        if uid is None:
            raise HTTPException(401, "No user cookie. Reload the page and try again.")
        return uid

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        uid = current_user(request) or uuid.uuid4().hex
        response = TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "session_id": uuid.uuid4().hex,  # new on every page load
                "tts_backend": settings.tts_backend,
                "data_flow": settings.data_flow(),
                "memory_enabled": memory is not None,
                "knowledge_enabled": knowledge is not None,
                "helpline": f"{safety.HELPLINE_NAME} {safety.HELPLINE_NUMBERS}",
                "emergency": safety.EMERGENCY_NUMBER,
            },
        )
        # Refreshed on each visit so the same browser is recognised for a year.
        response.set_cookie(
            USER_COOKIE,
            _sign(uid, settings.secret_key),
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=settings.cookie_secure,
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            version=__version__,
            mode=settings.mode,
            llm=llm.name,
            stt=stt.name,
            emotion_backend=settings.emotion_backend,
            emotion_available=emotion.available,
            emotion2vec_enabled=emotion.secondary is not None,
            memory_enabled=memory is not None,
            knowledge_enabled=knowledge is not None,
            tts_backend=settings.tts_backend,
        )

    @app.post(
        "/process_audio_stream",
        responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    def process_audio_stream(
        request: Request,
        audio: UploadFile = File(...),
        session_id: str = Form(...),
    ):
        """Plain-text stream. First chunk is `METADATA|<transcript>|<emotion>|`,
        followed by the reply as it is generated."""
        user_id = require_user(request)
        if not limiter.allow(user_id):
            raise HTTPException(429, "You're sending messages very quickly. Please take a breath and try again in a minute.")
        if not _SESSION_RE.match(session_id):
            raise HTTPException(400, "Invalid session_id.")
        raw = audio.file.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(413, f"Recording is larger than {settings.max_upload_mb} MB.")
        try:
            wave = decode_audio(raw)
        except AudioDecodeError as exc:
            log.warning("Rejected upload: %s", exc)
            raise HTTPException(400, "Could not read that recording.") from exc

        detected = emotion.detect(wave)
        stt_error = None
        transcript = ""
        if detected.source != "silence":
            try:
                transcript = stt.transcribe(wave, raw, audio.filename or "audio.webm")
            except STTError as exc:
                stt_error = exc.user_message
        # '|' delimits the header, so it cannot appear inside the transcript.
        header = f"METADATA|{transcript.replace('|', ' ')}|{detected.display}|"

        def generate():
            yield header
            if stt_error:
                yield stt_error
                return
            yield from engine.respond(user_id, session_id, transcript, detected.display)

        return StreamingResponse(
            generate(),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/get_audio", responses={409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    def get_audio(body: TTSRequest, request: Request):
        """Server-side speech, only when TTS_BACKEND=say. Returns M4A bytes."""
        require_user(request)
        if settings.tts_backend != "say":
            raise HTTPException(409, "Server TTS is disabled; the browser speaks the reply.")
        try:
            data = tts.synthesize_say(body.text)
        except tts.TTSError as exc:
            raise HTTPException(500, str(exc)) from exc
        return Response(data, media_type="audio/mp4", headers={"Cache-Control": "no-store"})

    @app.delete("/memory", response_model=ForgetResponse)
    def forget_me(request: Request):
        """Delete everything stored for this browser's user."""
        user_id = require_user(request)
        engine.sessions.forget_user(user_id)
        deleted = memory.forget_user(user_id) if memory is not None else 0
        return ForgetResponse(deleted_memories=deleted)

    return app


def __getattr__(name: str):
    # `uvicorn soulsync.api:app` builds the app on first access, so importing
    # this module (e.g. in tests) loads no models.
    if name == "app":
        app = create_app()
        globals()["app"] = app
        return app
    raise AttributeError(name)
