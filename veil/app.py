"""HTTP interface. Originals stay private; only finished exports can be streamed."""

import asyncio
import json
import secrets
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from veil.auth import COOKIE, Auth
from veil.config import Settings
from veil.jobs import JobManager
from veil.store import Store

STATIC = Path(__file__).parent / "static"


class Login(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


class BodyTooLarge(Exception):
    pass


class RequestGuard:
    """Authenticate uploads before multipart parsing and bound even chunked request bodies."""

    def __init__(self, app, auth: Auth, settings: Settings):
        self.app, self.auth, self.settings = app, auth, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path, method = scope["path"], scope["method"]
        request = Request(scope)
        mutating = method not in {"GET", "HEAD", "OPTIONS"}
        if path.startswith("/api/") and path != "/api/login":
            session = self.auth.read(request.cookies.get(COOKIE))
            if not session:
                return await JSONResponse({"detail": "Please sign in."}, status_code=401)(
                    scope, receive, send
                )
            if mutating and not secrets.compare_digest(
                request.headers.get("x-csrf-token", ""), session["csrf_token"]
            ):
                return await JSONResponse({"detail": "Invalid CSRF token."}, status_code=403)(
                    scope, receive, send
                )
        if mutating:
            origin = request.headers.get("origin")
            expected_origin = self.settings.trusted_origin or str(request.base_url).rstrip("/")
            if origin and origin.rstrip("/") != expected_origin:
                return await JSONResponse({"detail": "Cross-origin request rejected."}, 403)(
                    scope, receive, send
                )
        maximum = (
            self.settings.max_upload_bytes + 64 * 1024
            if path == "/api/jobs" and method == "POST"
            else 8192
        )
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            length = maximum + 1
        if length > maximum or length < 0:
            return await JSONResponse({"detail": "Request is too large."}, 413)(
                scope, receive, send
            )
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > maximum:
                raise BodyTooLarge
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except BodyTooLarge:
            await JSONResponse({"detail": "Request is too large."}, 413)(scope, receive, send)


def create_app(settings: Settings | None = None, *, start_worker: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = Store(settings.data_dir)
    manager = JobManager(settings, store)
    auth = Auth(settings)
    upload_lock = asyncio.Lock()
    system_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        # Multipart parsing spools before the handler copies uploads. Put those
        # files on the media disk, not an often RAM-backed system /tmp.
        private_temp = settings.data_dir / ".tmp"
        private_temp.mkdir(exist_ok=True, mode=0o700)
        previous_temp = tempfile.tempdir
        if start_worker:
            manager.start()
        tempfile.tempdir = str(private_temp)
        try:
            yield
        finally:
            if start_worker:
                await asyncio.to_thread(manager.stop)
            tempfile.tempdir = previous_temp

    app = FastAPI(title="Veil", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.manager, app.state.auth = store, manager, auth
    app.add_middleware(RequestGuard, auth=auth, settings=settings)

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; "
                "style-src 'self'; "
                "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
                "object-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
                "form-action 'self'",
            }
        )
        return response

    def session(request: Request) -> dict:
        result = auth.read(request.cookies.get(COOKIE))
        if not result:
            raise HTTPException(401, "Please sign in.")
        return result

    def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Video not found.")
        return job

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.post("/api/login")
    async def login(payload: Login, request: Request):
        address = request.client.host if request.client else "unknown"
        if not auth.allow_attempt(address):
            raise HTTPException(429, "Too many attempts. Try again in five minutes.")
        if not await asyncio.to_thread(auth.verify_password, payload.password):
            raise HTTPException(401, "Incorrect password.")
        cookie, result = auth.issue()
        response = JSONResponse(result)
        response.set_cookie(
            COOKIE,
            cookie,
            max_age=settings.session_seconds,
            httponly=True,
            secure=settings.secure_cookie,
            samesite="strict",
            path="/",
        )
        return response

    @app.get("/api/session")
    def current_session(request: Request):
        return session(request)

    @app.post("/api/logout")
    def logout():
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(COOKIE, path="/", secure=settings.secure_cookie, samesite="strict")
        return response

    @app.get("/api/system")
    async def system_info():
        async with system_lock:
            from veil.pipeline import capabilities

            system = await asyncio.to_thread(capabilities, settings.model_path)
        return {
            **system,
            "max_upload_bytes": settings.max_upload_bytes,
            "model_ready": settings.model_path.is_file(),
            "default_mode": settings.mode,
        }

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": store.list()}

    @app.post("/api/jobs", status_code=201)
    async def upload(request: Request):
        if upload_lock.locked():
            raise HTTPException(409, "Another upload is in progress. Try again when it finishes.")
        async with upload_lock:
            if manager.active_count() >= settings.max_queue:
                raise HTTPException(409, "The queue is full. Wait for a video to finish.")
            if not settings.model_path.is_file():
                raise HTTPException(503, "Face model missing. Run 'veil init' on the server first.")
            # Parsing happens only after authentication, size and queue checks.
            async with request.form(max_files=1, max_fields=1, max_part_size=8192) as form:
                file = form.get("file")
                if not isinstance(file, UploadFile):
                    # Starlette constructs its base UploadFile class.
                    from starlette.datastructures import UploadFile as StarletteUploadFile

                    if not isinstance(file, StarletteUploadFile):
                        raise HTTPException(422, "Choose a video file.")
                mode = str(form.get("mode", settings.mode))
                if mode not in {"auto", "gpu", "cpu"}:
                    raise HTTPException(422, "Unknown processing mode.")
                name = (file.filename or "video").replace("\\", "/").split("/")[-1]
                name = "".join(c for c in name if c.isprintable())[:180] or "video"
                if Path(name).suffix.lower() not in {".mp4", ".mov", ".m4v", ".mkv", ".webm"}:
                    raise HTTPException(415, "Use a MOV, MP4, M4V, MKV, or WebM video.")
                job_id = uuid.uuid4().hex
                staging = settings.data_dir / ".uploads"
                staging.mkdir(exist_ok=True, mode=0o700)
                temporary = staging / f"{job_id}.part"
                directory = store.directory(job_id)
                size = 0
                try:
                    with temporary.open("xb") as target:
                        while chunk := await file.read(1024 * 1024):
                            size += len(chunk)
                            if size > settings.max_upload_bytes:
                                raise HTTPException(413, "Video exceeds the upload limit.")
                            if shutil.disk_usage(staging).free < len(chunk) + 256 * 1024**2:
                                raise HTTPException(507, "Not enough free disk space.")
                            await asyncio.to_thread(target.write, chunk)
                    if not size:
                        raise HTTPException(422, "The video is empty.")
                    directory.mkdir(mode=0o700)
                    temporary.replace(directory / "source.bin")
                    job = store.create(name, size, mode, job_id)
                    manager.notify()
                    return job
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    shutil.rmtree(directory, ignore_errors=True)
                    raise

    @app.get("/api/jobs/{job_id}")
    def job_info(job_id: str):
        return get_job(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        result = manager.cancel(job_id)
        if result is None:
            raise HTTPException(404, "Video not found.")
        return result

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str):
        try:
            deleted = manager.delete(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        if not deleted:
            raise HTTPException(404, "Video not found.")
        return {"deleted": True}

    @app.get("/api/jobs/{job_id}/download")
    @app.get("/api/jobs/{job_id}/preview")
    def download(job_id: str, request: Request):
        job = get_job(job_id)
        output = store.directory(job_id) / "blurred.mp4"
        if job["status"] != "completed" or not output.is_file():
            raise HTTPException(409, "The export is not ready.")
        filename = f"{Path(job['filename']).stem}-blurred.mp4"
        return FileResponse(
            output,
            media_type="video/mp4",
            filename=filename,
            content_disposition_type="inline"
            if request.url.path.endswith("preview")
            else "attachment",
        )

    @app.get("/api/events")
    async def events(request: Request):
        async def stream():
            previous = ""
            ticks = 0
            while not await request.is_disconnected():
                if auth.read(request.cookies.get(COOKIE)) is None:
                    yield "event: expired\ndata: {}\n\n"
                    return
                current = json.dumps({"jobs": await asyncio.to_thread(store.list)})
                if current != previous:
                    yield f"event: jobs\ndata: {current}\n\n"
                    previous = current
                elif ticks % 15 == 0:
                    yield ": heartbeat\n\n"
                ticks += 1
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    return app
