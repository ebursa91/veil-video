"""A bounded, durable single-worker queue with cooperative cancellation."""

import fcntl
import logging
import shutil
import threading

from veil.config import Settings
from veil.store import Store

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store
        self.lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._cancel: dict[str, threading.Event] = {}
        self._thread: threading.Thread | None = None
        self._lease = None

    def start(self):
        self._lease = (self.settings.data_dir / "worker.lock").open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lease.close()
            raise RuntimeError("Another Veil process already uses this data directory.") from None
        for job in self.store.list():
            if job["status"] == "processing":
                self.store.update(
                    job["id"],
                    status="failed",
                    stage="failed",
                    error="Processing was interrupted by a server restart. Upload the video again.",
                    message="Interrupted by server restart",
                )
                self._clean_output(job["id"])
        uploads = self.settings.data_dir / ".uploads"
        if uploads.exists():
            shutil.rmtree(uploads)
        self._thread = threading.Thread(target=self._run, name="veil-video-worker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self.lock:
            for event in self._cancel.values():
                event.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                # Keep the lease while a worker still owns processes/files.
                log.error(
                    "Video worker did not stop in 30s; terminate this server before restarting."
                )
                return
        if self._lease:
            self._lease.close()

    def notify(self):
        self._wake.set()

    def active_count(self) -> int:
        return sum(job["status"] in {"queued", "processing"} for job in self.store.list())

    def cancel(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.store.get(job_id)
            if not job:
                return None
            if job["status"] == "queued":
                return self.store.update(
                    job_id, status="cancelled", stage="cancelled", message="Cancelled"
                )
            if job["status"] == "processing":
                self._cancel[job_id].set()
                return self.store.update(job_id, message="Cancelling…")
            return job

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.store.get(job_id)
            if not job:
                return False
            if job["status"] in {"queued", "processing"}:
                raise ValueError("Cancel the job and wait for it to stop before deleting it.")
            shutil.rmtree(self.store.directory(job_id), ignore_errors=True)
            self.store.delete(job_id)
            return True

    def _clean_output(self, job_id: str):
        directory = self.store.directory(job_id)
        if directory.exists():
            for file in directory.iterdir():
                if file.name != "source.bin" and file.is_file():
                    file.unlink(missing_ok=True)

    def _run(self):
        while not self._stop.is_set():
            with self.lock:
                queued = [job for job in reversed(self.store.list()) if job["status"] == "queued"]
                job = queued[0] if queued else None
                if job:
                    event = self._cancel[job["id"]] = threading.Event()
                    self.store.update(
                        job["id"], status="processing", stage="probing", message="Reading video"
                    )
            if not job:
                self._wake.wait(1)
                self._wake.clear()
                continue
            job_id = job["id"]
            directory = self.store.directory(job_id)
            try:
                from veil.media import probe
                from veil.pipeline import process_video

                media = probe(directory / "source.bin")
                self.store.update(job_id, media=media)

                def progress(fields: dict, current_job_id=job_id):
                    # A pipeline can report metrics, never modify queue identity/state.
                    allowed = {
                        "stage",
                        "progress",
                        "frames_processed",
                        "total_frames",
                        "fps",
                        "eta_seconds",
                        "backend",
                        "message",
                    }
                    self.store.update(
                        current_job_id, **{k: v for k, v in fields.items() if k in allowed}
                    )

                result = process_video(
                    directory / "source.bin",
                    directory / "blurred.mp4",
                    mode=job["mode"],
                    progress=progress,
                    cancelled=lambda current_event=event: (
                        current_event.is_set() or self._stop.is_set()
                    ),
                    model_path=self.settings.model_path,
                    effect=job.get("effect", "blur"),
                    expression_model_path=self.settings.expression_path,
                )
                with self.lock:
                    if event.is_set() or self._stop.is_set():
                        self._clean_output(job_id)
                        self.store.update(
                            job_id, status="cancelled", stage="cancelled", message="Cancelled"
                        )
                    else:
                        self.store.update(
                            job_id,
                            status="completed",
                            stage="completed",
                            progress=100,
                            message="Ready to review",
                            result=result,
                            eta_seconds=0,
                        )
            except Exception as exc:
                self._clean_output(job_id)
                if event.is_set() or self._stop.is_set():
                    self.store.update(
                        job_id, status="cancelled", stage="cancelled", message="Cancelled"
                    )
                else:
                    log.exception("Video processing failed for job %s", job_id)
                    # Avoid exposing absolute paths/subprocess command lines to the browser.
                    detail = str(exc).replace(str(self.settings.data_dir), "[video storage]")
                    detail = detail.replace(str(self.settings.model_path), "[face model]")
                    self.store.update(
                        job_id,
                        status="failed",
                        stage="failed",
                        message="Processing failed",
                        error=detail[:600] or "Processing failed. See the server log.",
                    )
            finally:
                with self.lock:
                    self._cancel.pop(job_id, None)
