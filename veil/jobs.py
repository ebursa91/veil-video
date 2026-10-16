"""A bounded, durable single-worker queue with cooperative cancellation."""

import fcntl
import logging
import os
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from veil import retention
from veil.config import Settings
from veil.store import Store

log = logging.getLogger(__name__)


class MediaUnavailableError(ValueError):
    """The export has expired, was archived, or is otherwise unavailable."""


@dataclass(frozen=True, slots=True)
class PreviewSnapshot:
    jpeg: bytes
    frame: int
    time: float


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
        self._previews: dict[str, PreviewSnapshot] = {}
        self._uploads: dict[str, int] = {}
        self._media_leases: dict[str, int] = {}
        self._maintenance_wake = threading.Event()
        self._maintenance_thread: threading.Thread | None = None

    def start(self):
        self._lease = (self.settings.data_dir / "worker.lock").open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lease.close()
            raise RuntimeError("Another Veil process already uses this data directory.") from None
        try:
            with self.lock:
                for job in self.store.list():
                    if job["status"] == "processing":
                        self.store.update(
                            job["id"],
                            status="failed",
                            stage="failed",
                            error="Processing was interrupted by a server restart. "
                            "Upload the video again.",
                            message="Interrupted by server restart",
                            preview_frame=None,
                            preview_time=None,
                        )
                        self._clean_output(job["id"])
                self.collect_retention()
        except Exception:
            self._lease.close()
            raise
        self._thread = threading.Thread(target=self._run, name="veil-video-worker", daemon=True)
        self._thread.start()
        self._maintenance_thread = threading.Thread(
            target=self._maintain,
            name="veil-media-maintenance",
            daemon=True,
        )
        self._maintenance_thread.start()

    def stop(self):
        self._stop.set()
        with self.lock:
            for event in self._cancel.values():
                event.set()
        self._wake.set()
        self._maintenance_wake.set()
        if self._thread:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                # Keep the lease while a worker still owns processes/files.
                log.error(
                    "Video worker did not stop in 30s; terminate this server before restarting."
                )
                return
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=30)
            if self._maintenance_thread.is_alive():
                log.error("Media maintenance did not stop; keeping the data-directory lease.")
                return
        if self._lease:
            self._lease.close()

    def notify(self):
        self._wake.set()

    @contextmanager
    def track_upload(self, job_id: str):
        self.store.directory(job_id)  # Reject anything outside the UUID namespace.
        with self.lock:
            self._uploads[job_id] = self._uploads.get(job_id, 0) + 1
        try:
            yield
        finally:
            with self.lock:
                self._release(self._uploads, job_id)
            self._maintenance_wake.set()

    @contextmanager
    def track_media(self, job_id: str):
        with self.lock:
            job = self.store.get(job_id)
            if not job or job["status"] != "completed" or retention.is_expired(job):
                raise MediaUnavailableError("This export is no longer available.")
            directory = self.store.directory(job_id)
            output = directory / "blurred.mp4"
            if (
                self.store.jobs_dir.is_symlink()
                or directory.is_symlink()
                or output.is_symlink()
                or not output.is_file()
            ):
                raise MediaUnavailableError("This export is no longer available.")
            self._media_leases[job_id] = self._media_leases.get(job_id, 0) + 1
        try:
            yield
        finally:
            with self.lock:
                self._release(self._media_leases, job_id)
            self._maintenance_wake.set()

    @staticmethod
    def _release(leases: dict[str, int], job_id: str):
        if leases[job_id] == 1:
            del leases[job_id]
        else:
            leases[job_id] -= 1

    def collect_retention(self, *, at: datetime | None = None) -> dict[str, int]:
        with self.lock:
            at = at or retention.utc_time()
            self.store.initialize_retention_deadlines(at)
            counts = retention.collect(
                self.store,
                active_uploads=set(self._uploads),
                leased_jobs=set(self._media_leases),
                at=at,
            )
            for job_id in list(self._previews):
                job = self.store.get(job_id)
                if not job or job["status"] != "processing":
                    self._previews.pop(job_id, None)
            return counts

    def _maintain(self):
        while not self._stop.is_set():
            self._maintenance_wake.wait(retention.MAINTENANCE_INTERVAL)
            self._maintenance_wake.clear()
            if self._stop.is_set():
                return
            try:
                self.collect_retention()
            except Exception:
                log.exception("Media maintenance will retry on its next interval.")

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
                self._previews.pop(job_id, None)
                return self.store.update(
                    job_id, message="Cancelling…", preview_frame=None, preview_time=None
                )
            return job

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.store.get(job_id)
            if not job:
                return False
            if job["status"] in {"queued", "processing"}:
                raise ValueError("Cancel the job and wait for it to stop before deleting it.")
            if job_id in self._media_leases:
                raise ValueError("The export is in use. Stop playback or downloading first.")
            try:
                retention.remove_job_media(self.store, job_id)
            except OSError as exc:
                raise ValueError("Media could not be removed. Try deleting again.") from exc
            self._previews.pop(job_id, None)
            self.store.delete(job_id)
            return True

    def archive(self, job_id: str) -> dict | None:
        with self.lock:
            if job_id in self._media_leases or job_id in self._uploads:
                raise ValueError("The video is in use; wait before archiving it.")
            try:
                archived = retention.archive_job(self.store, job_id, retention.utc_time())
            except OSError as exc:
                raise ValueError("Media could not be removed. Try archiving again.") from exc
            self._previews.pop(job_id, None)
            return archived

    def get_preview(self, job_id: str) -> PreviewSnapshot | None:
        with self.lock:
            job = self.store.get(job_id)
            if not job or job["status"] != "processing":
                return None
            return self._previews.get(job_id)

    def _clean_output(self, job_id: str):
        self.store.directory(job_id)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = os.open(self.store.jobs_dir, flags)
            try:
                directory_fd = os.open(job_id, flags, dir_fd=root_fd)
            finally:
                os.close(root_fd)
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("Output cleanup refused an unsafe directory (%s).", type(exc).__name__)
            return
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    if entry.name == "source.bin":
                        continue
                    try:
                        mode = entry.stat(follow_symlinks=False).st_mode
                        if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                            os.unlink(entry.name, dir_fd=directory_fd)
                    except OSError as exc:
                        log.warning("Output cleanup will retry (%s).", type(exc).__name__)
        finally:
            os.close(directory_fd)

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

                def preview(
                    jpeg: bytes,
                    frame: int,
                    timestamp: float,
                    current_job_id=job_id,
                    current_event=event,
                ):
                    # Only the pipeline's fully processed JPEG crosses this boundary.
                    # The latest image is transient; SQLite stores its version/time only.
                    with self.lock:
                        current = self.store.get(current_job_id)
                        if (
                            current_event.is_set()
                            or self._stop.is_set()
                            or not current
                            or current["status"] != "processing"
                        ):
                            return
                        self._previews[current_job_id] = PreviewSnapshot(jpeg, frame, timestamp)
                        self.store.update(
                            current_job_id, preview_frame=frame, preview_time=timestamp
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
                    preview=preview,
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
                            preview_frame=None,
                            preview_time=None,
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
                    self._previews.pop(job_id, None)
                    self.store.update(job_id, preview_frame=None, preview_time=None)
                    self._cancel.pop(job_id, None)
