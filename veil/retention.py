"""Conservative local-media collection; callers hold the manager lifecycle lock."""

import logging
import os
import shutil
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

from veil.store import TERMINAL_STATUSES, Store, valid_job_id

log = logging.getLogger(__name__)
ORPHAN_GRACE = timedelta(hours=24)
MAINTENANCE_INTERVAL = 15 * 60


def utc_time() -> datetime:
    return datetime.now(UTC)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value)
        return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)
    except (TypeError, ValueError):
        return None


def is_expired(job: dict, at: datetime | None = None) -> bool:
    deadline = parse_time(job.get("expires_at"))
    return deadline is not None and deadline <= (at or utc_time())


def _open_directory(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def remove_job_media(store: Store, job_id: str) -> None:
    """Remove one exact UUID directory, using Python's fd-based safe rmtree.

    A symlink at a managed root or job root is refused. Nested symlinks are
    unlinked by rmtree without touching their targets.
    """
    store.directory(job_id)
    try:
        root_fd = _open_directory(store.jobs_dir)
    except FileNotFoundError:
        return
    try:
        _remove_job_media(job_id, root_fd)
    finally:
        os.close(root_fd)


def _remove_job_media(job_id: str, root_fd: int) -> None:
    try:
        mode = os.stat(job_id, dir_fd=root_fd, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(mode):
        raise OSError("The job media path is not a plain directory.")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise OSError("This platform does not support safe directory cleanup.")
    # The opened root remains the same directory even if its pathname is
    # replaced during collection. rmtree separately protects the child.
    shutil.rmtree(job_id, dir_fd=root_fd)


def remove_upload_part(store: Store, job_id: str) -> None:
    """Remove one upload file for rollback or collection without following roots."""
    store.directory(job_id)
    try:
        root_fd = _open_directory(store.data_dir / ".uploads")
    except FileNotFoundError:
        return
    try:
        _remove_upload_part(job_id, root_fd)
    finally:
        os.close(root_fd)


def _remove_upload_part(job_id: str, root_fd: int) -> None:
    name = f"{job_id}.part"
    try:
        mode = os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise OSError("The upload part is not a plain file.")
    os.unlink(name, dir_fd=root_fd)


def archive_job(store: Store, job_id: str, at: datetime) -> dict | None:
    job = store.get(job_id)
    if not job:
        return None
    if job["status"] == "archived":
        return job
    if job["status"] not in TERMINAL_STATUSES:
        raise ValueError("Only finished jobs can be archived.")
    remove_job_media(store, job_id)
    # If this commit fails or the server stops here, the next collection sees a
    # terminal record with absent media and safely retries the metadata commit.
    return store.archive_metadata(job_id, at)


def _stale(name: str, root_fd: int, at: datetime, *, directory: bool) -> bool:
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    return expected and info.st_mtime <= (at - ORPHAN_GRACE).timestamp()


def collect(
    store: Store,
    *,
    active_uploads: set[str],
    leased_jobs: set[str],
    at: datetime | None = None,
) -> dict[str, int]:
    """Expire finished media and stale unreferenced remnants, never active jobs."""
    at = at or utc_time()
    counts = {"archived": 0, "stale_uploads": 0, "orphan_jobs": 0, "errors": 0}
    for job in store.list():
        job_id = job["id"]
        if (
            job["status"] not in TERMINAL_STATUSES
            or job_id in active_uploads
            or job_id in leased_jobs
            or not is_expired(job, at)
        ):
            continue
        try:
            # The inventory is advisory; verify current state immediately before
            # removal while the caller still owns the lifecycle lock.
            current = store.get(job_id)
            if current and current["status"] in TERMINAL_STATUSES and is_expired(current, at):
                archive_job(store, job_id, at)
                counts["archived"] += 1
        except Exception as exc:
            counts["errors"] += 1
            log.warning("Expired-media cleanup will retry (%s).", type(exc).__name__)

    candidates = (
        (store.data_dir / ".uploads", False, "stale_uploads"),
        (store.jobs_dir, True, "orphan_jobs"),
    )
    for root, directory, count_key in candidates:
        try:
            root_fd = _open_directory(root)
        except FileNotFoundError:
            continue
        except OSError as exc:
            counts["errors"] += 1
            log.warning("Media inventory will retry (%s).", type(exc).__name__)
            continue
        try:
            with os.scandir(root_fd) as entries:
                for entry in entries:
                    name = entry.name if directory else entry.name.removesuffix(".part")
                    if not directory and entry.name != f"{name}.part":
                        continue
                    if not valid_job_id(name) or name in active_uploads or name in leased_jobs:
                        continue
                    if store.get(name) is not None or not _stale(
                        entry.name,
                        root_fd,
                        at,
                        directory=directory,
                    ):
                        continue
                    try:
                        if directory:
                            _remove_job_media(name, root_fd)
                        else:
                            _remove_upload_part(name, root_fd)
                        counts[count_key] += 1
                    except OSError as exc:
                        counts["errors"] += 1
                        log.warning("Orphan-media cleanup will retry (%s).", type(exc).__name__)
        except OSError as exc:
            counts["errors"] += 1
            log.warning("Media inventory will retry (%s).", type(exc).__name__)
        finally:
            os.close(root_fd)
    return counts
