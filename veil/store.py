"""Small durable queue. One SQLite connection per operation, one worker per data directory."""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def valid_job_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


class Store:
    def __init__(self, data_dir: Path, *, retention_days: int = 7):
        if not 1 <= retention_days <= 365:
            raise ValueError("Retention must be between 1 and 365 days.")
        self.retention_days = retention_days
        self.data_dir = data_dir
        self.jobs_dir = data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = data_dir / "veil.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, "
                "created_at TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL)"
            )

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("PRAGMA busy_timeout=15000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(
        self,
        filename: str,
        size: int,
        mode: str,
        job_id: str | None = None,
        *,
        effect: str = "blur",
    ) -> dict:
        if effect not in {"blur", "emoji"}:
            raise ValueError("Unknown face effect.")
        job = {
            "id": job_id or uuid.uuid4().hex,
            "filename": filename,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "created_at": now(),
            "updated_at": now(),
            "source_bytes": size,
            "mode": mode,
            "effect": effect,
            "media": None,
            "frames_processed": 0,
            "total_frames": None,
            "fps": None,
            "eta_seconds": None,
            "backend": None,
            "message": "Waiting for the video worker",
            "error": None,
            "result": None,
            "preview_frame": None,
            "preview_time": None,
            "finished_at": None,
            "expires_at": None,
            "archived_at": None,
            "previous_status": None,
        }
        with self.connection() as db:
            db.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?)",
                (job["id"], job["created_at"], job["status"], json.dumps(job)),
            )
        return job

    def list(self) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT body FROM jobs ORDER BY created_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def get(self, job_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, job_id: str, **fields) -> dict | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            job = json.loads(row[0])
            previous_status = job["status"]
            finished_at, expires_at = job.get("finished_at"), job.get("expires_at")
            timestamp = now()
            job.update(fields, updated_at=timestamp)
            # Terminal timestamps belong to the first transition, not later
            # progress/preview updates. Legacy rows get their grace in migration.
            if finished_at:
                job["finished_at"] = finished_at
            if expires_at:
                job["expires_at"] = expires_at
            if job["status"] in TERMINAL_STATUSES and previous_status not in (
                TERMINAL_STATUSES | {"archived"}
            ):
                job["finished_at"] = finished_at or timestamp
                job["expires_at"] = (
                    expires_at
                    or (
                        datetime.fromisoformat(job["finished_at"])
                        + timedelta(days=self.retention_days)
                    ).isoformat()
                )
            db.execute(
                "UPDATE jobs SET status=?, body=? WHERE id=?",
                (job["status"], json.dumps(job), job_id),
            )
        return job

    def initialize_retention_deadlines(self, at: datetime) -> int:
        """Give legacy terminal jobs a full grace once, after acquiring the worker lease."""
        initialized = 0
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT id, body FROM jobs WHERE status IN ('completed', 'failed', 'cancelled')"
            ).fetchall()
            for job_id, body in rows:
                job = json.loads(body)
                if job.get("expires_at"):
                    continue
                job["finished_at"] = job.get("finished_at") or at.isoformat()
                job["expires_at"] = (at + timedelta(days=self.retention_days)).isoformat()
                job["updated_at"] = at.isoformat()
                db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(job), job_id))
                initialized += 1
        return initialized

    def archive_metadata(self, job_id: str, at: datetime) -> dict | None:
        """Commit the small history record only after all job media has been removed."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            job = json.loads(row[0])
            if job["status"] == "archived":
                return job
            if job["status"] not in TERMINAL_STATUSES:
                raise ValueError("Only finished jobs can be archived.")
            # lexists also catches a dangling symlink; absence must be real.
            if os.path.lexists(self.directory(job_id)):
                raise ValueError("Job media must be removed before archiving metadata.")
            keep = {
                "id",
                "filename",
                "created_at",
                "source_bytes",
                "mode",
                "effect",
                "progress",
                "frames_processed",
                "total_frames",
                "finished_at",
                "expires_at",
            }
            archived = {key: job.get(key) for key in keep}
            archived.update(
                status="archived",
                stage="archived",
                previous_status=job["status"],
                updated_at=at.isoformat(),
                archived_at=at.isoformat(),
                message="Media removed; history retained.",
                media=None,
                result=None,
                error=None,
                preview_frame=None,
                preview_time=None,
                fps=None,
                eta_seconds=None,
                backend=None,
            )
            db.execute(
                "UPDATE jobs SET status='archived', body=? WHERE id=?",
                (json.dumps(archived), job_id),
            )
        return archived

    def delete(self, job_id: str) -> None:
        with self.connection() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def directory(self, job_id: str) -> Path:
        # Only database-generated identifiers are ever resolved as paths.
        if not valid_job_id(job_id):
            raise ValueError("Invalid job identifier")
        return self.jobs_dir / job_id
