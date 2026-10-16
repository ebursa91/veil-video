"""Small durable queue. One SQLite connection per operation, one worker per data directory."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, data_dir: Path):
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
            job.update(fields, updated_at=now())
            db.execute(
                "UPDATE jobs SET status=?, body=? WHERE id=?",
                (job["status"], json.dumps(job), job_id),
            )
        return job

    def delete(self, job_id: str) -> None:
        with self.connection() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def directory(self, job_id: str) -> Path:
        # Only database-generated identifiers are ever resolved as paths.
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Invalid job identifier")
        return self.jobs_dir / job_id
