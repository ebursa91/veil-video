import fcntl
import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from veil import retention
from veil import store as store_module
from veil.jobs import JobManager, MediaUnavailableError
from veil.store import Store

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def finish(store, monkeypatch, *, status="completed", at=BASE):
    with monkeypatch.context() as clock:
        clock.setattr(store_module, "now", lambda: at.isoformat())
        job = store.create("phone.mov", 10, "cpu")
        directory = store.directory(job["id"])
        directory.mkdir()
        (directory / "source.bin").write_bytes(b"private original")
        (directory / "blurred.mp4").write_bytes(b"processed export")
        scratch = directory / ".veil-interrupted"
        scratch.mkdir()
        (scratch / "video.mp4").write_bytes(b"partial export")
        return store.update(
            job["id"],
            status=status,
            media={"details": "discard on archive"},
            result={"frames": 100},
            error="discard on archive",
        )


def legacy(store, job):
    job = dict(job)
    for key in ("finished_at", "expires_at", "archived_at", "previous_status"):
        job.pop(key, None)
    with store.connection() as db:
        db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(job), job["id"]))


def age(path, at=BASE):
    os.utime(path, (at.timestamp(), at.timestamp()), follow_symlinks=False)


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_finished_timestamp_and_deadline_are_stable_on_later_updates(settings, monkeypatch, status):
    store = Store(settings.data_dir, retention_days=3)
    job = finish(store, monkeypatch, status=status)
    assert job["finished_at"] == BASE.isoformat()
    assert job["expires_at"] == (BASE + timedelta(days=3)).isoformat()
    monkeypatch.setattr(store_module, "now", lambda: (BASE + timedelta(days=1)).isoformat())
    again = store.update(
        job["id"],
        progress=100,
        preview_frame=None,
        finished_at="replacement",
        expires_at="replacement",
    )
    assert again["finished_at"] == job["finished_at"]
    assert again["expires_at"] == job["expires_at"]


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_expired_terminal_media_and_scratch_are_removed_before_archival(
    settings,
    monkeypatch,
    status,
):
    store = Store(settings.data_dir)
    job = finish(store, monkeypatch, status=status)
    manager = JobManager(settings, store)
    counts = manager.collect_retention(at=BASE + timedelta(days=8))
    archived = store.get(job["id"])
    assert counts["archived"] == 1 and counts["errors"] == 0
    assert archived["status"] == archived["stage"] == "archived"
    assert archived["previous_status"] == status
    assert archived["filename"] == "phone.mov"
    assert archived["finished_at"] == job["finished_at"]
    assert archived["expires_at"] == job["expires_at"]
    assert archived["archived_at"] == (BASE + timedelta(days=8)).isoformat()
    assert archived["result"] is archived["media"] is archived["error"] is None
    assert not store.directory(job["id"]).exists()
    assert manager.collect_retention(at=BASE + timedelta(days=9))["archived"] == 0


def test_legacy_terminal_jobs_receive_grace_once_and_restarts_do_not_extend_it(
    settings, monkeypatch
):
    store = Store(settings.data_dir)
    job = finish(store, monkeypatch, at=BASE - timedelta(days=100))
    legacy(store, job)
    assert store.get(job["id"]).get("expires_at") is None
    manager = JobManager(settings, store)
    assert manager.collect_retention(at=BASE)["archived"] == 0
    migrated = store.get(job["id"])
    assert migrated["finished_at"] == BASE.isoformat()
    assert migrated["expires_at"] == (BASE + timedelta(days=7)).isoformat()
    restarted = JobManager(settings, Store(settings.data_dir))
    assert restarted.collect_retention(at=BASE + timedelta(days=2))["archived"] == 0
    assert restarted.store.get(job["id"])["expires_at"] == migrated["expires_at"]
    assert restarted.collect_retention(at=BASE + timedelta(days=7))["archived"] == 1


def test_failed_worker_lease_cannot_migrate_or_collect_media(settings, monkeypatch):
    store = Store(settings.data_dir)
    job = finish(store, monkeypatch)
    legacy(store, job)
    with (settings.data_dir / "worker.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another Veil process"):
            JobManager(settings, store).start()
    assert store.get(job["id"]).get("expires_at") is None
    assert (store.directory(job["id"]) / "source.bin").read_bytes() == b"private original"


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_old_active_jobs_are_never_collected(settings, status):
    store = Store(settings.data_dir)
    job = store.create("active.mov", 10, "cpu")
    directory = store.directory(job["id"])
    directory.mkdir()
    (directory / "source.bin").write_bytes(b"active upload")
    store.update(job["id"], status=status, expires_at=BASE.isoformat())
    age(directory)
    assert (
        JobManager(settings, store).collect_retention(at=BASE + timedelta(days=100))["archived"]
        == 0
    )
    assert store.get(job["id"])["status"] == status
    assert directory.exists()


def test_collection_retries_filesystem_failure_and_post_delete_metadata_failure(
    settings, monkeypatch
):
    store = Store(settings.data_dir)
    job = finish(store, monkeypatch)
    manager = JobManager(settings, store)
    real_rmtree = retention.shutil.rmtree

    def partial_failure(name, *, dir_fd):
        child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=dir_fd)
        try:
            os.unlink("blurred.mp4", dir_fd=child_fd)
        finally:
            os.close(child_fd)
        raise PermissionError("simulated partial cleanup")

    partial_failure.avoids_symlink_attacks = True
    monkeypatch.setattr(retention.shutil, "rmtree", partial_failure)
    assert manager.collect_retention(at=BASE + timedelta(days=8))["errors"] == 1
    assert store.get(job["id"])["status"] == "completed"
    assert store.directory(job["id"]).exists()
    monkeypatch.setattr(retention.shutil, "rmtree", real_rmtree)
    commit = store.archive_metadata

    def fail_commit(*args):
        raise RuntimeError("simulated database failure after removal")

    monkeypatch.setattr(store, "archive_metadata", fail_commit)
    assert manager.collect_retention(at=BASE + timedelta(days=8))["errors"] == 1
    assert not store.directory(job["id"]).exists()
    assert store.get(job["id"])["status"] == "completed"
    monkeypatch.setattr(store, "archive_metadata", commit)
    assert manager.collect_retention(at=BASE + timedelta(days=8))["archived"] == 1


def test_only_stale_unreferenced_uuid_remnants_are_collected(settings):
    store = Store(settings.data_dir)
    manager = JobManager(settings, store)
    uploads = settings.data_dir / ".uploads"
    uploads.mkdir()
    old_part, active_part, recent_part = [uploads / f"{digit * 32}.part" for digit in "abc"]
    for path in (old_part, active_part, recent_part):
        path.write_bytes(b"upload part")
        age(path)
    age(recent_part, BASE + timedelta(days=2))
    unsafe = uploads / "notes.part"
    unsafe.write_bytes(b"not managed media")
    age(unsafe)
    referenced = store.create("queued.mov", 1, "cpu")
    referenced_part = uploads / f"{referenced['id']}.part"
    referenced_part.write_bytes(b"referenced")
    age(referenced_part)
    outside = settings.data_dir / "outside"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_bytes(b"outside content")
    linked_part = uploads / f"{'d' * 32}.part"
    linked_part.symlink_to(marker)
    orphan, active_dir, linked_dir = [store.directory(digit * 32) for digit in "ef0"]
    orphan.mkdir()
    (orphan / "nested-link").symlink_to(marker)
    age(orphan)
    active_dir.mkdir()
    age(active_dir)
    linked_dir.symlink_to(outside, target_is_directory=True)
    with manager.track_upload("b" * 32), manager.track_upload("f" * 32):
        counts = manager.collect_retention(at=BASE + timedelta(days=2))
        assert counts["stale_uploads"] == counts["orphan_jobs"] == 1
        assert active_part.exists() and active_dir.exists()
    assert not old_part.exists() and not orphan.exists()
    assert recent_part.exists() and referenced_part.exists() and unsafe.exists()
    assert linked_part.is_symlink() and linked_dir.is_symlink()
    assert marker.read_bytes() == b"outside content"


def test_upload_registration_does_not_hold_lock_and_protects_database_handoff(settings):
    store = Store(settings.data_dir)
    manager = JobManager(settings, store)
    job_id = "a" * 32
    acquired = threading.Event()

    def acquire():
        with manager.lock:
            acquired.set()

    with manager.track_upload(job_id):
        thread = threading.Thread(target=acquire)
        thread.start()
        assert acquired.wait(1)
        thread.join()
        directory = store.directory(job_id)
        directory.mkdir()
        age(directory)
        assert manager.collect_retention(at=BASE + timedelta(days=2))["orphan_jobs"] == 0
        store.create("finished-upload.mov", 10, "cpu", job_id)
    assert manager.collect_retention(at=BASE + timedelta(days=2))["orphan_jobs"] == 0
    assert directory.exists()
    assert not manager._uploads


def test_media_lease_survives_expiry_blocks_new_leases_and_releases_after_exception(
    settings,
    monkeypatch,
):
    store = Store(settings.data_dir)
    job = finish(store, monkeypatch)
    manager = JobManager(settings, store)
    monkeypatch.setattr(retention, "utc_time", lambda: BASE)
    with pytest.raises(RuntimeError, match="client disconnected"):
        with manager.track_media(job["id"]):
            with pytest.raises(ValueError, match="in use"):
                manager.delete(job["id"])
            assert manager.collect_retention(at=BASE + timedelta(days=8))["archived"] == 0
            monkeypatch.setattr(retention, "utc_time", lambda: BASE + timedelta(days=8))
            with pytest.raises(MediaUnavailableError):
                with manager.track_media(job["id"]):
                    pytest.fail("Expired media cannot acquire a new lease")
            assert (store.directory(job["id"]) / "blurred.mp4").exists()
            raise RuntimeError("client disconnected")
    assert not manager._media_leases
    assert manager.collect_retention()["archived"] == 1


def test_restart_cleanup_refuses_job_directory_symlink(settings):
    store = Store(settings.data_dir)
    job = store.create("interrupted.mov", 10, "cpu")
    store.update(job["id"], status="processing")
    outside = settings.data_dir / "outside"
    outside.mkdir()
    marker = outside / "blurred.mp4"
    marker.write_bytes(b"do not remove")
    store.directory(job["id"]).symlink_to(outside, target_is_directory=True)
    manager = JobManager(settings, store)
    manager.start()
    manager.stop()
    assert marker.read_bytes() == b"do not remove"
    assert store.get(job["id"])["status"] == "failed"


def test_media_removal_stays_in_opened_root_during_parent_symlink_swap(settings, monkeypatch):
    store = Store(settings.data_dir)
    job_id = "a" * 32
    directory = store.directory(job_id)
    directory.mkdir()
    (directory / "source.bin").write_bytes(b"managed media")
    outside = settings.data_dir / "outside"
    target = outside / job_id
    target.mkdir(parents=True)
    marker = target / "keep"
    marker.write_bytes(b"outside content")
    moved = settings.data_dir / "original-jobs"
    remove = retention.shutil.rmtree

    def swap(name, *, dir_fd):
        store.jobs_dir.rename(moved)
        store.jobs_dir.symlink_to(outside, target_is_directory=True)
        remove(name, dir_fd=dir_fd)

    swap.avoids_symlink_attacks = True
    monkeypatch.setattr(retention.shutil, "rmtree", swap)
    retention.remove_job_media(store, job_id)
    assert not (moved / job_id).exists()
    assert marker.read_bytes() == b"outside content"


def test_upload_removal_stays_in_opened_root_during_parent_symlink_swap(settings, monkeypatch):
    store = Store(settings.data_dir)
    job_id = "b" * 32
    uploads = settings.data_dir / ".uploads"
    uploads.mkdir()
    (uploads / f"{job_id}.part").write_bytes(b"managed upload")
    outside = settings.data_dir / "outside"
    outside.mkdir()
    marker = outside / f"{job_id}.part"
    marker.write_bytes(b"outside content")
    moved = settings.data_dir / "original-uploads"
    unlink = retention.os.unlink

    def swap(name, *, dir_fd=None):
        uploads.rename(moved)
        uploads.symlink_to(outside, target_is_directory=True)
        unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(retention.os, "unlink", swap)
    retention.remove_upload_part(store, job_id)
    assert not (moved / f"{job_id}.part").exists()
    assert marker.read_bytes() == b"outside content"


def test_startup_preserves_recent_upload_and_periodic_collection_expires_finished_media(
    settings,
    monkeypatch,
):
    store = Store(settings.data_dir)
    current = retention.utc_time()
    job = finish(store, monkeypatch, at=current)
    uploads = settings.data_dir / ".uploads"
    uploads.mkdir()
    recent = uploads / f"{'c' * 32}.part"
    recent.write_bytes(b"recent unfinished upload")
    monkeypatch.setattr(retention, "MAINTENANCE_INTERVAL", 0.05)
    manager = JobManager(settings, store)
    manager.start()
    try:
        assert recent.exists()
        monkeypatch.setattr(retention, "utc_time", lambda: current + timedelta(days=8))
        for _ in range(100):
            if store.get(job["id"])["status"] == "archived":
                break
            time.sleep(0.02)
        assert store.get(job["id"])["status"] == "archived"
    finally:
        manager.stop()
    assert not manager._maintenance_thread.is_alive()
