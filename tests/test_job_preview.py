import threading
import time

import pytest

from veil.jobs import JobManager
from veil.store import Store


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
def test_worker_preview_is_transient_and_late_callbacks_are_ignored(settings, monkeypatch, outcome):
    import veil.media
    import veil.pipeline

    monkeypatch.setattr(veil.media, "probe", lambda _: {"width": 20, "height": 20})
    published, release = threading.Event(), threading.Event()
    callbacks = []

    def process(source, output, **kwargs):
        publish = kwargs["preview"]
        callbacks.append(publish)
        publish(b"processed JPEG one", 1, 0.0)
        published.set()
        assert release.wait(3)
        publish(b"processed JPEG two", 31, 1.0)
        if outcome == "failed":
            raise RuntimeError("injected export failure")
        output.write_bytes(b"processed export")
        return {"frames_processed": 31}

    monkeypatch.setattr(veil.pipeline, "process_video", process)
    store = Store(settings.data_dir)
    job = store.create("sample.mov", 10, "cpu")
    directory = store.directory(job["id"])
    directory.mkdir()
    (directory / "source.bin").write_bytes(b"private original")
    manager = JobManager(settings, store)
    manager.start()
    try:
        assert published.wait(3)
        preview = manager.get_preview(job["id"])
        assert preview.jpeg == b"processed JPEG one"
        assert (preview.frame, preview.time) == (1, 0.0)
        assert store.get(job["id"])["preview_frame"] == 1
        # Only the latest JPEG lives in memory; no thumbnail is persisted on disk.
        assert list(directory.iterdir()) == [directory / "source.bin"]
        if outcome == "cancelled":
            manager.cancel(job["id"])
            assert manager.get_preview(job["id"]) is None
            assert store.get(job["id"])["preview_frame"] is None
        release.set()
        for _ in range(150):
            if store.get(job["id"])["status"] == outcome and not manager._previews:
                break
            time.sleep(0.02)
        final = store.get(job["id"])
        assert final["status"] == outcome
        assert final["preview_frame"] is None
        assert final["preview_time"] is None
        assert not manager._previews
        callbacks[0](b"late preview must be ignored", 61, 2.0)
        assert manager.get_preview(job["id"]) is None
        assert not manager._previews
        assert store.get(job["id"])["preview_frame"] is None
        assert (directory / "source.bin").read_bytes() == b"private original"
    finally:
        release.set()
        manager.stop()
