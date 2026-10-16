import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from veil import pipeline


def test_prefetch_overlaps_work_and_preserves_order():
    second_started = threading.Event()
    release_second = threading.Event()
    calls = []
    main_thread = threading.get_ident()

    def read(check_cancel):
        index = len(calls)
        calls.append((index, threading.get_ident()))
        if index == 1:
            second_started.set()
            assert release_second.wait(2)
        return index if index < 8 else None

    def cancelled():
        assert threading.get_ident() == main_thread
        return False

    try:
        with pipeline._prefetch_frames(read, cancelled) as frames:
            first = next(frames)
            assert first == 0
            assert second_started.wait(2), "next inference must overlap current-frame work"
            assert len(calls) == 2, "only one lookahead may be in flight"
            release_second.set()
            assert [first, *frames] == list(range(8))
    finally:
        release_second.set()
    assert len({thread for _, thread in calls}) == 1
    assert calls[0][1] != main_thread


def test_prefetch_propagates_failure_without_skipping_a_frame():
    count = 0

    def read(check_cancel):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("inference failed")
        return "first masked frame"

    output = []
    with pytest.raises(ValueError, match="inference failed"):
        with pipeline._prefetch_frames(read, lambda: False) as frames:
            output.extend(frames)
    assert output == ["first masked frame"]
    assert not any(t.name.startswith("veil-prefetch") for t in threading.enumerate())


def test_writer_failure_aborts_a_blocked_prefetch_before_returning():
    blocked = threading.Event()
    stopped = threading.Event()
    count = 0

    def read(check_cancel):
        nonlocal count
        count += 1
        if count == 1:
            return 0
        blocked.set()
        while not check_cancel():
            stopped.wait(0.01)
        stopped.set()
        raise pipeline.CancelledError("internal prefetch stopped")

    with pytest.raises(OSError, match="encoder failed"):
        with pipeline._prefetch_frames(read, lambda: False) as frames:
            assert next(frames) == 0
            assert blocked.wait(2)
            raise OSError("encoder failed")
    assert stopped.is_set()
    assert not any(t.name.startswith("veil-prefetch") for t in threading.enumerate())


def test_cancel_while_waiting_for_first_frame_joins_worker():
    started = threading.Event()
    cancelled = threading.Event()
    finished = threading.Event()

    def read(check_cancel):
        started.set()
        while not check_cancel():
            finished.wait(0.01)
        finished.set()
        raise pipeline.CancelledError("prefetch aborted")

    def consume():
        with pipeline._prefetch_frames(read, cancelled.is_set) as frames:
            list(frames)

    with ThreadPoolExecutor(max_workers=1) as caller:
        task = caller.submit(consume)
        assert started.wait(2)
        cancelled.set()
        with pytest.raises(pipeline.CancelledError):
            task.result(timeout=2)
    assert finished.is_set()


def test_serial_fallback_keeps_reader_on_calling_thread():
    calls = []

    def read(check_cancel):
        calls.append(threading.get_ident())
        return "frame" if len(calls) == 1 else None

    with pipeline._prefetch_frames(read, lambda: False, enabled=False) as frames:
        assert list(frames) == ["frame"]
    assert calls == [threading.get_ident()] * 2


def test_writable_frame_buffer_and_partial_encoder_writes(monkeypatch):
    frame = np.arange(3 * 8 * 12, dtype=np.uint8).reshape(8, 12, 3)
    expected = frame.tobytes()
    writes = bytearray()

    class Pipe:
        def fileno(self):
            return 123

    class Selector:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def register(self, *args):
            pass

        def select(self, timeout):
            return [True]

    def write(fd, view):
        assert fd == 123
        part = view[:17]
        writes.extend(part)
        return len(part)

    monkeypatch.setattr(pipeline.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(os, "write", write)
    pipeline._write_frame(Pipe(), memoryview(frame).cast("B"), lambda: False)
    assert bytes(writes) == expected


def test_pipe_buffering_is_optional_on_restricted_hosts(monkeypatch):
    def denied(*args):
        raise PermissionError("not allowed")

    class Pipe:
        def fileno(self):
            return 123

    monkeypatch.setattr(pipeline.fcntl, "fcntl", denied)
    pipeline._buffer_pipe(Pipe())
