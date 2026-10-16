import gc
import weakref

import cv2
import numpy as np
import pytest

from veil import preview


@pytest.mark.parametrize(
    "width,height,expected",
    [(1920, 1080, (640, 360)), (1080, 1920, (360, 640)), (321, 123, (321, 123))],
)
def test_jpeg_dimensions_are_bounded_without_upscaling_or_mutating_frame(width, height, expected):
    frame = np.full((height, width, 3), (15, 80, 170), np.uint8)
    jpeg = preview.encode_jpeg(frame)
    assert isinstance(jpeg, bytes)
    assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert (decoded.shape[1], decoded.shape[0]) == expected
    assert np.all(frame == (15, 80, 170))


def test_encoding_and_callback_are_throttled_by_wall_time(monkeypatch):
    encoded, emitted = [], []

    def encode(frame):
        encoded.append(True)
        return b"jpeg"

    monkeypatch.setattr(preview, "encode_jpeg", encode)
    capture = preview.ProcessedPreview(lambda *values: emitted.append(values), fps=30)
    frame = np.zeros((16, 16, 3), np.uint8)
    for frame_no, now in enumerate([0, 0.2, 0.99, 1.0, 1.4, 2.1], start=1):
        capture.emit(frame, frame_no, now=now)
    assert len(encoded) == 3
    assert [entry[1] for entry in emitted] == [1, 4, 6]
    assert [entry[2] for entry in emitted] == pytest.approx([0, 3 / 30, 5 / 30])


def test_forced_final_snapshot_is_not_duplicated(monkeypatch):
    monkeypatch.setattr(preview, "encode_jpeg", lambda frame: b"jpeg")
    emitted = []
    capture = preview.ProcessedPreview(lambda *values: emitted.append(values), fps=30)
    frame = np.zeros((16, 16, 3), np.uint8)
    capture.emit(frame, 1, now=0)
    capture.emit(frame, 30, now=0.2, force=True)
    capture.emit(frame, 30, now=0.3, force=True)
    assert [entry[1] for entry in emitted] == [1, 30]
    assert emitted[-1][2] == pytest.approx(29 / 30)


@pytest.mark.parametrize("failure", ["encode", "callback"])
def test_capture_failure_logs_once_disables_capture_and_retains_no_image(
    monkeypatch,
    caplog,
    failure,
):
    encodes, callbacks = [], []

    def encode(frame):
        encodes.append(True)
        if failure == "encode":
            raise RuntimeError("test capture failure")
        return b"jpeg"

    def callback(*values):
        callbacks.append(True)
        raise RuntimeError("test callback failure")

    monkeypatch.setattr(preview, "encode_jpeg", encode)
    capture = preview.ProcessedPreview(callback, fps=30)
    frame = np.zeros((16, 16, 3), np.uint8)
    reference = weakref.ref(frame)
    capture.emit(frame, 1, now=0)
    capture.emit(frame, 2, now=2, force=True)
    del frame
    gc.collect()
    assert reference() is None
    assert not capture.enabled
    assert len(encodes) == 1
    assert len(callbacks) == (0 if failure == "encode" else 1)
    assert sum("Live preview disabled" in record.message for record in caplog.records) == 1


def test_jpeg_encoder_failure_is_reported(monkeypatch):
    monkeypatch.setattr(preview.cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(RuntimeError, match="JPEG encoding failed"):
        preview.encode_jpeg(np.zeros((16, 16, 3), np.uint8))
