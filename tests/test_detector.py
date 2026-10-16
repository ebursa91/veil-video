from pathlib import Path

import numpy as np
import pytest

from veil.detector import FaceDetector, MaskTracker, blur_faces, decode_predictions
from veil.model import verify_model


@pytest.mark.parametrize("height,width", [(16, 32), (32, 16), (32, 32)])
def test_detector_input_is_contiguous_nchw_with_unchanged_channels_and_padding(height, width):
    detector = FaceDetector.__new__(FaceDetector)
    detector.size, detector.threshold, detector.input_name = 32, 0.55, "input"
    outputs = {}
    for stride in (8, 16, 32):
        count = (detector.size // stride) ** 2
        outputs[f"cls_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"obj_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"bbox_{stride}"] = np.zeros((1, count, 4), np.float32)
    detector.output_names = list(outputs)
    captured = []

    class RecordingSession:
        def run(self, output_names, inputs):
            captured.append(inputs["input"])
            return list(outputs.values())

    detector.session = RecordingSession()
    frame = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    assert detector.detect(frame) == []
    blob = captured[0]
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous
    expected = np.zeros((1, 3, detector.size, detector.size), np.float32)
    for channel in range(3):
        expected[0, channel, :height, :width] = frame[:, :, channel]
    np.testing.assert_array_equal(blob, expected)


def test_yunet_prediction_coordinates_and_nms():
    outputs = {}
    for stride in (8, 16, 32):
        count = (32 // stride) ** 2
        outputs[f"cls_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"obj_{stride}"] = np.zeros((1, count, 1), np.float32)
        outputs[f"bbox_{stride}"] = np.zeros((1, count, 4), np.float32)
    outputs["cls_8"][0, 5, 0] = 0.81
    outputs["obj_8"][0, 5, 0] = 1
    boxes = decode_predictions(outputs, 32, 0.5, 0.55)
    assert len(boxes) == 1
    assert boxes[0] == pytest.approx((8, 8, 16, 16))


def test_expanded_mask_changes_pixels_and_preserves_background():
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    output = blur_faces(original.copy(), [(100, 70, 50, 50)])
    assert np.array_equal(output[:30], original[:30])
    assert not np.array_equal(output[60:130, 90:160], original[60:130, 90:160])
    assert output[70:120, 100:150].var() < original[70:120, 100:150].var() / 10


def test_tracker_covers_motion_and_brief_detection_dropout():
    tracker = MaskTracker(30)
    tracker.update([(50, 50, 30, 30)])
    masks = tracker.update([(65, 50, 30, 30)])
    assert (50, 50, 45, 30) in masks
    assert tracker.update([]) == [(65, 50, 30, 30)]
    for _ in range(tracker.hold):
        masks = tracker.update([])
    assert not masks


def test_offscreen_boxes_are_safe():
    frame = np.ones((64, 64, 3), np.uint8)
    blur_faces(frame, [(-100, -100, 10, 10), (-5, -5, 20, 20), (60, 60, 30, 30)])
    assert frame.shape == (64, 64, 3)


def test_missing_model_rejected_without_download(tmp_path):
    model = tmp_path / "model.onnx"
    assert not verify_model(model)
    with pytest.raises(ValueError, match="veil init"):
        FaceDetector(model)
    assert not model.exists()


def test_installed_model_detects_blank_frame_if_available():
    import os

    path = Path(os.getenv("VEIL_TEST_MODEL", "models/yunet.onnx"))
    if not verify_model(path):
        pytest.skip("Run veil init or set VEIL_TEST_MODEL for real inference smoke test")
    assert FaceDetector(path).detect(np.zeros((480, 640, 3), np.uint8)) == []
