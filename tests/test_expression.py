import hashlib
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from veil import model
from veil.detector import Face, Landmarks
from veil.expression import (
    ALIGNMENT_POINTS,
    EXPRESSION_LABELS,
    ExpressionClassifier,
    alignment_matrix,
    expression_blob,
    expression_from_logits,
)


def aligned_face(points=ALIGNMENT_POINTS):
    return Face((20, 25, 75, 80), Landmarks(*(tuple(point) for point in points)), 0.99)


def test_alignment_recovers_rotation_scale_and_translation_without_mirroring():
    angle = np.deg2rad(20)
    forward = np.array([[np.cos(angle), -np.sin(angle), 15], [np.sin(angle), np.cos(angle), -7]])
    forward[:, :2] *= 1.7
    source = np.column_stack((ALIGNMENT_POINTS, np.ones(5))) @ forward.T
    transform = alignment_matrix(source)
    np.testing.assert_allclose(transform, cv2.invertAffineTransform(forward), atol=1e-12)
    transformed = np.column_stack((source, np.ones(5))) @ transform.T
    np.testing.assert_allclose(transformed, ALIGNMENT_POINTS, atol=1e-11)


def test_preprocessing_is_rgb_normalized_contiguous_nchw():
    frame = np.full((112, 112, 3), (0, 127, 255), np.uint8)
    blob = expression_blob(frame, aligned_face())
    assert blob.shape == (1, 3, 112, 112)
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous
    np.testing.assert_array_equal(blob[0, 0], np.ones((112, 112), np.float32))
    np.testing.assert_array_equal(blob[0, 2], -np.ones((112, 112), np.float32))
    expected_green = (np.float32(127) / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
    np.testing.assert_array_equal(blob[0, 1], np.full((112, 112), expected_green, np.float32))


def test_softmax_is_stable_and_uses_official_label_order():
    result = expression_from_logits(np.array([[1000, 1001, 1002, 1010, 1003, 1004, 1005]]))
    assert result.label == "happy"
    assert result.class_id == 3
    assert result.confidence > 0.98
    assert sum(result.probabilities) == pytest.approx(1)
    assert result.confidence == max(result.probabilities)
    for index, label in enumerate(EXPRESSION_LABELS):
        logits = np.zeros(7)
        logits[index] = 5
        assert expression_from_logits(logits).label == label
    with pytest.raises(RuntimeError, match="invalid class scores"):
        expression_from_logits(np.array([np.nan] * 7))


@pytest.mark.parametrize("points", [np.zeros((5, 2)), np.full((5, 2), np.nan)])
def test_invalid_landmarks_use_low_confidence_neutral_without_inference(points):
    classifier = ExpressionClassifier.__new__(ExpressionClassifier)

    class MustNotRun:
        def run(self, *args):
            pytest.fail("Degenerate face alignment must not reach model inference")

    classifier.session = MustNotRun()
    result = classifier.classify(np.zeros((112, 112, 3), np.uint8), aligned_face(points))
    assert result.label == "neutral"
    assert result.confidence == pytest.approx(1 / 7)
    assert sum(result.probabilities) == pytest.approx(1)


def test_missing_expression_model_fails_without_downloading(tmp_path, monkeypatch):
    def no_download(*args, **kwargs):
        pytest.fail("Upload-time inference must not download model weights")

    monkeypatch.setattr(model.urllib.request, "urlopen", no_download)
    with pytest.raises(ValueError, match="veil init"):
        ExpressionClassifier(tmp_path / "missing.onnx")


def test_crowded_frame_classification_checks_cancellation_between_faces(monkeypatch):
    classifier = ExpressionClassifier.__new__(ExpressionClassifier)
    processed, checks = [], []
    prediction = expression_from_logits(np.zeros(7))

    def classify(frame, face):
        processed.append(face)
        return prediction

    def check_cancel():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError("cancelled between faces")

    monkeypatch.setattr(classifier, "classify", classify)
    face = aligned_face()
    with pytest.raises(RuntimeError, match="cancelled between faces"):
        classifier.classify_many(
            np.zeros((112, 112, 3), np.uint8), [face] * 100, check_cancel=check_cancel
        )
    assert processed == [face]
    assert len(checks) == 2


def test_expression_model_install_is_atomic_verified_and_reuses_cache(tmp_path, monkeypatch):
    weights = b"test-model"
    monkeypatch.setattr(model, "EXPRESSION_MODEL_BYTES", len(weights))
    monkeypatch.setattr(model, "EXPRESSION_MODEL_SHA256", hashlib.sha256(weights).hexdigest())
    calls = []

    def download(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(weights)

    monkeypatch.setattr(model.urllib.request, "urlopen", download)
    target = tmp_path / "models" / "expression.onnx"
    assert model.ensure_expression_model(target) == target
    assert model.verify_expression_model(target)
    assert model.ensure_expression_model(target) == target
    assert calls == [model.EXPRESSION_MODEL_URL]
    assert list(target.parent.iterdir()) == [target]
    target.write_bytes(b"x" * len(weights))
    assert not model.verify_expression_model(target)


def test_corrupt_expression_download_does_not_replace_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(model.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad"))
    target = tmp_path / "expression.onnx"
    target.write_bytes(b"previous incomplete file")
    with pytest.raises(ValueError, match="checksum mismatch"):
        model.ensure_expression_model(target)
    assert target.read_bytes() == b"previous incomplete file"
    assert list(tmp_path.iterdir()) == [target]


def test_real_expression_model_cpu_inference_if_installed():
    path = Path(os.getenv("VEIL_TEST_EXPRESSION_MODEL", "models/expression.onnx"))
    if not model.verify_expression_model(path):
        pytest.skip("Install expression weights or set VEIL_TEST_EXPRESSION_MODEL")
    classifier = ExpressionClassifier(path, gpu=False)
    predictions = classifier.classify_many(np.zeros((112, 112, 3), np.uint8), [aligned_face()])
    assert len(predictions) == 1
    assert predictions[0].label in EXPRESSION_LABELS
    assert sum(predictions[0].probabilities) == pytest.approx(1)
    assert 0 <= predictions[0].confidence <= 1


def test_real_expression_model_cuda_matches_cpu_when_requested():
    if os.getenv("VEIL_TEST_CUDA") != "1":
        pytest.skip("Set VEIL_TEST_CUDA=1 for actual GPU inference validation")
    path = Path(os.getenv("VEIL_TEST_EXPRESSION_MODEL", "models/expression.onnx"))
    if not model.verify_expression_model(path):
        pytest.skip("Install expression weights or set VEIL_TEST_EXPRESSION_MODEL")
    frame, face = np.zeros((112, 112, 3), np.uint8), aligned_face()
    cpu = ExpressionClassifier(path).classify(frame, face)
    gpu_classifier = ExpressionClassifier(path, gpu=True)
    gpu = gpu_classifier.classify(frame, face)
    assert gpu_classifier.backend == "CUDA"
    assert gpu.label == cpu.label
    # CUDA's TF32 convolution path can differ slightly from CPU float32 kernels.
    np.testing.assert_allclose(gpu.probabilities, cpu.probabilities, atol=2e-4)
