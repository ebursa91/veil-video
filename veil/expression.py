"""Approximate visible expression classification for decorative emoji selection.

The alignment/preprocessing follows the Apache-2.0 OpenCV Zoo reference:
https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/
models/facial_expression_recognition/facial_fer_model.py

Alignment adapted from Copyright (C) 2022, Shenzhen Institute of Artificial
Intelligence and Robotics for Society. See docs/licenses/Expression-Apache-2.0.txt.
These scores are model outputs, not calibrated claims about a person's feelings.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from veil.detector import Face, _create_session
from veil.model import verify_expression_model

EXPRESSION_LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
ALIGNMENT_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class Expression:
    label: str
    confidence: float
    probabilities: tuple[float, ...]

    @property
    def class_id(self) -> int:
        return EXPRESSION_LABELS.index(self.label)


def alignment_matrix(landmarks) -> np.ndarray:
    """Fit the reference's non-reflective inverse similarity, then invert it.

    Keeping all five points and the same least-squares direction reproduces
    upstream preprocessing without the randomness of RANSAC landmark subsets.
    """
    source = np.asarray(landmarks, dtype=np.float64)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise ValueError("Face alignment requires five finite landmark coordinates.")
    x, y = ALIGNMENT_POINTS[:, 0], ALIGNMENT_POINTS[:, 1]
    ones, zeros = np.ones(5), np.zeros(5)
    design = np.vstack(
        (np.column_stack((x, y, ones, zeros)), np.column_stack((y, -x, zeros, ones)))
    )
    values = np.concatenate((source[:, 0], source[:, 1]))
    a, b, tx, ty = np.linalg.lstsq(design, values, rcond=-1)[0]
    if a * a + b * b < 1e-8:
        raise ValueError("Face landmarks are degenerate.")
    return cv2.invertAffineTransform(np.array([[a, b, tx], [-b, a, ty]]))


def expression_blob(frame: np.ndarray, face: Face) -> np.ndarray:
    transform = alignment_matrix(face.landmarks)
    aligned = cv2.warpAffine(
        frame,
        transform,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Preserve the exact operation order used by the upstream model demo.
    rgb /= 255.0
    rgb -= 0.5
    rgb /= 0.5
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def expression_from_logits(logits: np.ndarray) -> Expression:
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if scores.size != len(EXPRESSION_LABELS) or not np.isfinite(scores).all():
        raise RuntimeError("Expression model returned invalid class scores.")
    exponentials = np.exp(scores - scores.max())
    probabilities = exponentials / exponentials.sum()
    class_id = int(np.argmax(probabilities))
    return Expression(
        EXPRESSION_LABELS[class_id],
        float(probabilities[class_id]),
        tuple(map(float, probabilities)),
    )


class ExpressionClassifier:
    """Classify aligned faces locally, using the same CPU/CUDA selection as YuNet."""

    def __init__(self, model_path: Path, *, gpu: bool = False):
        if not verify_expression_model(Path(model_path)):
            raise ValueError("Expression model missing or checksum invalid. Run `veil init`.")
        self.backend = "CUDA" if gpu else "CPU"
        self.session = _create_session(model_path, gpu=gpu, size=112)
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, frame: np.ndarray, face: Face) -> Expression:
        try:
            blob = expression_blob(frame, face)
        except ValueError:
            # A valid face box must still be covered if its landmarks are unusable.
            return Expression("neutral", 1 / 7, (1 / 7,) * 7)
        logits = self.session.run(None, {self.input_name: blob})[0]
        return expression_from_logits(logits)

    def classify_many(
        self,
        frame: np.ndarray,
        faces: list[Face],
        *,
        check_cancel: Callable[[], None] | None = None,
    ) -> list[Expression]:
        # The pinned upstream graph has fixed batch size one.
        predictions = []
        for face in faces:
            if check_cancel is not None:
                check_cancel()
            predictions.append(self.classify(frame, face))
        return predictions
