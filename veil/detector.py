"""YuNet ONNX inference, conservative masking, and short dropout protection.

YuNet output decoding follows OpenCV's FaceDetectorYN implementation:
https://github.com/opencv/opencv/blob/4.11.0/modules/objdetect/src/face_detect.cpp
The weights are MIT licensed; see docs/licenses/YuNet-MIT.txt.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from veil.model import verify_model


class FaceDetector:
    """Detect every frame; a CUDA session must actually execute CUDA nodes."""

    def __init__(self, model_path: Path, *, gpu: bool = False, threshold: float = 0.55):
        if not verify_model(Path(model_path)):
            raise ValueError("YuNet model missing or checksum invalid. Run `veil init`.")
        self.threshold = threshold
        self.backend = "CUDA" if gpu else "CPU"
        if gpu and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CUDA provider is not installed.")
        if gpu and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        with tempfile.TemporaryDirectory(prefix="veil-ort-") as profiling:
            if gpu:
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(profiling) / "cuda")
            providers = (
                [
                    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                    "CPUExecutionProvider",
                ]
                if gpu
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(str(model_path), options, providers=providers)
            self.session.disable_fallback()
            if gpu and "CUDAExecutionProvider" not in self.session.get_providers():
                raise RuntimeError("CUDA initialization failed; ONNX Runtime fell back to CPU.")
            self.input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape
            self.size = int(shape[2]) if isinstance(shape[2], int) else 640
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.session.run(
                None, {self.input_name: np.zeros((1, 3, self.size, self.size), np.float32)}
            )
            if gpu:
                profile = Path(self.session.end_profiling())
                events = json.loads(profile.read_text())
                if not any(
                    e.get("args", {}).get("provider") == "CUDAExecutionProvider" for e in events
                ):
                    raise RuntimeError("CUDA did not execute any model operations.")

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        height, width = frame.shape[:2]
        scale = self.size / max(width, height)
        resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        image = np.zeros((self.size, self.size, 3), np.uint8)
        image[: resized.shape[0], : resized.shape[1]] = resized
        # Pack NCHW here so ONNX Runtime does not copy strided HWC storage again.
        blob = image.transpose(2, 0, 1)[None].astype(np.float32, order="C")
        outputs = dict(
            zip(self.output_names, self.session.run(None, {self.input_name: blob}), strict=True)
        )
        return decode_predictions(outputs, self.size, scale, self.threshold)


def decode_predictions(outputs: dict, size: int, scale: float, threshold: float) -> list:
    boxes, scores = [], []
    for stride in (8, 16, 32):
        cls = np.asarray(outputs[f"cls_{stride}"]).reshape(-1)
        obj = np.asarray(outputs[f"obj_{stride}"]).reshape(-1)
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        bbox = np.asarray(outputs[f"bbox_{stride}"]).reshape(-1, 4)
        indices = np.flatnonzero(score >= threshold)
        for index in indices:
            row, col = divmod(int(index), size // stride)
            cx, cy = (np.array([col, row]) + bbox[index, :2]) * stride
            w, h = np.exp(np.clip(bbox[index, 2:], -10, 10)) * stride
            box = [
                float((cx - w / 2) / scale),
                float((cy - h / 2) / scale),
                float(w / scale),
                float(h / scale),
            ]
            if np.isfinite(box).all():
                boxes.append(box)
                scores.append(float(score[index]))
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.3, top_k=5000)
    return [tuple(boxes[int(i)]) for i in np.asarray(keep).reshape(-1)]


class MaskTracker:
    """Hold lost masks briefly and mask the swept region between matching faces.

    This reduces one-frame leaks, but does not guarantee anonymity. New, small,
    occluded, or profile faces may still be missed and need human review.
    """

    def __init__(self, fps: float):
        self.hold = max(2, round(fps * 0.25))
        self.tracks: list[tuple[tuple, int]] = []

    def update(self, boxes: list[tuple]) -> list[tuple]:
        masks = list(boxes)
        unused = set(range(len(self.tracks)))
        next_tracks = []
        for box in boxes:
            x, y, w, h = box
            best, distance = None, float("inf")
            for index in unused:
                previous, _ = self.tracks[index]
                px, py, pw, ph = previous
                candidate = ((x + w / 2 - px - pw / 2) ** 2 + (y + h / 2 - py - ph / 2) ** 2) ** 0.5
                if candidate < max(w, h, pw, ph) and candidate < distance:
                    best, distance = index, candidate
            if best is not None:
                previous, _ = self.tracks[best]
                px, py, pw, ph = previous
                left, top = min(x, px), min(y, py)
                masks.append((left, top, max(x + w, px + pw) - left, max(y + h, py + ph) - top))
                unused.remove(best)
            next_tracks.append((box, 0))
        for index in unused:
            box, missed = self.tracks[index]
            if missed < self.hold:
                masks.append(box)
                next_tracks.append((box, missed + 1))
        self.tracks = next_tracks
        return masks


def blur_faces(frame: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """Strong pixelation plus Gaussian blur in expanded rectangular face regions."""
    height, width = frame.shape[:2]
    for x, y, w, h in boxes:
        # Include forehead, chin, ears and box jitter; rectangular coverage avoids
        # the unmasked corners left by an ellipse on angled faces.
        x1 = max(0, int(np.floor(x - w * 0.35)))
        y1 = max(0, int(np.floor(y - h * 0.45)))
        x2 = min(width, int(np.ceil(x + w * 1.35)))
        y2 = min(height, int(np.ceil(y + h * 1.3)))
        if x2 <= x1 or y2 <= y1:
            continue
        region = frame[y1:y2, x1:x2]
        # At most 8x8 samples regardless of recording resolution.
        tiny = cv2.resize(region, (min(8, x2 - x1), min(8, y2 - y1)), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(pixelated, (0, 0), sigmaX=max(3, w * 0.09))
    return frame
