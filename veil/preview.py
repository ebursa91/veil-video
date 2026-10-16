"""Small, optional JPEG snapshots of frames already accepted by the encoder."""

import logging
import time
from collections.abc import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)
PreviewCallback = Callable[[bytes, int, float], None]


def encode_jpeg(frame: np.ndarray) -> bytes:
    """Resize without upscaling; never retain the caller's video-frame buffer."""
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    if scale < 1:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


class ProcessedPreview:
    """Throttle encoding as well as callbacks; keep no raw image or JPEG history.

    Frame numbers are one-based. Timestamps refer to the first displayed instant
    of the frame in the exported video's constant frame rate timeline.
    """

    def __init__(self, callback: PreviewCallback, fps: float):
        self.callback = callback
        self.fps = fps
        self.enabled = True
        self.last_emit_at: float | None = None
        self.last_frame_no = 0

    def emit(
        self, frame: np.ndarray, frame_no: int, *, now: float | None = None, force: bool = False
    ) -> None:
        if not self.enabled or frame_no == self.last_frame_no:
            return
        now = time.monotonic() if now is None else now
        if not force and self.last_emit_at is not None and now - self.last_emit_at < 1:
            return
        try:
            jpeg = encode_jpeg(frame)
            self.callback(jpeg, frame_no, (frame_no - 1) / self.fps)
        except Exception as exc:
            self.enabled = False
            # Avoid retaining an image through an exception traceback in logging
            # handlers, and never include video content or callback arguments.
            logger.warning("Live preview disabled after capture failure (%s).", type(exc).__name__)
            return
        self.last_emit_at = now
        self.last_frame_no = frame_no
