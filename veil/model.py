"""Explicit, integrity-checked installation of the MIT-licensed YuNet weights."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/"
    f"{MODEL_REVISION}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_BYTES = 232589


def verify_model(path: Path) -> bool:
    """Do not execute a substituted or incomplete model."""
    if not path.is_file() or path.stat().st_size != MODEL_BYTES:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256


def ensure_model(path: Path) -> Path:
    """Download only from the explicit setup command, never from an upload job."""
    path = Path(path)
    if verify_model(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Veil-model-installer"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
                temporary = Path(target.name)
                # A bounded read also rejects Git LFS pointers and unexpected responses.
                data = response.read(MODEL_BYTES + 1)
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        if not verify_model(temporary):
            raise ValueError("YuNet model checksum mismatch; no model was installed.")
        os.replace(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
