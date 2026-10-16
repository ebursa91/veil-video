"""Explicit, integrity-checked installation of the locally executed models."""

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
EXPRESSION_MODEL_REVISION = MODEL_REVISION
EXPRESSION_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/"
    f"{EXPRESSION_MODEL_REVISION}/models/facial_expression_recognition/"
    "facial_expression_recognition_mobilefacenet_2022july.onnx"
)
EXPRESSION_MODEL_SHA256 = "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9"
EXPRESSION_MODEL_BYTES = 4791892


def verify_model(path: Path) -> bool:
    """Do not execute a substituted or incomplete model."""
    return _verify_pinned_model(Path(path), MODEL_BYTES, MODEL_SHA256)


def verify_expression_model(path: Path) -> bool:
    return _verify_pinned_model(Path(path), EXPRESSION_MODEL_BYTES, EXPRESSION_MODEL_SHA256)


def _verify_pinned_model(path: Path, size: int, checksum: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def ensure_model(path: Path) -> Path:
    """Download only from the explicit setup command, never from an upload job."""
    return _ensure_pinned_model(path, MODEL_URL, MODEL_BYTES, MODEL_SHA256, "YuNet")


def ensure_expression_model(path: Path) -> Path:
    """Install the Apache-2.0 expression model through explicit setup only."""
    return _ensure_pinned_model(
        path,
        EXPRESSION_MODEL_URL,
        EXPRESSION_MODEL_BYTES,
        EXPRESSION_MODEL_SHA256,
        "Expression",
    )


def _ensure_pinned_model(path: Path, url: str, size: int, checksum: str, name: str) -> Path:
    path = Path(path)
    if _verify_pinned_model(path, size, checksum):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Veil-model-installer"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
                temporary = Path(target.name)
                # A bounded read also rejects Git LFS pointers and unexpected responses.
                data = response.read(size + 1)
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        if not _verify_pinned_model(temporary, size, checksum):
            raise ValueError(f"{name} model checksum mismatch; no model was installed.")
        os.replace(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
