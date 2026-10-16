"""Explicit, fail-closed configuration for a single local deployment."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_path: Path
    password_hash: str
    secret_key: str
    max_upload_bytes: int = 8 * 1024**3
    max_queue: int = 10
    session_seconds: int = 12 * 3600
    secure_cookie: bool = False
    mode: str = "auto"
    trusted_origin: str = ""
    trusted_proxies: str = "127.0.0.1"
    host: str = "127.0.0.1"
    port: int = 8765
    expression_model_path: Path | None = None

    @property
    def expression_path(self) -> Path:
        return self.expression_model_path or self.model_path.with_name("expression.onnx")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        password_hash = os.environ.get("VEIL_PASSWORD_HASH", "")
        secret = os.environ.get("VEIL_SECRET_KEY", "")
        if not password_hash.startswith("$argon2id$") or len(secret) < 32:
            raise ValueError(
                "Run 'veil init' first: a password hash and session secret are required."
            )
        mode = os.environ.get("VEIL_MODE", "auto")
        if mode not in {"auto", "gpu", "cpu"}:
            raise ValueError("VEIL_MODE must be auto, gpu, or cpu.")
        maximum = int(os.environ.get("VEIL_MAX_UPLOAD_GB", "8"))
        if not 1 <= maximum <= 100:
            raise ValueError("VEIL_MAX_UPLOAD_GB must be between 1 and 100.")
        return cls(
            data_dir=Path(os.environ.get("VEIL_DATA_DIR", "data")).resolve(),
            model_path=Path(os.environ.get("VEIL_MODEL_PATH", "models/yunet.onnx")).resolve(),
            password_hash=password_hash,
            secret_key=secret,
            max_upload_bytes=maximum * 1024**3,
            secure_cookie=os.environ.get("VEIL_SECURE_COOKIE", "false").lower() == "true",
            mode=mode,
            trusted_origin=os.environ.get("VEIL_ORIGIN", "").rstrip("/"),
            trusted_proxies=os.environ.get("VEIL_TRUSTED_PROXIES", "127.0.0.1"),
            host=os.environ.get("VEIL_HOST", "127.0.0.1"),
            port=int(os.environ.get("VEIL_PORT", "8765")),
            expression_model_path=Path(os.environ["VEIL_EXPRESSION_MODEL_PATH"]).resolve()
            if os.environ.get("VEIL_EXPRESSION_MODEL_PATH")
            else None,
        )
