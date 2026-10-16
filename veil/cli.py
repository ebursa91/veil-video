"""Local setup and operation. No credentials passed on command lines."""

import argparse
import getpass
import json
import os
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv

from veil.auth import HASHER


def initialize(env_file: Path, *, no_model: bool = False) -> None:
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Using existing configuration: {env_file}")
        from veil.config import Settings

        Settings.from_env()
    else:
        password = getpass.getpass("Choose a password (at least 12 characters): ")
        if len(password) < 12:
            raise ValueError("Use at least 12 characters.")
        if password != getpass.getpass("Confirm password: "):
            raise ValueError("Passwords do not match.")
        password_hash = HASHER.hash(password)
        secret_key = secrets.token_urlsafe(48)
        env_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(env_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as target:
            target.write(
                "# Veil local configuration. Never commit this file.\n"
                f"VEIL_PASSWORD_HASH='{password_hash}'\n"
                f"VEIL_SECRET_KEY='{secret_key}'\n"
                "VEIL_DATA_DIR=data\nVEIL_MODEL_PATH=models/yunet.onnx\n"
                "VEIL_MODE=auto\nVEIL_MAX_UPLOAD_GB=8\nVEIL_SECURE_COOKIE=false\n"
                "VEIL_RETENTION_DAYS=7\n"
            )
        load_dotenv(env_file)
        print(f"Configuration saved to {env_file} (owner access only).")
    if not no_model:
        from veil.model import ensure_expression_model, ensure_model

        path = Path(os.environ.get("VEIL_MODEL_PATH", "models/yunet.onnx"))
        print("Preparing the pinned face detector…")
        ensure_model(path)
        print(f"Face model ready: {path}")
        expression_path = Path(
            os.environ.get("VEIL_EXPRESSION_MODEL_PATH", path.with_name("expression.onnx"))
        )
        print("Preparing the pinned expression model…")
        ensure_expression_model(expression_path)
        print(f"Expression model ready: {expression_path}")
    print("Start with: veil serve")


def main() -> None:
    parser = argparse.ArgumentParser(description="Veil: private GPU face anonymization")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="Create password configuration and download face models"
    )
    init.add_argument(
        "--no-model", action="store_true", help="Configure without downloading the model"
    )
    serve = commands.add_parser("serve", help="Start the local web server and one video worker")
    serve.add_argument("--host", help="Listener address (default: VEIL_HOST or 127.0.0.1)")
    serve.add_argument("--port", type=int, help="Listener port (default: VEIL_PORT or 8765)")
    commands.add_parser("doctor", help="Check FFmpeg, face model, CUDA inference and NVENC")
    args = parser.parse_args()
    try:
        # Environment variables (e.g. Docker mounts) take precedence over .env.
        load_dotenv(args.env_file)
        if args.command == "init":
            initialize(args.env_file, no_model=args.no_model)
        elif args.command == "doctor":
            from veil.pipeline import capabilities

            result = capabilities(Path(os.environ.get("VEIL_MODEL_PATH", "models/yunet.onnx")))
            print(json.dumps(result, indent=2))
        else:
            from veil.config import Settings

            settings = Settings.from_env()
            import uvicorn

            from veil.app import create_app

            # Never use multiple Uvicorn workers: the queue owns one GPU processor.
            uvicorn.run(
                create_app(settings),
                host=args.host or settings.host,
                port=args.port if args.port is not None else settings.port,
                proxy_headers=True,
                forwarded_allow_ips=settings.trusted_proxies,
            )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Veil: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
