import os

import pytest
from dotenv import dotenv_values

from veil.auth import Auth
from veil.cli import initialize
from veil.config import Settings


def test_init_creates_private_password_hash_and_preserves_docker_paths(tmp_path, monkeypatch):
    answers = iter(["correct horse battery staple"] * 2)
    monkeypatch.setattr("veil.cli.getpass.getpass", lambda _: next(answers))
    monkeypatch.setenv("VEIL_DATA_DIR", "/data")
    monkeypatch.setenv("VEIL_MODEL_PATH", "/models/yunet.onnx")
    env = tmp_path / ".env"
    initialize(env, no_model=True)
    values = dotenv_values(env)
    assert values["VEIL_PASSWORD_HASH"].startswith("$argon2id$")
    assert "correct horse" not in env.read_text()
    assert env.stat().st_mode & 0o777 == 0o600
    assert len(values["VEIL_SECRET_KEY"]) >= 32
    assert os.environ["VEIL_MODEL_PATH"] == "/models/yunet.onnx"
    assert os.environ["VEIL_DATA_DIR"] == "/data"


def test_init_refuses_short_or_mismatched_passwords(tmp_path, monkeypatch):
    monkeypatch.setattr("veil.cli.getpass.getpass", lambda _: "short")
    with pytest.raises(ValueError, match="12 characters"):
        initialize(tmp_path / ".env", no_model=True)
    assert not (tmp_path / ".env").exists()


def test_init_existing_configuration_is_not_replaced(tmp_path, settings, monkeypatch):
    env = tmp_path / "existing.env"
    env.write_text(
        f"VEIL_PASSWORD_HASH='{settings.password_hash}'\nVEIL_SECRET_KEY='{settings.secret_key}'\n"
    )
    monkeypatch.delenv("VEIL_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("VEIL_SECRET_KEY", raising=False)
    original = env.read_bytes()
    initialize(env, no_model=True)
    assert env.read_bytes() == original


def test_missing_configuration_fails_closed(monkeypatch):
    monkeypatch.setattr("veil.config.load_dotenv", lambda: None)
    monkeypatch.delenv("VEIL_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("VEIL_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="veil init"):
        Settings.from_env()


def test_password_rotation_invalidates_signed_sessions(settings):
    from dataclasses import replace

    from veil.auth import HASHER

    cookie, _ = Auth(settings).issue()
    rotated = replace(settings, password_hash=HASHER.hash("different long password"))
    assert Auth(rotated).read(cookie) is None
