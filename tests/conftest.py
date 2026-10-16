from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from veil.app import create_app
from veil.auth import HASHER
from veil.config import Settings


@pytest.fixture(scope="session")
def password_hash():
    return HASHER.hash("correct horse battery staple")


@pytest.fixture
def settings(tmp_path: Path, password_hash):
    model = tmp_path / "test-model.onnx"
    model.write_bytes(b"test model only; never used for processing")
    return Settings(
        data_dir=tmp_path / "data",
        model_path=model,
        password_hash=password_hash,
        secret_key="s" * 48,
        max_upload_bytes=1024 * 1024,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, start_worker=False)) as client:
        yield client


@pytest.fixture
def signed_in(client):
    response = client.post("/api/login", json={"password": "correct horse battery staple"})
    assert response.status_code == 200
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return client
