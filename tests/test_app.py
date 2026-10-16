import time
from dataclasses import replace

from fastapi.testclient import TestClient

from veil.app import create_app
from veil.jobs import JobManager
from veil.store import Store


def upload(client, **kwargs):
    return client.post(
        "/api/jobs",
        files={"file": ("iPhone.MOV", b"video contents", "video/quicktime")},
        data={"mode": "cpu"},
        **kwargs,
    )


def test_auth_protects_all_private_routes_before_body_parse(client):
    for path in (
        "/api/jobs",
        "/api/session",
        "/api/system",
        "/api/events",
        "/api/jobs/invalid/preview",
        "/api/jobs/invalid/download",
    ):
        assert client.get(path).status_code == 401
    assert client.post("/api/jobs", content=b"x" * 100).status_code == 401
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/data/veil.sqlite3").status_code == 404


def test_password_cookie_and_csrf(client):
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
    result = client.post("/api/login", json={"password": "correct horse battery staple"})
    assert result.status_code == 200
    assert "HttpOnly" in result.headers["set-cookie"]
    assert "SameSite=strict" in result.headers["set-cookie"]
    assert client.get("/api/session").json()["authenticated"]
    assert upload(client).status_code == 403
    client.headers["X-CSRF-Token"] = result.json()["csrf_token"]
    assert upload(client).status_code == 201
    assert client.post("/api/logout").status_code == 200
    assert client.get("/api/jobs").status_code == 401


def test_throttle(client):
    for _ in range(10):
        assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 429


def test_origin_guard(signed_in):
    assert upload(signed_in, headers={"Origin": "https://evil.test"}).status_code == 403
    assert upload(signed_in, headers={"Origin": "http://testserver"}).status_code == 201


def test_queue_lifecycle_and_private_original(signed_in):
    response = upload(signed_in)
    assert response.status_code == 201
    job = response.json()
    assert job["status"] == "queued"
    assert job["filename"] == "iPhone.MOV"
    assert len(signed_in.get("/api/jobs").json()["jobs"]) == 1
    path = f"/api/jobs/{job['id']}"
    assert signed_in.get(path + "/download").status_code == 409
    assert signed_in.get(path + "/source").status_code == 404
    assert signed_in.delete(path).status_code == 409
    assert signed_in.post(path + "/cancel").json()["status"] == "cancelled"
    assert signed_in.delete(path).status_code == 200
    assert signed_in.get(path).status_code == 404


def test_download_is_range_capable(signed_in, settings):
    job = upload(signed_in).json()
    store = Store(settings.data_dir)
    (store.directory(job["id"]) / "blurred.mp4").write_bytes(b"0123456789")
    store.update(job["id"], status="completed")
    path = f"/api/jobs/{job['id']}"
    response = signed_in.get(path + "/preview", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert "inline" in response.headers["content-disposition"]
    assert "attachment" in signed_in.get(path + "/download").headers["content-disposition"]


def test_upload_validation_and_limits(signed_in, settings):
    assert signed_in.post("/api/jobs", files={"file": ("bad.txt", b"hello")}).status_code == 415
    assert signed_in.post("/api/jobs", files={"file": ("empty.mov", b"")}).status_code == 422
    assert (
        signed_in.post("/api/jobs", files={"file": ("big.mov", b"x" * (1024**2 + 1))}).status_code
        == 413
    )
    assert (
        signed_in.post(
            "/api/jobs", files={"file": ("x.mov", b"x")}, data={"mode": "wat"}
        ).status_code
        == 422
    )
    assert not list((settings.data_dir / ".uploads").glob("*"))


def test_chunked_limit(signed_in):
    def chunks():
        yield b'--veil\r\nContent-Disposition: form-data; name="file"; filename="x.mov"\r\n\r\n'
        for _ in range(20):
            yield b"x" * 100_000
        yield b"\r\n--veil--\r\n"

    response = signed_in.post(
        "/api/jobs",
        content=chunks(),
        headers={"Content-Type": "multipart/form-data; boundary=veil"},
    )
    assert response.status_code == 413


def test_job_filename_cannot_escape_storage(signed_in, settings):
    response = signed_in.post("/api/jobs", files={"file": ("../../private.MOV", b"video")})
    assert response.status_code == 201
    assert response.json()["filename"] == "private.MOV"
    assert signed_in.get("/api/jobs/../../etc/passwd/download").status_code == 404
    assert not (settings.data_dir / "private.MOV").exists()


def test_queue_capacity(signed_in):
    for _ in range(10):
        assert upload(signed_in).status_code == 201
    assert upload(signed_in).status_code == 409


def test_worker_restart_marks_interrupted_failed(settings):
    store = Store(settings.data_dir)
    job = store.create("interrupted.mov", 20, "cpu")
    store.update(job["id"], status="processing")
    manager = JobManager(settings, store)
    manager.start()
    manager.stop()
    assert store.get(job["id"])["status"] == "failed"


def test_worker_calls_pipeline_and_persists_result(settings, monkeypatch):
    import veil.media
    import veil.pipeline

    monkeypatch.setattr(veil.media, "probe", lambda _: {"width": 20, "height": 20})

    def process(source, output, **kwargs):
        kwargs["progress"]({"stage": "detecting", "progress": 55, "frames_processed": 3})
        output.write_bytes(b"processed")
        return {"frames_with_faces": 3, "backend": "CPU"}

    monkeypatch.setattr(veil.pipeline, "process_video", process)
    store = Store(settings.data_dir)
    job = store.create("phone.mov", 10, "cpu")
    directory = store.directory(job["id"])
    directory.mkdir()
    (directory / "source.bin").write_bytes(b"input")
    manager = JobManager(settings, store)
    manager.start()
    try:
        for _ in range(100):
            if store.get(job["id"])["status"] == "completed":
                break
            time.sleep(0.02)
        result = store.get(job["id"])
        assert result["status"] == "completed"
        assert result["progress"] == 100
        assert result["result"]["frames_with_faces"] == 3
    finally:
        manager.stop()


def test_secure_cookie(settings):
    with TestClient(
        create_app(replace(settings, secure_cookie=True), start_worker=False),
        base_url="https://testserver",
    ) as client:
        response = client.post("/api/login", json={"password": "correct horse battery staple"})
        assert "; Secure" in response.headers["set-cookie"]
        assert client.get("/api/session").status_code == 200
