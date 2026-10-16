import json
import shutil
import subprocess

import numpy as np
import pytest

from veil import pipeline
from veil.media import probe

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


class FakeDetector:
    backend = "CPU"

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        return [(70, 35, 45, 45)]


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "input.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=192x128:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-metadata",
            "location=+52.0+021.0/",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def frames(path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-threads",
            "1",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    media = probe(path)
    return np.frombuffer(result.stdout, np.uint8).reshape(-1, media["height"], media["width"], 3)


def run_pipeline(source, output, monkeypatch, **kwargs):
    monkeypatch.setattr(pipeline, "FaceDetector", FakeDetector)
    return pipeline.process_video(
        source,
        output,
        mode="cpu",
        progress=lambda value: None,
        cancelled=lambda: False,
        model_path=source,
        **kwargs,
    )


def test_masks_pixels_preserves_audio_duration_and_strips_location(source, tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(pipeline, "FaceDetector", FakeDetector)
    output = tmp_path / "output.mp4"
    result = pipeline.process_video(
        source,
        output,
        mode="cpu",
        progress=events.append,
        cancelled=lambda: False,
        model_path=source,
    )
    media = probe(output)
    assert result["frames_processed"] == 10
    assert result["frames_with_faces"] == 10
    assert media["has_audio"]
    assert media["duration"] == pytest.approx(1, abs=0.05)
    assert media["codec"] == "h264"
    assert result["backend"] == "CPU + libx264"
    original, masked = frames(source), frames(output)
    assert (
        np.mean(
            np.abs(
                original[:, 40:75, 75:105].astype(float) - masked[:, 40:75, 75:105].astype(float)
            )
        )
        > 10
    )
    metadata = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(output)],
        capture_output=True,
        check=True,
    )
    assert "location" not in json.loads(metadata.stdout)["format"].get("tags", {})
    assert events[0]["stage"] == "probing"
    assert events[-1]["stage"] == "finalizing"
    assert all(0 <= event["progress"] < 100 for event in events)


def test_rotation_and_cancellation_cleanup(source, tmp_path, monkeypatch):
    rotated = tmp_path / "rotated.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-display_rotation",
            "90",
            "-i",
            str(source),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    output = tmp_path / "rotated-output.mp4"
    result = run_pipeline(rotated, output, monkeypatch)
    assert (result["width"], result["height"]) == (128, 192)
    assert probe(output)["rotation"] == 0
    assert len(frames(output)) == 10

    processes = []
    real_popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(pipeline.subprocess, "Popen", capture)
    count = 0

    def cancel():
        nonlocal count
        count += 1
        return count > 15

    output = tmp_path / "cancelled.mp4"
    with pytest.raises(pipeline.CancelledError):
        pipeline.process_video(
            source,
            output,
            mode="cpu",
            progress=lambda value: None,
            cancelled=cancel,
            model_path=source,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".veil-*"))
    assert all(process.poll() is not None for process in processes)


def test_gpu_mode_never_silently_uses_cpu(source, tmp_path, monkeypatch):
    def failed_cuda(*args, **kwargs):
        raise RuntimeError("CUDA missing")

    monkeypatch.setattr(pipeline, "FaceDetector", failed_cuda)
    with pytest.raises(RuntimeError, match="GPU mode requires working CUDA"):
        pipeline.process_video(
            source,
            tmp_path / "output.mp4",
            mode="gpu",
            progress=lambda value: None,
            cancelled=lambda: False,
            model_path=source,
        )


def test_auto_mode_reports_cpu_fallback(source, tmp_path, monkeypatch):
    def detector(*args, gpu=False, **kwargs):
        if gpu:
            raise RuntimeError("CUDA missing")
        return FakeDetector()

    monkeypatch.setattr(pipeline, "FaceDetector", detector)
    monkeypatch.setattr(pipeline, "_nvenc_check", lambda: (False, "No NVENC"))
    events = []
    result = pipeline.process_video(
        source,
        tmp_path / "output.mp4",
        mode="auto",
        progress=events.append,
        cancelled=lambda: False,
        model_path=source,
    )
    assert result["backend"] == "CPU + libx264"
    assert any("using CPU inference" in e.get("message", "") for e in events)


def test_hevc_10bit_hdr_is_tone_mapped_to_sdr(tmp_path, monkeypatch):
    source = tmp_path / "iphone-hlg.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=192x128:rate=10:duration=0.6",
            "-vf",
            "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-threads",
            "1",
            "-x265-params",
            "pools=1:frame-threads=1:log-level=error:"
            "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    assert probe(source)["hdr"]
    output = tmp_path / "sdr.mp4"
    result = run_pipeline(source, output, monkeypatch)
    assert result["hdr_converted_to_sdr"]
    assert result["frames_processed"] == 6
    assert not probe(output)["hdr"]
    metadata = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
        check=True,
        capture_output=True,
    )
    video = json.loads(metadata.stdout)["streams"][0]
    assert video["pix_fmt"] == "yuv420p"
    assert video["color_space"] == "bt709"
    assert video["color_transfer"] == "bt709"
    assert video["color_primaries"] == "bt709"
    assert frames(output).mean() > 10


def test_audio_offset_is_preserved_when_video_start_is_nonzero(tmp_path, monkeypatch):
    source = tmp_path / "offset.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=192x128:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.7",
            "-vf",
            "setpts=PTS+2/TB",
            "-af",
            "asetpts=PTS+2.3/TB",
            "-copyts",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    assert probe(source)["start_time"] == pytest.approx(2)
    output = tmp_path / "offset-output.mp4"
    run_pipeline(source, output, monkeypatch)
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(result.stdout, np.float32)
    assert len(samples) / 48000 == pytest.approx(1, abs=0.04)
    assert np.abs(samples[: int(0.25 * 48000)]).max() < 0.01
    assert np.abs(samples[int(0.35 * 48000) : int(0.6 * 48000)]).max() > 0.05
    assert probe(output)["duration"] == pytest.approx(1, abs=0.04)


def test_variable_frame_rate_preserves_wall_clock_duration(tmp_path, monkeypatch):
    source = tmp_path / "vfr.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=192x128:rate=12:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.42",
            "-vf",
            "setpts='if(lt(N,6),N/(12*TB),(0.5+(N-6)/6)/TB)'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    info = probe(source)
    assert info["fps"] != 12
    output = tmp_path / "cfr-output.mp4"
    result = run_pipeline(source, output, monkeypatch)
    assert result["duration"] == pytest.approx(info["duration"], abs=1 / info["fps"])
    assert probe(output)["has_audio"]
    assert probe(output)["duration"] == pytest.approx(info["duration"], abs=1 / info["fps"])


def test_iphone_240fps_slow_motion_is_accepted(tmp_path, monkeypatch):
    source = tmp_path / "slowmo.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=192x128:rate=240:duration=0.1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    assert probe(source)["fps"] == 240
    result = run_pipeline(source, tmp_path / "slowmo-output.mp4", monkeypatch)
    assert result["frames_processed"] == 24
