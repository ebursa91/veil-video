import json
import shutil
import subprocess
import threading
import weakref

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


def test_inference_overlaps_ordered_encoding(source, tmp_path, monkeypatch):
    second_started = threading.Event()
    release_second = threading.Event()
    detected, written, references = [], [], []
    read = pipeline._read_frame
    write = pipeline._write_frame
    read_count = 0

    def indexed_read(*args):
        nonlocal read_count
        raw = read(*args)
        if raw:
            raw[0] = read_count
            read_count += 1
        return raw

    class Detector(FakeDetector):
        def detect(self, frame):
            references.append(weakref.ref(frame))
            assert sum(ref() is not None for ref in references) <= 2
            index = int(frame[0, 0, 0])
            detected.append(index)
            if index == 1:
                second_started.set()
                assert release_second.wait(2)
            return [(70 + index, 35, 30, 30)]

    def ordered_write(pipe, buffer, cancelled):
        if not written:
            overlapped = second_started.wait(2)
            release_second.set()
            assert overlapped, "next inference should run before the current write completes"
        written.append(int(buffer[0]))
        return write(pipe, buffer, cancelled)

    monkeypatch.setattr(pipeline, "FaceDetector", Detector)
    monkeypatch.setattr(pipeline, "_read_frame", indexed_read)
    monkeypatch.setattr(pipeline, "_write_frame", ordered_write)
    try:
        result = pipeline.process_video(
            source,
            tmp_path / "overlapped.mp4",
            mode="cpu",
            progress=lambda _: None,
            cancelled=lambda: False,
            model_path=source,
        )
    finally:
        release_second.set()
    assert detected == written == list(range(10))
    assert result["frames_with_faces"] == result["frames_processed"] == 10


def test_prefetched_inference_failure_reaps_processes_and_removes_export(
    source,
    tmp_path,
    monkeypatch,
):
    processes = []
    popen = pipeline.subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    class Detector(FakeDetector):
        calls = 0

        def detect(self, frame):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError("injected inference failure")
            return super().detect(frame)

    monkeypatch.setattr(pipeline.subprocess, "Popen", capture)
    monkeypatch.setattr(pipeline, "FaceDetector", Detector)
    output = tmp_path / "failed.mp4"
    with pytest.raises(RuntimeError, match="injected inference failure"):
        pipeline.process_video(
            source,
            output,
            mode="cpu",
            progress=lambda _: None,
            cancelled=lambda: False,
            model_path=source,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".veil-*"))
    assert all(process.poll() is not None for process in processes)
    assert not any(t.name.startswith("veil-prefetch") for t in threading.enumerate())


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


def test_emoji_classifies_original_then_draws_over_blur(source, tmp_path, monkeypatch):
    from veil import emoji
    from veil.detector import Face, Landmarks

    class Detector(FakeDetector):
        def detect_faces(self, frame):
            return [Face((70, 35, 45, 45), Landmarks(*([(80, 50)] * 5)), 0.9)]

    original_regions = []

    class Classifier:
        backend = "CPU"

        def __init__(self, path, *, gpu):
            assert not gpu

        def classify_many(self, frame, faces, **kwargs):
            original_regions.append(frame[40:75, 75:105].copy())
            return []

    class Tracker:
        def __init__(self, fps, *, seed):
            pass

        def update(self, faces, expressions):
            return ["overlay"]

    drawn = []

    def draw(frame, overlays):
        assert overlays == ["overlay"]
        # Inference saw the original; the renderer receives a masked image.
        original = original_regions[len(drawn)]
        assert np.abs(original.astype(float) - frame[40:75, 75:105].astype(float)).mean() > 10
        drawn.append(True)
        frame[50:65, 85:100] = (0, 220, 255)

    monkeypatch.setattr(pipeline, "FaceDetector", Detector)
    monkeypatch.setattr(pipeline, "ExpressionClassifier", Classifier)
    monkeypatch.setattr(emoji, "EmojiTracker", Tracker)
    monkeypatch.setattr(emoji, "draw_emojis", draw)
    output = tmp_path / "emoji.mp4"
    result = pipeline.process_video(
        source,
        output,
        mode="cpu",
        effect="emoji",
        model_path=source,
        progress=lambda _: None,
        cancelled=lambda: False,
    )
    assert result["effect"] == "emoji"
    assert len(drawn) == result["frames_with_faces"] == result["frames_processed"] == 10
    assert probe(output)["has_audio"]
    assert frames(output)[:, 55:60, 90:95, 1:].mean() > 180


@pytest.mark.parametrize("failure", ["classifier", "renderer"])
def test_emoji_failure_removes_partial_export(source, tmp_path, monkeypatch, failure):
    from veil import emoji

    class Detector(FakeDetector):
        def detect_faces(self, frame):
            return []

    class Classifier:
        backend = "CPU"

        def __init__(self, *args, **kwargs):
            pass

        def classify_many(self, frame, faces, **kwargs):
            if failure == "classifier":
                raise RuntimeError("expression failed")
            return []

    def draw(*args):
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(pipeline, "FaceDetector", Detector)
    monkeypatch.setattr(pipeline, "ExpressionClassifier", Classifier)
    monkeypatch.setattr(emoji, "draw_emojis", draw)
    output = tmp_path / "failed-emoji.mp4"
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.process_video(
            source,
            output,
            mode="cpu",
            effect="emoji",
            model_path=source,
            progress=lambda _: None,
            cancelled=lambda: False,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".veil-*"))
    assert not any(t.name.startswith("veil-prefetch") for t in threading.enumerate())
