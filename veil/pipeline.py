"""Streaming face blur jobs with bounded memory and cancellable FFmpeg pipes."""

import os
import selectors
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from veil.detector import FaceDetector, MaskTracker, blur_faces
from veil.media import INPUT_FORMATS, decode_filter, probe


class CancelledError(Exception):
    """A job was cancelled; partial output has been removed."""


def _nvenc_check() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=size=320x240:rate=25",
                "-frames:v",
                "1",
                "-an",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=20,
        )
        return result.returncode == 0, (
            "NVENC encode test passed."
            if result.returncode == 0
            else "NVENC unavailable: " + result.stderr.decode(errors="replace")[-600:].strip()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"NVENC unavailable: {exc}"


def capabilities(model_path: Path | None = None) -> dict:
    """Report executed hardware checks, never infer support from a GPU name."""
    nvenc, encoder_detail = _nvenc_check()
    gpu = False
    details = [encoder_detail]
    if model_path is None:
        details.append("CUDA inference not verified: initialize the model first.")
    else:
        try:
            detector = FaceDetector(model_path, gpu=True)
            gpu = True
            del detector
            details.append("CUDA model inference passed; profiling confirms CUDA node execution.")
        except Exception as exc:
            details.append(f"CUDA inference unavailable: {exc}")
    return {
        "gpu_available": gpu,
        "nvenc_available": nvenc,
        "backend": f"{'CUDA' if gpu else 'CPU'} + {'NVENC' if nvenc else 'libx264'}",
        "details": details,
    }


def _stop(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
    for pipe in (process.stdin, process.stdout):
        if pipe is not None:
            pipe.close()


def _check_cancel(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise CancelledError("Job cancelled.")


def _read_frame(pipe, length: int, cancelled: Callable[[], bool]) -> bytes:
    chunks = bytearray()
    deadline = time.monotonic() + 120
    with selectors.DefaultSelector() as selector:
        selector.register(pipe, selectors.EVENT_READ)
        while len(chunks) < length:
            _check_cancel(cancelled)
            if time.monotonic() > deadline:
                raise RuntimeError("Video decoding stalled for two minutes.")
            if not selector.select(timeout=0.2):
                continue
            try:
                chunk = os.read(pipe.fileno(), min(length - len(chunks), 1024 * 1024))
            except BlockingIOError:
                continue
            if not chunk:
                if chunks:
                    raise RuntimeError("FFmpeg returned an incomplete video frame.")
                return b""
            chunks.extend(chunk)
            deadline = time.monotonic() + 120
    return bytes(chunks)


def _write_frame(pipe, data: bytes, cancelled: Callable[[], bool]) -> None:
    view = memoryview(data)
    deadline = time.monotonic() + 120
    with selectors.DefaultSelector() as selector:
        selector.register(pipe, selectors.EVENT_WRITE)
        while view:
            _check_cancel(cancelled)
            if time.monotonic() > deadline:
                raise RuntimeError("Video encoding stalled for two minutes.")
            if not selector.select(timeout=0.2):
                continue
            try:
                written = os.write(pipe.fileno(), view[: 1024 * 1024])
            except BlockingIOError:
                continue
            view = view[written:]
            deadline = time.monotonic() + 120


def _wait(process: subprocess.Popen, cancelled: Callable[[], bool], timeout: int = 300) -> None:
    started = time.monotonic()
    while process.poll() is None:
        _check_cancel(cancelled)
        if time.monotonic() - started > timeout:
            raise RuntimeError("FFmpeg did not finish within the processing timeout.")
        time.sleep(0.1)
    if process.returncode:
        raise RuntimeError("FFmpeg could not process this video.")


def _error_tail(log) -> str:
    log.seek(0, os.SEEK_END)
    log.seek(max(0, log.tell() - 2500))
    return log.read().decode(errors="replace").strip()


def process_video(
    source: Path,
    output: Path,
    *,
    mode: str,
    progress: Callable[[dict], None],
    cancelled: Callable[[], bool],
    model_path: Path,
) -> dict:
    if mode not in {"auto", "gpu", "cpu"}:
        raise ValueError("Processing mode must be auto, gpu or cpu.")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError("Output must be a new file distinct from the source.")
    _check_cancel(cancelled)
    progress({"stage": "probing", "progress": 0, "message": "Inspecting video and hardware."})
    media = probe(source)
    _check_cancel(cancelled)
    fallback = []
    if mode == "cpu":
        detector = FaceDetector(model_path, gpu=False)
        nvenc = False
    else:
        try:
            detector = FaceDetector(model_path, gpu=True)
        except Exception as exc:
            if mode == "gpu":
                raise RuntimeError(f"GPU mode requires working CUDA inference: {exc}") from exc
            fallback.append(f"CUDA unavailable ({exc}); using CPU inference.")
            detector = FaceDetector(model_path, gpu=False)
        nvenc, reason = _nvenc_check()
        if not (144 <= media["width"] <= 4096 and 64 <= media["height"] <= 4096):
            nvenc, reason = False, "H.264 NVENC dimensions must be 144..4096 by 64..4096."
        if not nvenc:
            if mode == "gpu":
                raise RuntimeError(f"GPU mode requires working NVENC: {reason}")
            fallback.append("NVENC unavailable; using CPU encoding.")
    backend = f"{detector.backend} + {'NVENC' if nvenc else 'libx264'}"
    width, height, fps = media["width"], media["height"], media["fps"]
    total = max(1, round(media["duration"] * fps))
    tracker = MaskTracker(fps)
    frame_count = face_count = 0
    decoder = encoder = muxer = None
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_update = 0.0
    with tempfile.TemporaryDirectory(prefix=".veil-", dir=output.parent) as directory:
        intermediate = Path(directory) / "video.mp4"
        final = Path(directory) / "result.mp4"
        with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
            try:
                decode_command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-xerror",
                    "-threads",
                    "2",
                    "-filter_threads",
                    "2",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-format_whitelist",
                    INPUT_FORMATS,
                    "-i",
                    str(source),
                    "-map",
                    f"0:{media['video_stream_index']}",
                    "-an",
                    "-sn",
                    "-dn",
                    "-vf",
                    decode_filter(media),
                    "-fps_mode",
                    "passthrough",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "pipe:1",
                ]
                encode_command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-filter_threads",
                    "2",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-framerate",
                    f"{fps:.9f}",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "h264_nvenc" if nvenc else "libx264",
                ]
                encode_command += (
                    ["-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
                    if nvenc
                    else ["-preset", "fast", "-crf", "20", "-threads", "4"]
                )
                encode_command += [
                    "-pix_fmt",
                    "yuv420p",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-colorspace",
                    "bt709",
                    "-color_range",
                    "tv",
                    "-map_metadata",
                    "-1",
                    "-map_chapters",
                    "-1",
                    str(intermediate),
                ]
                decoder = subprocess.Popen(
                    decode_command,
                    stdout=subprocess.PIPE,
                    stderr=decode_log,
                    start_new_session=True,
                )
                encoder = subprocess.Popen(
                    encode_command, stdin=subprocess.PIPE, stderr=encode_log, start_new_session=True
                )
                os.set_blocking(decoder.stdout.fileno(), False)
                os.set_blocking(encoder.stdin.fileno(), False)
                progress(
                    {
                        "stage": "detecting",
                        "progress": 1,
                        "backend": backend,
                        "total_frames": total,
                        "frames_processed": 0,
                        "message": " ".join(fallback)
                        or "Detecting faces and encoding masked frames.",
                    }
                )
                while True:
                    raw = _read_frame(decoder.stdout, width * height * 3, cancelled)
                    if not raw:
                        break
                    # Protect disk/time use if a corrupt container lies about duration.
                    if frame_count > (4 * 60 * 60 + 1) * fps:
                        raise ValueError("Decoded video exceeds the four-hour limit.")
                    frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
                    boxes = detector.detect(frame)
                    masks = tracker.update(boxes)
                    if masks:
                        face_count += 1
                        blur_faces(frame, masks)
                    _write_frame(encoder.stdin, frame.tobytes(), cancelled)
                    frame_count += 1
                    now = time.monotonic()
                    if now - last_update >= 0.3:
                        speed = frame_count / max(0.001, now - started)
                        progress(
                            {
                                "stage": "detecting",
                                "progress": min(96, 1 + frame_count / total * 95),
                                "frames_processed": frame_count,
                                "total_frames": total,
                                "fps": round(speed, 1),
                                "eta_seconds": max(0, round((total - frame_count) / speed)),
                                "backend": backend,
                                "message": "Detecting faces and encoding masked frames.",
                            }
                        )
                        last_update = now
                _wait(decoder, cancelled)
                encoder.stdin.close()
                progress(
                    {
                        "stage": "encoding",
                        "progress": 97,
                        "frames_processed": frame_count,
                        "backend": backend,
                        "message": "Finishing video encoding.",
                        "eta_seconds": None,
                    }
                )
                _wait(encoder, cancelled)
                if not frame_count:
                    raise ValueError("The video contains no decodable frames.")
                progress(
                    {
                        "stage": "finalizing",
                        "progress": 98,
                        "frames_processed": frame_count,
                        "backend": backend,
                        "message": "Adding audio and preparing download.",
                        "eta_seconds": None,
                    }
                )
                # Raw frames use an explicit constant time base. Rebase the source
                # audio by the *video* start time, preserving relative audio delay.
                mux_command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-copyts",
                    "-i",
                    str(intermediate),
                    "-itsoffset",
                    f"{-media['start_time']:.9f}",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-format_whitelist",
                    INPUT_FORMATS,
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:v",
                    "copy",
                    "-bsf:v",
                    "h264_metadata=colour_primaries=1:transfer_characteristics=1:"
                    "matrix_coefficients=1:video_full_range_flag=0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-af",
                    "aresample=async=1:first_pts=0",
                    "-threads",
                    "2",
                    "-t",
                    f"{frame_count / fps:.9f}",
                    "-map_metadata",
                    "-1",
                    "-map_chapters",
                    "-1",
                    "-metadata:s:v:0",
                    "rotate=0",
                    "-movflags",
                    "+faststart",
                    str(final),
                ]
                muxer = subprocess.Popen(
                    mux_command,
                    stderr=encode_log,
                    stdout=subprocess.DEVNULL,
                    start_new_session=True,
                )
                _wait(muxer, cancelled)
                _check_cancel(cancelled)
                os.replace(final, output)
            except CancelledError:
                raise
            except Exception as exc:
                details = " ".join(filter(None, [_error_tail(decode_log), _error_tail(encode_log)]))
                if details:
                    raise RuntimeError(f"{exc} FFmpeg: {details}") from exc
                raise
            finally:
                for process in (decoder, encoder, muxer):
                    _stop(process)
    return {
        "frames_processed": frame_count,
        "frames_with_faces": face_count,
        "backend": backend,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
        "fps": fps,
        "has_audio": media["has_audio"],
        "hdr_converted_to_sdr": media["hdr"],
        "message": "Review the entire video before sharing; automatic detection can miss faces.",
    }
