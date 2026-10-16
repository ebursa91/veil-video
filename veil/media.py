"""Bounded local media inspection and explicit SDR decoding filters."""

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

MAX_DURATION = 4 * 60 * 60
MAX_PIXELS = 7680 * 4320
INPUT_FORMATS = "mov,matroska,webm"


def probe(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("The uploaded video is empty or missing.")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                INPUT_FORMATS,
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        data = json.loads(result.stdout)
        formats = set(data.get("format", {}).get("format_name", "").split(","))
        if not formats.intersection({"mov", "matroska", "webm"}):
            raise ValueError("Upload an MP4, MOV, M4V, MKV or WebM video.")
        videos = [
            s
            for s in data.get("streams", [])
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ]
        video = videos[0]
        width, height = int(video["width"]), int(video["height"])
        start_time = float(video.get("start_time") or data.get("format", {}).get("start_time") or 0)
        if not math.isfinite(start_time) or abs(start_time) > 1e9:
            raise ValueError("Unsupported video timestamps.")
        # Reordered VFR streams can under-report their stream duration. The
        # container endpoint also includes audio and is a safer progress estimate.
        # Subtract the video origin because output timestamps begin at zero.
        duration = max(
            float(video.get("duration") or 0),
            float(data.get("format", {}).get("duration") or 0) - max(0, start_time),
        )
        fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
        if not math.isfinite(fps) or fps <= 0:
            fps = float(Fraction(video.get("r_frame_rate", "0/1")))
        rotation = float(video.get("tags", {}).get("rotate", 0))
        for side in video.get("side_data_list", []):
            if "rotation" in side:
                rotation = float(side["rotation"])
        if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 0.01:
            raise ValueError("Only videos with 0, 90, 180 or 270 degree rotation are supported.")
        rotation = int(round(rotation)) % 360
        if width < 16 or height < 16 or max(width, height) > 8192 or width * height > MAX_PIXELS:
            raise ValueError("Video dimensions exceed the supported limit (up to 8K).")
        if not math.isfinite(duration) or not 0 < duration <= MAX_DURATION:
            raise ValueError("Video duration must be between zero and four hours.")
        if not math.isfinite(fps) or not 0 < fps <= 240.1:
            raise ValueError("Video frame rate must be between zero and 240 fps.")
        if rotation in (90, 270):
            width, height = height, width
        # H.264 yuv420p requires even output dimensions; pad rather than crop.
        width += width % 2
        height += height % 2
        return {
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "codec": video.get("codec_name", "unknown"),
            "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
            "hdr": video.get("color_transfer") in ("smpte2084", "arib-std-b67"),
            "rotation": rotation,
            "start_time": start_time,
            "video_stream_index": int(video["index"]),
        }
    except ValueError:
        raise
    except FileNotFoundError as exc:
        raise ValueError("FFmpeg and ffprobe must be installed.") from exc
    except (subprocess.SubprocessError, KeyError, IndexError, TypeError, ZeroDivisionError) as exc:
        raise ValueError("The upload is not a readable video.") from exc


def decode_filter(media: dict) -> str:
    filters = []
    if media["hdr"]:
        # iPhone Dolby Vision's HLG/PQ base layer becomes a broadly playable SDR image.
        # Requires an FFmpeg build with libzimg; fail clearly rather than export washed out HDR.
        filters.extend(
            [
                "zscale=t=linear:npl=100",
                "format=gbrpf32le",
                "zscale=p=bt709",
                "tonemap=tonemap=hable:desat=0",
                "zscale=t=bt709:m=bt709:r=tv",
            ]
        )
    filters.extend(
        [
            "setpts=PTS-STARTPTS",
            f"fps={media['fps']:.9f}:round=near",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "setsar=1",
            "format=bgr24",
        ]
    )
    return ",".join(filters)
