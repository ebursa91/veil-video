import shutil
import subprocess

import pytest

from veil.media import decode_filter, probe

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")


def make_video(path, *, duration="0.8", extra=()):
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
            duration,
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-metadata",
            "location=+52.0+021.0/",
            *extra,
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_probe_reads_audio_and_dimensions(tmp_path):
    result = probe(make_video(tmp_path / "input.mov"))
    assert (result["width"], result["height"]) == (192, 128)
    assert result["fps"] == 10
    assert result["has_audio"] and not result["hdr"]
    assert result["duration"] == pytest.approx(0.8)


def test_rotation_is_baked_into_output_geometry(tmp_path):
    source = make_video(tmp_path / "input.mp4")
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
    result = probe(rotated)
    assert (result["width"], result["height"]) == (128, 192)
    assert result["rotation"] in (90, 270)


def test_hdr_uses_explicit_tonemap():
    filters = decode_filter({"hdr": True, "fps": 30})
    assert "tonemap=tonemap=hable" in filters
    assert "zscale=t=bt709" in filters
    assert "fps=30." in filters


def test_invalid_input_is_rejected(tmp_path):
    path = tmp_path / "invalid.mov"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="readable video"):
        probe(path)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        probe(path)


def test_disguised_playlist_cannot_reference_local_media(tmp_path):
    target = make_video(tmp_path / "private.mp4")
    playlist = tmp_path / "upload.mov"
    playlist.write_text(f"ffconcat version 1.0\nfile '{target}'\n")
    with pytest.raises(ValueError, match="readable video"):
        probe(playlist)
