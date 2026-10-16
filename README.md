# Veil

Self-hosted face blurring for video. Upload a clip, track progress, review the result, and download an MP4. Single-user, password-protected, with private Tailscale access.

- Supports MOV/HEVC uploads, including iPhone recordings, rotation, and variable frame rates.
- Uses CUDA face detection and NVENC encoding when available; CPU mode is supported.
- Exports SDR H.264/AAC video. HDR is converted to SDR; source metadata is removed.
- Includes a persistent queue, cancellation, preview, download, and deletion.
- Stores media locally. No telemetry or cloud processing.

**Review exports before sharing.** Detection can miss faces; audio and other identifying details remain.

![Veil workspace](docs/screenshots/workspace.png)

## Setup

Requires Linux, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), and FFmpeg/FFprobe. GPU mode also requires a compatible NVIDIA driver and FFmpeg with `h264_nvenc`.

```sh
uv sync --locked --extra gpu
uv run veil init
uv run veil doctor
uv run veil serve
```

For CPU-only operation, replace `--extra gpu` with `--extra cpu`. `init` creates a password configuration and downloads the verified face model. Open <http://127.0.0.1:8765>.

See [deployment](docs/deployment.md) for Tailscale, Docker, configuration, and limits; [contributing](CONTRIBUTING.md) for development; and [architecture](docs/architecture.md) for implementation details.

## License

[MIT](LICENSE). YuNet has a separate [MIT notice](docs/licenses/YuNet-MIT.txt). Dependencies retain their own licenses.
