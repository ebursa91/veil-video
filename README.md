# Veil

A small, self-hosted app for blurring faces in video. Upload a clip, watch its progress, review the result, and download an MP4. Built for one person running it on their own machine through Tailscale.

Veil uses YuNet face detection through ONNX Runtime and FFmpeg. NVIDIA GPUs can accelerate detection and H.264 encoding. The interface reports the backend actually used; CPU processing is also available.

**Review every export before sharing it.** Automatic detection can miss small, obscured, fast-moving, or profile faces. Face blurring does not guarantee anonymity: voices, clothing, surroundings, and other identifying details remain.

## What it does

- Accepts browser uploads, including iPhone `.mov` / HEVC clips.
- Detects faces in every processed frame, expands the blur area, and briefly holds masks across missed detections.
- Handles rotation and variable frame rates; exports an SDR H.264/AAC MP4 with source metadata removed.
- Shows upload progress, processing stages, frames, speed, and estimated time remaining.
- Runs one processing job at a time with a persistent queue, cancellation, preview, download, and deletion.
- Protects the workspace with a single password and CSRF-protected sessions.
- Keeps uploaded media, the model, and job state on the host. No telemetry, cloud processing, or runtime model downloads.

HDR input is converted to SDR. This first version does not preserve HDR or Dolby Vision, provide manual face masks, alter voices, or edit audio.

![Veil workspace showing a reviewed face-blurred export and GPU processing controls](docs/screenshots/workspace.png)

## Run it

Linux, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), and FFmpeg/FFprobe on `PATH` are required. FFmpeg must include H.264 encoding; GPU encoding also needs `h264_nvenc`. For NVIDIA, install a compatible driver and the GPU dependency extra:

```sh
uv sync --locked --extra gpu --extra dev
uv run veil init
uv run veil doctor
uv run veil serve
```

`init` prompts for a password, creates local secrets in `.env`, and downloads the pinned, checksum-verified face detection model. Run it once with internet access. The server then works offline. Open <http://127.0.0.1:8765> and sign in with your chosen password.

For a machine without NVIDIA support, use `uv sync --locked --extra cpu --extra dev` instead. Install only one inference extra at a time. See [deployment](docs/deployment.md) for GPU prerequisites, Docker, configuration, and Tailscale access.

## Processing modes

| Mode | Behavior |
| --- | --- |
| Auto | Uses available acceleration and reports any fallback. |
| GPU | Requires working CUDA inference and NVIDIA encoding; fails if either is unavailable. |
| CPU | Uses CPU inference and software H.264 encoding. |

H.264 GPU encoding supports output dimensions from 144–4,096 pixels wide and 64–4,096 pixels high. Larger supported videos use CPU encoding in Auto mode; GPU mode reports an error. Inputs up to 240 fps are supported, including iPhone slow-motion clips.

GPU acceleration covers face inference and final video encoding. Decoding, image transfer, mask drawing, and blurring also consume CPU and memory. Speed depends on resolution, duration, codec, frame rate, face count, and hardware.

## Private access from your iPhone

Keep Veil bound to `127.0.0.1`. Connect the host and phone to your tailnet, then run:

```sh
tailscale serve --bg http://127.0.0.1:8765
tailscale serve status
```

Open the HTTPS URL printed by Tailscale on your phone. Restrict access with your tailnet policy. [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve) proxies the local app to devices in your tailnet. Do not enable Funnel for this app.

## Development

```sh
uv sync --locked --extra cpu --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

See [contributing](CONTRIBUTING.md) and [architecture](docs/architecture.md). The CI suite uses CPU inference; GPU availability must be checked on the deployment host with `veil doctor` and a GPU-mode video job.

## License

Veil is [MIT licensed](LICENSE). YuNet's [MIT notice](docs/licenses/YuNet-MIT.txt) is retained separately. FFmpeg and other dependencies retain their own licenses; review the license of your FFmpeg build before redistributing binaries.
