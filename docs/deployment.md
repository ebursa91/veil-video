# Deployment

## Native installation

The recommended target is an NVIDIA Linux host. Install FFmpeg with `ffprobe`, Python 3.12, and [uv](https://docs.astral.sh/uv/getting-started/installation/). On Ubuntu, FFmpeg is available through `sudo apt install ffmpeg`; on Arch Linux, use `sudo pacman -S ffmpeg`.

FFmpeg needs software H.264/AAC encoding, `h264_nvenc` for GPU encoding, and the `zscale` / `tonemap` filters for HDR sources. Common distribution builds include these; a reduced custom FFmpeg build may not.

```sh
uv sync --locked --extra gpu --extra dev
uv run veil init
uv run veil doctor
uv run veil serve
```

`init` prompts for a password and confirmation. It writes an Argon2id password hash and random session secret to `.env` and downloads the face detection model. Do not use a shared password or publish `.env`. Keep the working directory private and run the app as an ordinary user.

The GPU extra includes CUDA/cuDNN Python runtime packages. A compatible NVIDIA host driver is still required. Veil pins ONNX Runtime below 1.27 to stay on the CUDA 12 package line; the container uses CUDA 12.8 and cuDNN 9. See the [official CUDA provider compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for driver/runtime requirements. `nvidia-smi` alone does not establish working video encoding or model inference: check `veil doctor` and process a clip in GPU mode.

For CPU-only operation:

```sh
uv sync --locked --extra cpu --extra dev
uv run veil init
uv run veil serve
```

Only one inference extra can be selected. `uv sync` removes the other environment's packages when switching between CPU and GPU.

## Configuration

The CLI loads `.env` from the working directory. Set `--env-file` before the command to use another file, for example `uv run veil --env-file /private/veil.env serve`. Environment variables take precedence.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VEIL_PASSWORD_HASH` | Required | Argon2id hash created by `init`. |
| `VEIL_SECRET_KEY` | Required | Random signing secret created by `init`; at least 32 characters. |
| `VEIL_DATA_DIR` | `data` | SQLite database, uploaded sources, and exported results. |
| `VEIL_MODEL_PATH` | `models/yunet.onnx` | Local checksum-verified model. |
| `VEIL_MAX_UPLOAD_GB` | `8` | Per-file upload limit, integer GiB from 1 to 100. |
| `VEIL_MODE` | `auto` | Default processing mode: `auto`, `gpu`, or `cpu`. |
| `VEIL_HOST` | `127.0.0.1` | Listener address; use an exact Tailscale IP for direct tailnet access. |
| `VEIL_PORT` | `8765` | Listener port; command-line `--port` takes precedence. |
| `VEIL_SECURE_COOKIE` | `false` | Set `true` when accessing the app over Tailscale HTTPS. |
| `VEIL_ORIGIN` | Empty | Public origin, such as `https://host.tailnet.ts.net`, for CSRF origin checks behind a proxy. |
| `VEIL_TRUSTED_PROXIES` | `127.0.0.1` | Comma-separated proxy IPs trusted to supply forwarded headers. Keep this narrow. |

Use absolute paths in a service. The default listener is `127.0.0.1:8765`; `serve --host` and `--port` change it. Run one server process per data directory. Each server owns one worker; do not use Uvicorn's multiple-worker mode.

When using HTTPS through Tailscale, set `VEIL_SECURE_COOKIE=true` in `.env` and restart Veil. A secure cookie will not sign in over ordinary HTTP on a remote hostname. Keep `false` only while testing locally over HTTP.

For unattended native operation on Linux with systemd, stop any manually started Veil server, then install the included user service:

```sh
bash scripts/install-user-service.sh
systemctl --user status veil
journalctl --user -u veil -f
```

The installer creates only `~/.config/systemd/user/veil.service`, runs as your own user, listens on loopback, and restarts the app after a crash. It refuses to replace an existing service. It starts with your user session; if you need it to run after logout or before login, enable lingering for your user through your system administrator. To stop it, run `systemctl --user disable --now veil`. Keep the checkout at the same path and use absolute data/model paths in `.env`.

## Tailscale

Connect the server and iPhone to your tailnet. Leave Veil on loopback and configure the host's [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve):

```sh
tailscale serve --bg http://127.0.0.1:8765
tailscale serve status
```

Tailscale may prompt to enable HTTPS certificates for the tailnet. Open the resulting `https://…ts.net` URL on the phone, then enter the Veil password. Your tailnet access policy controls which devices can reach it. Do not enable Funnel: this app is intended to stay private. To stop the proxy, use `tailscale serve off`.

If Serve is not enabled and you cannot change tailnet settings, bind directly to the host's **exact Tailscale IPv4 address**, obtained with `tailscale ip -4`. Set `VEIL_HOST` to that address in `.env`, leave `VEIL_SECURE_COOKIE=false`, clear `VEIL_ORIGIN`, and restart Veil. Open `http://<tailscale-ip>:8765` from a tailnet device. Tailscale encrypts this network connection; the browser connection itself uses HTTP. This listener accepts traffic on the Tailscale interface, and does not listen on the LAN or public addresses. Prefer Serve HTTPS when available. Keep your Tailscale IP stable and update the setting if it changes.

## GPU Docker

Install Docker Engine with Compose v2 and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Configure the Docker runtime as described by NVIDIA. The host driver must support CUDA 12.8. The container does not install a host driver.

The supplied image uses Python 3.12, CUDA 12.8, cuDNN 9, and GPU inference dependencies. It runs as UID/GID `10001`, publishes only `127.0.0.1:8765`, and stores secrets, jobs, and the model in named volumes.

```sh
docker compose build
docker compose run --rm veil init
docker compose run --rm veil doctor
docker compose up -d
docker compose logs -f veil
```

The interactive `init` command writes `/config/.env` inside the private config volume and downloads `/models/yunet.onnx`; no secrets are built into the image. Never run `docker compose down -v` unless you intend to delete the stored configuration, videos, and model.

Configure Tailscale Serve on the host using the same loopback URL above. For Docker behind Tailscale HTTPS, set both values in the checkout's local Compose `.env`, using the exact HTTPS origin printed by `tailscale serve status`:

```dotenv
VEIL_SECURE_COOKIE=true
VEIL_ORIGIN=https://your-host.your-tailnet.ts.net
```

Then run `docker compose up -d` again. The explicit public origin allows CSRF validation through Docker's bridge without trusting arbitrary forwarded headers. Keep these values when recreating the service. The app password and secret remain in the separate config volume.

Before trusting a deployment, upload a short permissioned clip, select GPU mode, confirm that progress reports CUDA/NVENC, inspect the complete export, and test deletion. This validates the host GPU path; a successful image build does not.

## Storage and updates

Accepted filename extensions are `.mov`, `.mp4`, `.m4v`, `.mkv`, and `.webm`; the actual file must also be a supported video container. The pipeline limits a clip to four hours, 240 fps, 8,192 pixels on either edge, and 33,177,600 total pixels. This includes 240 fps iPhone slow-motion input. The configurable upload limit defaults to 8 GiB.

H.264 NVENC output is limited to widths of 144–4,096 pixels and heights of 64–4,096 pixels. Auto mode uses CPU encoding when a supported clip falls outside these bounds; strict GPU mode fails with a dimension error. CUDA face inference can still run while Auto mode uses software encoding.

Source uploads, partial work, and outputs need free disk space. A large upload may need several times its source size while processing. Jobs stay on disk until deleted. Back up the data directory and `.env` together only if you need job history; backups contain private media and authentication material.

Stop the service before copying SQLite state. The model can be downloaded again by `init`. Do not share backup archives publicly. On restart, interrupted processing is marked failed; upload that source again to retry.

To update a native installation, stop Veil, update the checkout, run `uv sync --locked --extra gpu --extra dev`, check `veil doctor`, and restart. For Docker, rebuild with `docker compose build --pull`, then recreate with `docker compose up -d` while preserving the named volumes.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Startup asks to initialize | Run `veil init`, or point `--env-file` at the configured file. |
| CUDA unavailable | GPU extra, NVIDIA driver, runtime library loading, and `veil doctor` output. |
| NVENC unavailable | FFmpeg's `h264_nvenc` encoder, GPU support, driver, and container `video` capability. |
| GPU mode fails but Auto succeeds | Auto can use CPU fallback. Read the job backend and error before diagnosing performance. |
| Phone cannot reach the app | Tailnet connection, Serve status, HTTPS URL, and tailnet access policy. |
| Phone returns to login | Secure-cookie setting, correct HTTPS URL, or expired session. |
| Clip rejected | Container/codec support, configured upload size, duration, frame-rate, and resolution limits. |
| Colors differ from the source | Export is SDR; HDR/Dolby Vision metadata is not preserved. |
