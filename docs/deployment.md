# Deployment

## Requirements

- Linux, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), and FFmpeg/FFprobe.
- FFmpeg with H.264/AAC encoding and `zscale` / `tonemap` for HDR input.
- GPU mode: a compatible NVIDIA driver and FFmpeg with `h264_nvenc`. The GPU extra supplies CUDA/cuDNN libraries. See [ONNX Runtime compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

Follow the [README setup](../README.md#setup). Model downloads require internet access during initialization; processing then works offline. Run `veil doctor` to check inference and encoding support. Existing installations can rerun `veil init` to install the expression model while retaining their password configuration.

## Configuration

The CLI loads `.env`; environment variables take precedence. Use `veil --env-file /path/to/veil.env serve` for another file. Keep secrets and media private, and use absolute storage paths for a service.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VEIL_PASSWORD_HASH` | Required | Argon2id hash created by `init`. |
| `VEIL_SECRET_KEY` | Required | Session secret created by `init`. |
| `VEIL_DATA_DIR` | `data` | Media, temporary uploads, and job database. |
| `VEIL_MODEL_PATH` | `models/yunet.onnx` | Verified local face model. |
| `VEIL_EXPRESSION_MODEL_PATH` | `expression.onnx` beside the face model | Verified expression model for optional emoji covers. |
| `VEIL_MAX_UPLOAD_GB` | `8` | Upload limit in GiB, from 1 to 100. |
| `VEIL_MODE` | `auto` | `auto`, `gpu`, or `cpu`. |
| `VEIL_HOST` | `127.0.0.1` | Listener address. |
| `VEIL_PORT` | `8765` | Listener port. |
| `VEIL_SECURE_COOKIE` | `false` | Set `true` for HTTPS. |
| `VEIL_ORIGIN` | Empty | Exact browser origin for reverse-proxy deployments. |
| `VEIL_TRUSTED_PROXIES` | `127.0.0.1` | Proxy IPs trusted to supply forwarded headers. |

`serve --host` and `--port` override listener settings. Run one server process per data directory. Auto mode reports any CPU fallback; GPU mode requires both CUDA inference and NVENC encoding. Decoding and masking use CPU.

Blur remains available without the expression model. Emoji jobs require that model and add expression inference per detected face, so crowded clips can take longer. Model scores choose an approximate visible expression; they do not establish how someone feels. Overlays follow estimated face geometry, with blur underneath to cover tracking gaps. Small, obscured, or strongly turned faces can be missed or classified incorrectly; review the entire export.

During processing, live previews show small processed snapshots about once a second. Hide the preview to stop image requests; requests also pause in background tabs. These snapshots supplement progress updates; review the completed video before sharing.

## Private access

Keep Veil on loopback and configure [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve):

```sh
tailscale serve --bg http://127.0.0.1:8765
tailscale serve status
```

Enable tailnet HTTPS if prompted. Set `VEIL_SECURE_COOKIE=true`, restart Veil, and open the HTTPS URL on a tailnet device. Restrict access through your tailnet policy. Do not enable Funnel.

Alternatively, set `VEIL_HOST` to the exact address from `tailscale ip -4`, leave `VEIL_SECURE_COOKIE=false`, and clear `VEIL_ORIGIN`. Restart and open `http://<tailscale-ip>:8765`. Tailscale encrypts the transport; the browser connection uses HTTP. Prefer Serve HTTPS when available.

## User service

Stop any manually started server, then run:

```sh
bash scripts/install-user-service.sh
systemctl --user status veil
journalctl --user -u veil -f
```

The installer creates `~/.config/systemd/user/veil.service` and refuses to overwrite an existing unit. It starts with the user session and restarts after failure. Keep the checkout path fixed. For operation before login or after logout, configure user lingering with your administrator. Stop with `systemctl --user disable --now veil`.

## Docker

Requires Docker Compose and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```sh
docker compose build
docker compose run --rm veil init
docker compose run --rm veil doctor
docker compose up -d
```

Compose publishes only the loopback port and stores configuration, media, and models in named volumes. For Tailscale Serve, set these in the checkout's Compose `.env` and recreate the service:

```dotenv
VEIL_SECURE_COOKIE=true
VEIL_ORIGIN=https://your-host.your-tailnet.ts.net
```

Use the exact origin from `tailscale serve status`. App credentials remain in the separate config volume. `docker compose down -v` deletes all named volumes.

## Limits and storage

Uploads: MOV, MP4, M4V, MKV, or WebM; up to four hours, 240 fps, 8,192 pixels per edge, and 33,177,600 total pixels. NVENC output supports widths of 144–4,096 and heights of 64–4,096 pixels. Auto mode uses CPU encoding outside those bounds; GPU mode fails.

Jobs remain until deleted. Allow several times the source size in free disk space. Interrupted jobs must be uploaded again. Stop Veil before backing up SQLite; backups of media and `.env` contain private data.

For updates, stop Veil, update the checkout, sync the selected dependency extra, run `veil doctor`, and restart. For Docker, rebuild and recreate without removing volumes.
