FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.12.20 /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TMPDIR=/data \
    VEIL_DATA_DIR=/data \
    VEIL_MODEL_PATH=/models/yunet.onnx \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 veil \
    && useradd --uid 10001 --gid veil --no-create-home veil \
    && install -d -o veil -g veil /app /config /data /models

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY veil ./veil
COPY docs/licenses ./docs/licenses
RUN uv sync --frozen --extra gpu --no-dev --no-editable \
    && chmod -R a+rX /opt/venv

USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3)"]
ENTRYPOINT ["veil", "--env-file", "/config/.env"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765"]
