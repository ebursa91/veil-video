# Contributing

Use Python 3.12, uv, and a recent FFmpeg build.

```sh
uv sync --locked --extra cpu --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Run locally with `uv run veil init` and `uv run veil serve`. Use synthetic or permissioned test media. Keep private media, `.env`, credentials, databases, model binaries, and exports out of Git.

Before opening a pull request:

- Describe the change and its verification; add regression coverage for privacy, job state, and media handling.
- For pipeline changes, check upload, cancellation, deletion, and export. Verify GPU changes with a real GPU-mode job; CPU CI cannot verify CUDA or NVENC.
- Check UI changes with a keyboard and an iPhone-sized viewport.
- Update setup instructions and limitations when they change.

Keep the single-user design. New editing operations should use the worker's progress and cancellation interface. Discuss accounts, remote media, or external services before adding them.
