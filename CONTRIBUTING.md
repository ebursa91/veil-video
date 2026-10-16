# Contributing

Use Python 3.12, uv, and a recent FFmpeg build. Keep changes focused and include tests for behavior that can break privacy, job state, or media handling.

```sh
uv sync --locked --extra cpu --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

For a manual run, use `uv run veil init` and `uv run veil serve`. Use synthetic or permissioned media. Do not commit uploaded videos, face images, `.env`, passwords, databases, model binaries, or generated exports.

Before opening a pull request:

- Describe the user-visible behavior and how you verified it.
- Include a regression test for a bug fix where practical.
- Exercise upload, cancellation, deletion, and export if your change touches the pipeline.
- Test GPU changes with a real GPU-mode job. CPU CI cannot establish CUDA or NVENC correctness.
- Keep the UI keyboard accessible and usable on an iPhone-sized viewport.
- Update the documentation when setup or limitations change.

Veil deliberately has one user, one processing worker, and local files. New editing operations should fit the worker's progress/cancellation boundary. Avoid adding accounts, remote media fetches, analytics, or external services without a clear project decision.
