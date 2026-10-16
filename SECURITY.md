# Security

Keep Veil on a private tailnet with a strong, unique app password and restricted tailnet access. Prefer loopback binding with Tailscale Serve for HTTPS; see [deployment](docs/deployment.md) for configuration. Never expose it through Tailscale Funnel or a public reverse proxy.

Uploads are untrusted input to FFmpeg and the detector. Run as an unprivileged user and keep dependencies, the GPU driver, and the host updated. Docker drops capabilities and uses a read-only root filesystem; it is not a strong sandbox for hostile media.

Protect the data directory, credentials, and backups. Finished media expires after seven days by default; archived history retains the filename and job summary until deleted. Automatic and manual deletion use ordinary filesystem removal, without secure erasure. Existing backups are unaffected.

Face detection can miss people. Exports retain audio and other identifying content. Review the complete video before sharing; Veil does not guarantee anonymization.

Report vulnerabilities through the repository's private reporting feature when available. Otherwise, open an issue requesting a private contact channel. Keep exploit details, private footage, and secrets out of public issues.
