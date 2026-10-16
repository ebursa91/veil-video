# Security

Veil is intended for a single trusted user on a private tailnet. Bind the native server to loopback, use Tailscale Serve for HTTPS, limit access in the tailnet policy, and use a strong unique app password. Never publish this service with Tailscale Funnel or a public reverse proxy.

Uploaded videos are untrusted inputs to FFmpeg and the detector. Run Veil as an unprivileged user and keep FFmpeg, Python dependencies, the NVIDIA driver, and the host updated. The Docker service drops Linux capabilities and uses a read-only root filesystem, but it is not a strong sandbox for hostile media.

Videos, job records, exports, password hashes, and session secrets remain on the host. Protect the data directory and backups. Deleting a job removes its files and record; this is ordinary filesystem deletion, not secure erasure. There is no automatic retention policy.

The application does not guarantee anonymization. Detection can miss faces, and the output retains audio and other identifying information. Review the complete result before sharing it.

To report a vulnerability, use the repository's private vulnerability reporting feature if enabled. Do not attach private footage, passwords, or secrets to a public issue. If private reporting is unavailable, open a minimal issue requesting a private contact channel without publishing exploit details or sensitive data.
