#!/usr/bin/env bash
# Install only this app's unprivileged systemd unit; never changes networking.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$project_dir/.venv/bin/veil" || ! -f "$project_dir/.env" ]]; then
  echo "Run uv sync and veil init from the checkout first." >&2
  exit 1
fi

# Avoid ambiguous systemd quoting/specifier expansion in unusual checkout paths.
if [[ "$project_dir" == *[\"\%\$\\]* ]]; then
  echo "Move the checkout to a path without quotes, percent signs, dollar signs, or backslashes." >&2
  exit 1
fi

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_path="$unit_dir/veil.service"
mkdir -p -- "$unit_dir"
if [[ -e "$unit_path" || -L "$unit_path" ]]; then
  echo "$unit_path already exists. Inspect it before replacing the service." >&2
  exit 1
fi
umask 077
cat > "$unit_path" <<EOF
[Unit]
Description=Veil private video workbench
After=network.target

[Service]
Type=simple
WorkingDirectory=$project_dir
ExecStart="$project_dir/.venv/bin/veil" --env-file "$project_dir/.env" serve
Restart=on-failure
RestartSec=5
TimeoutStopSec=45
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
EOF
systemd-analyze --user verify "$unit_path"
systemctl --user daemon-reload
systemctl --user enable --now veil.service
systemctl --user --no-pager status veil.service
