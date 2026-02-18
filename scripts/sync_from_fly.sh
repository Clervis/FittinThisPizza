#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-fittinthispizza}"
REMOTE_DIR="${REMOTE_DIR:-/data}"
MACHINE_ID="${FLY_MACHINE_ID:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

required_files=(
  "fitness_data.csv"
  "fitness_data_prototype.csv"
)

optional_files=(
  "nutrition_christian.csv"
  "nutrition_krysty.csv"
)

if ! command -v fly >/dev/null 2>&1; then
  echo "Error: fly CLI is not installed or not on PATH." >&2
  exit 1
fi

machine_args=()
select_args=()
if [[ -n "$MACHINE_ID" ]]; then
  machine_args=(--machine "$MACHINE_ID")
else
  select_args=(-s)
fi

source_machine_id="$MACHINE_ID"
source_volume=""
if [[ -z "$source_machine_id" ]]; then
  machine_line="$(fly machine list -a "$APP_NAME" | grep ' started ' | head -n 1 || true)"
  if [[ -n "$machine_line" ]]; then
    source_machine_id="$(awk '{print $1}' <<<"$machine_line")"
    source_volume="$(grep -o 'vol_[a-z0-9]*' <<<"$machine_line" | head -n 1 || true)"
  fi
fi

if [[ -n "$source_machine_id" && -z "$source_volume" ]]; then
  id_line="$(fly machine list -a "$APP_NAME" | grep "^$source_machine_id " | head -n 1 || true)"
  source_volume="$(grep -o 'vol_[a-z0-9]*' <<<"$id_line" | head -n 1 || true)"
fi

download_file() {
  local filename="$1"
  local required="$2"
  local tmp_path="./.$filename.flytmp"
  rm -f "$tmp_path"

  local cmd=(fly ssh sftp get "$REMOTE_DIR/$filename" "$tmp_path" -a "$APP_NAME")
  cmd+=("${machine_args[@]}")
  cmd+=("${select_args[@]}")

  if "${cmd[@]}"; then
    mv "$tmp_path" "./$filename"
    echo "Downloaded $filename"
    return 0
  fi

  rm -f "$tmp_path"

  if [[ "$required" == "required" ]]; then
    echo "Error: Failed to download required file: $filename" >&2
    exit 1
  fi

  echo "Warning: Optional file not found or failed to download: $filename"
}

for file in "${required_files[@]}"; do
  download_file "$file" "required"
done

for file in "${optional_files[@]}"; do
  download_file "$file" "optional"
done

pulled_at_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
cat > ./.fly_data_sync.json <<EOF
{
  "app_name": "$APP_NAME",
  "remote_dir": "$REMOTE_DIR",
  "source_machine_id": "${source_machine_id:-}",
  "source_volume": "${source_volume:-/data}",
  "pulled_at_utc": "$pulled_at_utc"
}
EOF

echo "Sync from Fly complete. Local CSVs are up to date."
