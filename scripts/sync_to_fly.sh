#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-fittinthispizza}"
REMOTE_DIR="${REMOTE_DIR:-/data/data}"
MACHINE_ID="${FLY_MACHINE_ID:-}"
FORCE="${FORCE_SYNC:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
LOCAL_DIR="$REPO_ROOT/data"

for arg in "$@"; do
  if [[ "$arg" == "-y" || "$arg" == "--yes" ]]; then
    FORCE=1
  fi
done

files=(
  "fitness_data.csv"
  "fitness_data_prototype.csv"
  "nutrition_christian.csv"
  "nutrition_krysty.csv"
)

if ! command -v fly >/dev/null 2>&1; then
  echo "Error: fly CLI is not installed or not on PATH." >&2
  exit 1
fi

if [[ "$FORCE" != "1" ]]; then
  echo "This will upload local CSV files to Fly and overwrite remote data files."
  read -r -p "Continue? [y/N] " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

machine_args=()
select_args=()
if [[ -n "$MACHINE_ID" ]]; then
  machine_args=(--machine "$MACHINE_ID")
else
  select_args=(-s)
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir=".fly-sync-backups/$timestamp"
mkdir -p "$backup_dir"

status=0

for file in "${files[@]}"; do
  if [[ ! -f "$LOCAL_DIR/$file" ]]; then
    echo "Warning: Local file missing, skipping upload: $LOCAL_DIR/$file"
    continue
  fi

  get_cmd=(fly ssh sftp get "$REMOTE_DIR/$file" "$backup_dir/$file" -a "$APP_NAME")
  get_cmd+=("${machine_args[@]}")
  get_cmd+=("${select_args[@]}")
  if "${get_cmd[@]}"; then
    echo "Backed up remote $file to $backup_dir/$file"
  else
    echo "Warning: Could not back up remote $file (may not exist yet)."
  fi

  put_cmd=(fly ssh sftp put "$LOCAL_DIR/$file" "$REMOTE_DIR/$file" -a "$APP_NAME")
  put_cmd+=("${machine_args[@]}")
  put_cmd+=("${select_args[@]}")
  if "${put_cmd[@]}"; then
    echo "Uploaded $file"
  else
    echo "Error: Failed to upload $file" >&2
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  echo "Sync to Fly complete."
else
  echo "Sync to Fly completed with errors." >&2
fi

exit "$status"
