#!/usr/bin/env bash

set -Eeuo pipefail

if [[ $# -lt 1 || -z "${1//[[:space:]]/}" ]]; then
    echo "Usage: bash scripts/publish_and_sync.sh \"commit message\"" >&2
    exit 2
fi

message="$1"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

bash "$ROOT/scripts/export_third_party_patches.sh"
git add -A

if git diff --cached --quiet; then
    echo "No staged changes; skipping commit."
else
    git commit -m "$message"
fi

if ! branch="$(git symbolic-ref --quiet --short HEAD)"; then
    echo "Cannot publish from detached HEAD." >&2
    exit 1
fi

git push origin "$branch"

deploy_host="${DEPLOY_HOST:-zhaoyanghe@172.28.11.129}"
deploy_root="${DEPLOY_ROOT:-/data/users/zhaoyanghe/control-frequency-wm}"

ssh "$deploy_host" bash -s -- "$branch" "$deploy_root" <<'REMOTE'
set -Eeuo pipefail

branch="$1"
deploy_root="$2"

cd "$deploy_root"
git fetch origin "$branch"
git reset --hard "origin/$branch"
bash scripts/apply_third_party_patches.sh
REMOTE

