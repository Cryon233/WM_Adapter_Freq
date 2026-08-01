#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || -z "${1//[[:space:]]/}" ]]; then
    echo 'Usage: bash scripts/publish_and_sync.sh "commit message"' >&2
    exit 2
fi

message="$1"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Direct-vendor mode requires third_party directories to be ordinary directories.
nested_git="$(find third_party -mindepth 2 -maxdepth 2 -name .git -print -quit 2>/dev/null || true)"
if [[ -n "$nested_git" ]]; then
    echo "Nested Git metadata still exists: $nested_git" >&2
    echo "Move all third_party/*/.git entries outside the main repository first." >&2
    exit 1
fi

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
REMOTE

echo "Published branch $branch and synchronized $deploy_host:$deploy_root"
