#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PATCH_DIR="$ROOT/patches/third_party"

REPO_NAMES=(
    "jepa-wms"
    "dinov3"
    "robosuite"
    "robocasa"
)
BASE_COMMITS=(
    "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
    "6876159a11b4df116f30f667f8c9888617df0751"
    "9548a5a35bde8eabf47f760802045cca447e9c0c"
    "2544dc2e38bb44f5ced80fbc91114a2f7934016a"
)

for index in "${!REPO_NAMES[@]}"; do
    name="${REPO_NAMES[$index]}"
    repo="$ROOT/third_party/$name"
    base_commit="${BASE_COMMITS[$index]}"

    if [[ ! -d "$repo" || ! -e "$repo/.git" ]]; then
        echo "Third-party repository is missing or is not a Git repository: $repo" >&2
        exit 1
    fi

    if ! git -C "$repo" cat-file -e "$base_commit^{commit}" 2>/dev/null; then
        echo "Fixed base commit is unavailable in $name: $base_commit" >&2
        exit 1
    fi
done

for index in "${!REPO_NAMES[@]}"; do
    name="${REPO_NAMES[$index]}"
    repo="$ROOT/third_party/$name"
    base_commit="${BASE_COMMITS[$index]}"
    patch="$PATCH_DIR/$name.patch"

    git -C "$repo" reset --hard "$base_commit"

    if [[ -s "$patch" ]]; then
        if ! git -C "$repo" apply --check "$patch"; then
            echo "Third-party patch check failed." >&2
            echo "Repository: $name" >&2
            echo "Base commit: $base_commit" >&2
            echo "Patch: $patch" >&2
            exit 1
        fi
        git -C "$repo" apply "$patch"
        patch_status="applied"
    else
        patch_status="no non-empty patch"
    fi

    git -C "$repo" diff --check

    echo "Repository: $name"
    echo "  Base commit: $base_commit"
    echo "  Patch status: $patch_status"
    echo "  Changed tracked files:"
    changed_files="$(git -C "$repo" diff --name-only "$base_commit" --)"
    if [[ -n "$changed_files" ]]; then
        while IFS= read -r path; do
            printf '    %s\n' "$path"
        done <<< "$changed_files"
    else
        echo "    (none)"
    fi
done
