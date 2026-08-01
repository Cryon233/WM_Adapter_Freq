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

mkdir -p "$PATCH_DIR"
TEMP_DIR="$(mktemp -d "$PATCH_DIR/.export.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT

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

    untracked_file="$TEMP_DIR/$name.untracked"
    git -C "$repo" ls-files --others --exclude-standard -z > "$untracked_file"
    if [[ -s "$untracked_file" ]]; then
        echo "Cannot export $name: non-ignored untracked files would be omitted:" >&2
        while IFS= read -r -d '' path; do
            printf '  %s\n' "$path" >&2
        done < "$untracked_file"
        exit 1
    fi
done

generated_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
manifest="$TEMP_DIR/MANIFEST.txt"
{
    echo "Third-party patch manifest"
    echo "Generated at (UTC): $generated_at"
    echo
} > "$manifest"

for index in "${!REPO_NAMES[@]}"; do
    name="${REPO_NAMES[$index]}"
    repo="$ROOT/third_party/$name"
    base_commit="${BASE_COMMITS[$index]}"
    patch_name="$name.patch"
    temporary_patch="$TEMP_DIR/$patch_name"

    git -C "$repo" diff --binary "$base_commit" -- > "$temporary_patch"

    current_head="$(git -C "$repo" rev-parse HEAD)"
    patch_sha256="$(sha256sum "$temporary_patch" | awk '{print $1}')"
    if [[ -s "$temporary_patch" ]]; then
        patch_status="present"
    else
        patch_status="empty (patch file omitted)"
    fi

    {
        echo "Repository: $name"
        echo "Base commit: $base_commit"
        echo "Current HEAD: $current_head"
        echo "Patch file: $patch_name"
        echo "Patch SHA256: $patch_sha256"
        echo "Patch status: $patch_status"
        echo
    } >> "$manifest"
done

for name in "${REPO_NAMES[@]}"; do
    temporary_patch="$TEMP_DIR/$name.patch"
    final_patch="$PATCH_DIR/$name.patch"
    if [[ -s "$temporary_patch" ]]; then
        mv -f "$temporary_patch" "$final_patch"
    else
        rm -f "$final_patch"
    fi
done
mv -f "$manifest" "$PATCH_DIR/MANIFEST.txt"

rm -rf "$TEMP_DIR"
trap - EXIT

git -C "$ROOT" add patches/third_party

echo "Exported third-party patches to $PATCH_DIR"
for name in "${REPO_NAMES[@]}"; do
    if [[ -s "$PATCH_DIR/$name.patch" ]]; then
        echo "  $name: $PATCH_DIR/$name.patch"
    else
        echo "  $name: no tracked changes"
    fi
done
