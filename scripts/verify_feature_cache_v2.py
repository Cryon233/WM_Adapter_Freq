#!/usr/bin/env python3
from __future__ import annotations

import argparse

from wm_adapter.data.feature_cache_v2 import verify_cache_content_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Deeply verify a v2 feature cache")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--chunk-windows", type=int, default=8)
    args = parser.parse_args()

    def progress(key: str, completed: int, total: int) -> None:
        print(
            f"CACHE_VERIFY_PROGRESS key={key} completed={completed} total={total}",
            flush=True,
        )

    result = verify_cache_content_v2(
        args.cache,
        chunk_windows=args.chunk_windows,
        progress=progress,
    )
    print(
        "CACHE_VERIFY_COMPLETE "
        f"path={args.cache} content_sha256={result['recomputed_content_sha256']} "
        f"file_sha256={result['cache_file_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
