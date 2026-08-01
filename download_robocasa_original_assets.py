#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.cookiejar
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ASSETS = {
    "textures.zip": "https://utexas.box.com/shared/static/otdsyfjontk17jdp24bkhy2hgalofbh4.zip",
    "fixtures.zip": "https://utexas.box.com/shared/static/pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip",
    "objaverse.zip": "https://utexas.box.com/shared/static/ejt1kc2v5vhae1rl4k5697i4xvpbjcox.zip",
    "aigen_objs.zip": "https://utexas.box.com/shared/static/os3hrui06lasnuvwqpmwn0wcrduh6jg3.zip",
    "generative_textures.zip": "https://utexas.box.com/shared/static/gf9nkadvfrowkb9lmkcx58jwt4d6c1g3.zip",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def human_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{value} B"


def make_opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPRedirectHandler(),
    )


def download(opener, name: str, url: str, output_dir: Path, retries: int) -> None:
    destination = output_dir / name
    partial = output_dir / f"{name}.part"
    partial.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/octet-stream,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            method="GET",
        )
        try:
            print(f"\n[{name}] attempt {attempt}/{retries}")
            with opener.open(request, timeout=90) as response:
                status = getattr(response, "status", None)
                content_type = response.headers.get("Content-Type", "")
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header and total_header.isdigit() else 0
                print(f"[{name}] HTTP status: {status}")
                print(f"[{name}] final URL: {response.geturl()}")
                print(f"[{name}] content type: {content_type or 'unknown'}")
                if total:
                    print(f"[{name}] expected size: {human_bytes(total)}")

                downloaded = 0
                started = time.monotonic()
                next_report = started
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now >= next_report:
                            speed = downloaded / max(now - started, 0.001)
                            if total:
                                percent = 100.0 * downloaded / total
                                text = (
                                    f"\r[{name}] {percent:6.2f}% "
                                    f"{human_bytes(downloaded)}/{human_bytes(total)} "
                                    f"{human_bytes(int(speed))}/s"
                                )
                            else:
                                text = (
                                    f"\r[{name}] {human_bytes(downloaded)} "
                                    f"{human_bytes(int(speed))}/s"
                                )
                            print(text, end="", flush=True)
                            next_report = now + 1.0
                print()
                if downloaded == 0:
                    raise RuntimeError("empty response body")
                partial.replace(destination)
                print(f"[{name}] saved: {destination}")
                return
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            RuntimeError,
        ) as error:
            partial.unlink(missing_ok=True)
            print(f"[{name}] failed: {error}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(min(10 * attempt, 60))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="robocasa-original-assets")
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Asset family: RoboCasa Original / v0.2-era")
    print("Pinned commit: 2544dc2e38bb44f5ced80fbc91114a2f7934016a")
    print(f"Output: {output_dir}")
    print("RoboCasa365-only assets are not downloaded.")

    opener = make_opener()
    for name, url in ASSETS.items():
        download(opener, name, url, output_dir, args.retries)

    (output_dir / "SOURCE.txt").write_text(
        "asset_family=RoboCasa Original / v0.2-era\n"
        "pinned_commit=2544dc2e38bb44f5ced80fbc91114a2f7934016a\n"
        + "".join(f"{name}={url}\n" for name, url in ASSETS.items()),
        encoding="utf-8",
    )
    print("\nAll five archives downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
