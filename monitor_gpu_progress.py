#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
LOG_ROOT = ROOT / "logs" / "full_a100_experiment"
CKPT_ROOT = ROOT / "checkpoints" / "jepa_wm_droid" / "robocasa"
REFRESH = float(os.environ.get("MONITOR_REFRESH", "2"))
BAR_WIDTH = int(os.environ.get("MONITOR_BAR_WIDTH", "30"))
METHODS = ("dct_adapter", "token_mlp", "lora")


@dataclass
class Progress:
    percent: float
    label: str
    state: str = "running"


def read_log(name: str) -> str:
    try:
        return (LOG_ROOT / name).read_text(
            encoding="utf-8", errors="replace"
        ).replace("\r", "\n")
    except FileNotFoundError:
        return ""


def last_match(pattern: str, text: str, flags: int = 0):
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def failure(text: str) -> Progress | None:
    if "Traceback (most recent call last)" not in text:
        return None
    lines = [
        line.strip()
        for line in text.splitlines()
        if any(token in line for token in (
            "RuntimeError:",
            "ModuleNotFoundError:",
            "ValueError:",
            "Error:",
        ))
    ]
    detail = lines[-1] if lines else "see log"
    return Progress(0.0, f"FAILED: {detail[:85]}", "failed")


def cache_progress() -> Progress:
    text = read_log("build_feature_cache.log")
    if bad := failure(text):
        return bad
    if "Feature cache written:" in text and "Cache fingerprint:" in text:
        return Progress(100.0, "feature cache complete", "done")
    if match := last_match(r"(\d{1,3})%\|", text):
        return Progress(float(match.group(1)), "building shared feature cache")
    return Progress(0.0, "initializing" if text else "not started", "waiting")


def training_progress(method: str) -> Progress:
    checkpoint = CKPT_ROOT / f"{method}_final.pt"
    if checkpoint.is_file() and checkpoint.stat().st_size:
        return Progress(100.0, "checkpoint complete", "done")

    text = read_log(f"train_{method}.log")
    if bad := failure(text):
        return bad

    match = last_match(
        r"epoch\s+(\d+)\s*/\s*(\d+).*?(\d{1,3})%\|",
        text,
        re.IGNORECASE,
    )
    if match:
        epoch, total, inner = map(int, match.groups())
        overall = 100.0 * ((epoch - 1) + inner / 100.0) / max(total, 1)
        return Progress(overall, f"epoch {epoch}/{total}, current epoch {inner}%")

    if match := last_match(r"epoch=(\d+)\b", text):
        epoch = int(match.group(1))
        return Progress(epoch * 5.0, f"completed epoch {epoch}/20")

    if "Identity invariant passed:" in text:
        return Progress(0.5, "identity passed; starting epoch 1")
    if "Trainable parameters:" in text:
        return Progress(0.2, "model initialized")
    return Progress(0.0, "initializing" if text else "not started", "waiting")


def planning_progress(method: str, domain: str) -> Progress:
    result_pattern = f"outputs/**/{method}/{domain}/results.json"
    if any(ROOT.glob(result_pattern)):
        return Progress(100.0, "planning complete", "done")

    text = read_log(f"plan_{method}_{domain}.log")
    if bad := failure(text):
        return bad
    if match := last_match(r"(\d{1,3})%\|", text):
        return Progress(float(match.group(1)), f"planning {method}/{domain}")
    if match := last_match(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", text):
        done, total = map(int, match.groups())
        if 0 < total and done <= total:
            return Progress(100.0 * done / total, f"episode {done}/{total}")
    return Progress(0.0, "initializing" if text else "not started", "waiting")


def active_tasks() -> dict[int, tuple[str, Progress]]:
    tasks: dict[int, tuple[str, Progress]] = {}

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode(errors="replace")
            if not any(name in cmd for name in (
                "build_feature_cache.py",
                "train_adapter.py",
                "scripts/plan.py",
            )):
                continue

            env = {}
            for item in (entry / "environ").read_bytes().split(b"\0"):
                if b"=" in item:
                    key, value = item.split(b"=", 1)
                    env[key.decode(errors="ignore")] = value.decode(errors="ignore")

            visible = env.get("CUDA_VISIBLE_DEVICES", "")
            if not visible:
                continue
            gpu = int(visible.split(",")[0])

            if "build_feature_cache.py" in cmd:
                tasks[gpu] = ("feature cache", cache_progress())
                continue

            method_match = re.search(r"(?:^|\s)method=([\w]+)", cmd)
            method = method_match.group(1) if method_match else "unknown"

            if "train_adapter.py" in cmd:
                tasks[gpu] = (f"train {method}", training_progress(method))
                continue

            domain_match = re.search(r"(?:^|\s)domain=([\w]+)", cmd)
            domain = domain_match.group(1) if domain_match else "unknown"
            tasks[gpu] = (
                f"plan {method}/{domain}",
                planning_progress(method, domain),
            )
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            ValueError,
        ):
            continue

    return tasks


def gpu_stats() -> dict[int, tuple[int, int, int, int]]:
    output = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, stderr=subprocess.DEVNULL)

    stats = {}
    for line in output.splitlines():
        index, util, used, total, temp = [
            int(part.strip()) for part in line.split(",")
        ]
        stats[index] = (util, used, total, temp)
    return stats


def bar(percent: float, width: int = BAR_WIDTH) -> str:
    value = max(0.0, min(100.0, percent))
    filled = round(width * value / 100.0)
    return "█" * filled + "░" * (width - filled)


def marker(state: str) -> str:
    return {
        "running": "RUN ",
        "done": "DONE",
        "failed": "FAIL",
        "waiting": "WAIT",
    }.get(state, state[:4].upper())


def draw() -> None:
    stats = gpu_stats()
    tasks = active_tasks()

    print("\033[2J\033[H", end="")
    print(f"WM Adapter GPU Dashboard | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {ROOT}")
    print("=" * 116)

    for gpu in sorted(stats):
        util, used, total, temp = stats[gpu]
        task, progress = tasks.get(
            gpu,
            ("idle", Progress(0.0, "no active experiment process", "waiting")),
        )
        print(
            f"GPU {gpu}  util {util:3d}%  mem {used:5d}/{total:5d} MiB  "
            f"temp {temp:2d}C  {marker(progress.state)}"
        )
        print(
            f"       [{bar(progress.percent)}] {progress.percent:6.2f}%  "
            f"{task:<28} {progress.label}"
        )
        print("-" * 116)

    print("Training summary:")
    for method in METHODS:
        progress = training_progress(method)
        print(
            f"  {method:<12} [{bar(progress.percent, 20)}] "
            f"{progress.percent:6.2f}%  {progress.label}"
        )

    completed = sum(
        any(ROOT.glob(f"outputs/**/{method}/{domain}/results.json"))
        for method in ("base", *METHODS)
        for domain in ("clean", "ood")
    )
    print(f"\nPlanning results: {completed}/8 complete")
    print("Ctrl+C to exit")


def main() -> None:
    try:
        while True:
            draw()
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
