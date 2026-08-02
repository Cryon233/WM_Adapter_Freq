from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from wm_adapter.utils.reproducibility import resolve_path


PROGRESS_PATTERN = re.compile(r"PLANNING_PROGRESS\s+(.*)")
FIELD_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s]+)")
STEP_PATTERN = re.compile(r"executing agent.*?(\d+)\s*/\s*(\d+)")
CEM_PATTERN = re.compile(r"Action optim at step\s+(\d+)\s+took\s+([0-9.]+)\s+seconds")
TRAIN_SUMMARY_PATTERN = re.compile(
    r"epoch[= ](\d+)(?:/(\d+))?.*?total[=: ]+([0-9.eE+-]+)"
)
TRAIN_TQDM_PATTERN = re.compile(
    r"epoch\s+(\d+)/(\d+):.*?(\d+)/(\d+)\s*\["
)
OFFLINE_PATTERN = re.compile(r"OFFLINE_PROGRESS completed=(\d+) total=(\d+)")
GENERIC_TQDM_PATTERN = re.compile(r"(\d{1,3})%\|.*?\|\s*(\d+)/(\d+)")


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _terminal_width() -> int:
    return max(80, shutil.get_terminal_size(fallback=(120, 30)).columns)


def _progress_bar(progress: float, width: int) -> str:
    bounded = min(1.0, max(0.0, progress))
    width = max(8, width)
    filled = min(width, int(round(bounded * width)))
    return "█" * filled + "░" * (width - filled)


def _gpu_rows() -> dict[int, dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for raw_line in completed.stdout.strip().splitlines():
        fields = [value.strip() for value in raw_line.split(",")]
        if len(fields) != 6:
            continue
        try:
            index = int(fields[0])
            rows[index] = {
                "name": fields[1],
                "utilization": float(fields[2]),
                "memory_used": float(fields[3]),
                "memory_total": float(fields[4]),
                "temperature": float(fields[5]),
            }
        except ValueError:
            continue
    return rows


def _tail(path: Path, size: int = 256 * 1024) -> tuple[str, float | None]:
    if not path.is_file():
        return "", None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        length = handle.tell()
        handle.seek(max(0, length - size))
        text = handle.read().decode("utf-8", errors="replace")
    return text, time.time() - path.stat().st_mtime


def _job_progress(entry: dict[str, Any]) -> tuple[float | None, str]:
    path_value = entry.get("log_path")
    if not path_value:
        return None, "waiting for log"
    text, age = _tail(Path(path_value))
    heartbeat = (
        "log unavailable"
        if age is None
        else (
            f"log {_duration(age)} ago"
            if age <= 120
            else f"log quiet {_duration(age)}"
        )
    )

    planning_records = [
        dict(FIELD_PATTERN.findall(match.group(1)))
        for match in PROGRESS_PATTERN.finditer(text)
    ]
    if planning_records:
        latest = planning_records[-1]
        total = int(latest.get("total", 0))
        completed = int(latest.get("completed", 0))
        current = (
            int(latest.get("episode", min(completed + 1, total)))
            if total
            else 0
        )
        success = int(latest.get("success_count", 0))
        step_matches = list(STEP_PATTERN.finditer(text))
        episode_fraction = 0.0
        step_detail = ""
        if (
            latest.get("phase") == "episode"
            and latest.get("status") == "started"
            and step_matches
        ):
            step, step_total = map(int, step_matches[-1].groups())
            if step_total > 0:
                episode_fraction = min(1.0, max(0.0, step / step_total))
                step_detail = f"step {step}/{step_total} | "
        progress = (
            min(1.0, max(0.0, (completed + episode_fraction) / total))
            if total
            else None
        )
        cem = list(CEM_PATTERN.finditer(text))
        cem_detail = (
            f" | CEM {float(cem[-1].group(2)):.1f}s"
            if cem
            else ""
        )
        completed_durations = [
            float(record["elapsed_seconds"])
            for record in planning_records
            if record.get("phase") == "episode"
            and record.get("status") == "completed"
            and "elapsed_seconds" in record
        ]
        eta_detail = ""
        if completed_durations and total:
            remaining = max(0.0, total - completed - episode_fraction)
            average = sum(completed_durations) / len(completed_durations)
            eta_detail = f" | ETA {_duration(average * remaining)}"
        detail = (
            f"episode {current}/{total} | {step_detail}"
            f"done {completed}/{total} | success {success}/{max(completed, 1)}"
            f"{cem_detail}{eta_detail} | {heartbeat}"
        )
        return progress, detail

    offline = list(OFFLINE_PATTERN.finditer(text))
    if offline:
        completed, total = map(int, offline[-1].groups())
        progress = min(1.0, completed / total) if total else None
        return progress, f"offline windows {completed}/{total} | {heartbeat}"

    training_tqdm = list(TRAIN_TQDM_PATTERN.finditer(text))
    if training_tqdm:
        epoch, epochs, batch, batches = map(int, training_tqdm[-1].groups())
        progress = (
            min(1.0, max(0.0, ((epoch - 1) + batch / batches) / epochs))
            if epochs > 0 and batches > 0
            else None
        )
        summaries = list(TRAIN_SUMMARY_PATTERN.finditer(text))
        loss = summaries[-1].group(3) if summaries else "n/a"
        return (
            progress,
            f"training epoch {epoch}/{epochs} batch {batch}/{batches} | loss {loss} | {heartbeat}",
        )

    training = list(TRAIN_SUMMARY_PATTERN.finditer(text))
    if training:
        epoch, epochs, loss = training[-1].groups()
        if epochs is not None and int(epochs) > 0:
            progress = min(1.0, int(epoch) / int(epochs))
            return progress, f"training epoch {epoch}/{epochs} | loss {loss} | {heartbeat}"
        return None, f"training epoch {epoch} | loss {loss} | {heartbeat}"

    generic = list(GENERIC_TQDM_PATTERN.finditer(text))
    if generic:
        percent, completed, total = map(int, generic[-1].groups())
        return min(1.0, percent / 100.0), f"progress {completed}/{total} | {heartbeat}"

    return None, f"initialization | {heartbeat}"


def _overall_progress(
    jobs: dict[str, dict[str, Any]], expected_jobs: int
) -> float:
    if expected_jobs <= 0:
        return 0.0
    completed_units = 0.0
    for entry in jobs.values():
        status = str(entry.get("status", "unknown"))
        if status in {"completed", "reused"}:
            completed_units += 1.0
        elif status == "running":
            progress, _ = _job_progress(entry)
            completed_units += 0.0 if progress is None else progress
    return min(1.0, completed_units / expected_jobs)


def _render_gpu(
    gpu: int,
    telemetry: dict[str, Any] | None,
    running_entry: tuple[str, dict[str, Any]] | None,
    width: int,
) -> list[str]:
    lines: list[str] = []
    if telemetry is None:
        gpu_summary = f"GPU {gpu} telemetry unavailable"
    else:
        util = float(telemetry["utilization"])
        memory_used = float(telemetry["memory_used"])
        memory_total = float(telemetry["memory_total"])
        temperature = float(telemetry["temperature"])
        util_bar = _progress_bar(util / 100.0, 10)
        gpu_summary = (
            f"GPU {gpu} [{util_bar}] {util:5.1f}% | "
            f"mem {memory_used / 1024:.1f}/{memory_total / 1024:.1f} GiB | "
            f"{temperature:.0f}C"
        )
    lines.append(gpu_summary)

    if running_entry is None:
        lines.append("  [IDLE] no active paper-suite job")
        return lines

    job_id, entry = running_entry
    elapsed = time.time() - float(entry.get("started_at_unix", time.time()))
    progress, detail = _job_progress(entry)
    available = max(12, min(42, width - 25))
    if progress is None:
        lines.append(
            f"  [RUN] {job_id} | elapsed {_duration(elapsed)}"
        )
        lines.append(f"        {detail}")
    else:
        lines.append(
            f"  [{_progress_bar(progress, available)}] {progress * 100:6.2f}%"
        )
        lines.append(
            f"  {job_id} | elapsed {_duration(elapsed)}"
        )
        lines.append(f"  {detail}")
    return lines


def _render(state_path: Path) -> str:
    width = _terminal_width()
    if not state_path.is_file():
        return (
            f"Paper-suite state not found: {state_path}\n"
            f"[{_progress_bar(0.0, min(50, width - 12))}]   0.00%"
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    jobs: dict[str, dict[str, Any]] = state.get("jobs", {})
    expected_jobs = int(state.get("expected_jobs", len(jobs)))
    counts: dict[str, int] = {}
    for entry in jobs.values():
        status = str(entry.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    counts["pending"] = max(0, expected_jobs - len(jobs))

    active_phases = sorted(
        {
            str(entry.get("phase"))
            for entry in jobs.values()
            if entry.get("status") == "running"
        }
    )
    overall = _overall_progress(jobs, expected_jobs)
    overall_bar_width = max(20, min(56, width - 26))
    status_counts = " ".join(
        f"{key}={value}" for key, value in sorted(counts.items())
    )
    lines = [
        "ICRA 2027 PAPER EXPERIMENTS",
        (
            f"[{_progress_bar(overall, overall_bar_width)}] "
            f"{overall * 100:6.2f}%"
        ),
        (
            f"status={state.get('status', 'unknown')} | "
            f"phase={','.join(active_phases) or 'idle'} | "
            f"jobs {status_counts}"
        ),
        "",
    ]

    telemetry = _gpu_rows()
    running_by_gpu: dict[int, tuple[str, dict[str, Any]]] = {}
    for job_id, entry in jobs.items():
        if entry.get("status") == "running" and entry.get("gpu") is not None:
            running_by_gpu[int(entry["gpu"])] = (job_id, entry)

    gpu_ids = sorted(set(telemetry).union(running_by_gpu))
    if not gpu_ids:
        lines.append("GPU telemetry unavailable")
    else:
        for index, gpu in enumerate(gpu_ids):
            if index:
                lines.append("")
            lines.extend(
                _render_gpu(
                    gpu,
                    telemetry.get(gpu),
                    running_by_gpu.get(gpu),
                    width,
                )
            )

    failed = [
        (job_id, entry)
        for job_id, entry in jobs.items()
        if entry.get("status") == "failed"
    ]
    if failed:
        lines.extend(["", "FAILED JOBS"])
        for job_id, entry in failed:
            lines.append(f"- {job_id}: {entry.get('error', '')}")

    completed_times = [
        float(entry.get("elapsed_seconds", 0.0))
        for entry in jobs.values()
        if entry.get("status") == "completed"
        and float(entry.get("elapsed_seconds", 0.0)) > 0
    ]
    pending_jobs = counts["pending"] + sum(
        entry.get("status") == "blocked" for entry in jobs.values()
    )
    if completed_times and pending_jobs:
        naive_eta = (
            sum(completed_times) / len(completed_times) * pending_jobs
        )
        lines.extend(
            [
                "",
                (
                    "rough job-count ETA "
                    f"{_duration(naive_eta)} "
                    "(jobs have different costs)"
                ),
            ]
        )

    lines.extend(
        [
            "",
            f"state updated {_duration(time.time() - state_path.stat().st_mtime)} ago",
            "Ctrl+C exits monitor only; experiments keep running.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="logs/paper_suite/state.json")
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh <= 0:
        raise ValueError(
            f"refresh must be positive, received {arguments.refresh}"
        )
    state_path = resolve_path(arguments.state)
    try:
        while True:
            print("\033[2J\033[H" + _render(state_path), flush=True)
            if arguments.once:
                return
            time.sleep(arguments.refresh)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
