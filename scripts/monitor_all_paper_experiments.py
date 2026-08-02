from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from wm_adapter.utils.reproducibility import resolve_path


PROGRESS_PATTERN = re.compile(r"PLANNING_PROGRESS\s+(.*)")
FIELD_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s]+)")
STEP_PATTERN = re.compile(r"executing agent.*?(\d+)\s*/\s*(\d+)")
CEM_PATTERN = re.compile(r"Action optim at step\s+(\d+)\s+took\s+([0-9.]+)\s+seconds")
TRAIN_PATTERN = re.compile(r"epoch[= ](\d+)(?:/|\s).*?total[=: ]+([0-9.eE+-]+)")
OFFLINE_PATTERN = re.compile(r"OFFLINE_PROGRESS completed=(\d+) total=(\d+)")


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _gpu_rows() -> list[str]:
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
        return ["GPU telemetry unavailable"]
    return [f"GPU {line}" for line in completed.stdout.strip().splitlines()]


def _tail(path: Path, size: int = 256 * 1024) -> tuple[str, float | None]:
    if not path.is_file():
        return "", None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        length = handle.tell()
        handle.seek(max(0, length - size))
        text = handle.read().decode("utf-8", errors="replace")
    return text, time.time() - path.stat().st_mtime


def _job_detail(entry: dict[str, Any]) -> str:
    path_value = entry.get("log_path")
    if not path_value:
        return "waiting for log"
    text, age = _tail(Path(path_value))
    heartbeat = "log unavailable" if age is None else (
        f"log updated {_duration(age)} ago" if age <= 120 else f"log unchanged {_duration(age)}"
    )
    planning_records: list[dict[str, str]] = []
    for match in PROGRESS_PATTERN.finditer(text):
        planning_records.append(dict(FIELD_PATTERN.findall(match.group(1))))
    if planning_records:
        latest = planning_records[-1]
        total = int(latest.get("total", 0))
        completed = int(latest.get("completed", 0))
        current = int(latest.get("episode", min(completed + 1, total))) if total else 0
        success = int(latest.get("success_count", 0))
        step_matches = list(STEP_PATTERN.finditer(text))
        fraction = 0.0
        step_detail = ""
        if latest.get("phase") == "episode" and latest.get("status") == "started" and step_matches:
            step, step_total = map(int, step_matches[-1].groups())
            if step_total > 0:
                fraction = min(1.0, max(0.0, step / step_total))
                step_detail = f" | step {step}/{step_total}"
        percent = 100.0 * (completed + fraction) / total if total else 0.0
        cem = list(CEM_PATTERN.finditer(text))
        cem_detail = f" | last CEM {float(cem[-1].group(2)):.2f}s" if cem else ""
        completed_durations = [
            float(record["elapsed_seconds"])
            for record in planning_records
            if record.get("phase") == "episode"
            and record.get("status") == "completed"
            and "elapsed_seconds" in record
        ]
        eta_detail = ""
        if completed_durations and total:
            remaining = max(0.0, total - completed - fraction)
            eta_detail = (
                f" | ETA≈{_duration(sum(completed_durations) / len(completed_durations) * remaining)}"
            )
        quiet = ", log quiet" if age is not None and age > 120 else ""
        return (
            f"{percent:.2f}% episode {current}/{total}{step_detail} | completed "
            f"{completed}/{total} | success {success}/{max(completed, 1)}"
            f"{cem_detail}{eta_detail} | {heartbeat}{quiet}"
        )
    offline = list(OFFLINE_PATTERN.finditer(text))
    if offline:
        completed, total = map(int, offline[-1].groups())
        percent = 100.0 * completed / total if total else 0.0
        return f"{percent:.2f}% offline windows {completed}/{total} | {heartbeat}"
    training = list(TRAIN_PATTERN.finditer(text))
    if training:
        epoch, loss = training[-1].groups()
        return f"training epoch {epoch} | loss {loss} | {heartbeat}"
    return f"initialization | {heartbeat}"


def _render(state_path: Path) -> str:
    if not state_path.is_file():
        return f"Paper-suite state not found: {state_path}"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    jobs = state.get("jobs", {})
    counts: dict[str, int] = {}
    for entry in jobs.values():
        status = str(entry.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    expected_jobs = int(state.get("expected_jobs", len(jobs)))
    counts["pending"] = max(0, expected_jobs - len(jobs))
    active_phases = sorted(
        {str(entry.get("phase")) for entry in jobs.values() if entry.get("status") == "running"}
    )
    lines = [
        f"ICRA 2027 paper suite | status={state.get('status', 'unknown')} | phase={','.join(active_phases) or 'idle'}",
        "jobs " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        *_gpu_rows(),
        "",
    ]
    completed_times = [
        float(entry.get("elapsed_seconds", 0.0))
        for entry in jobs.values()
        if entry.get("status") in {"completed", "reused"} and float(entry.get("elapsed_seconds", 0.0)) > 0
    ]
    pending = counts["pending"] + sum(
        entry.get("status") == "blocked" for entry in jobs.values()
    )
    if completed_times and pending:
        lines.append(f"overall ETA≈{_duration(sum(completed_times) / len(completed_times) * pending)}")
    for job_id, entry in sorted(jobs.items()):
        status = str(entry.get("status", "unknown")).upper()
        if status == "RUNNING":
            elapsed = time.time() - float(entry.get("started_at_unix", time.time()))
            lines.append(
                f"[{status}] {job_id} GPU={entry.get('gpu')} PID={entry.get('pid')} "
                f"elapsed={_duration(elapsed)}"
            )
            lines.append(f"  {_job_detail(entry)}")
        elif status == "FAILED":
            lines.append(f"[{status}] {job_id}: {entry.get('error', '')}")
    lines.append(f"state updated {_duration(time.time() - state_path.stat().st_mtime)} ago")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="logs/paper_suite/state.json")
    parser.add_argument("--refresh", type=float, default=2.0)
    arguments = parser.parse_args()
    if arguments.refresh <= 0:
        raise ValueError(f"refresh must be positive, received {arguments.refresh}")
    state_path = resolve_path(arguments.state)
    try:
        while True:
            print("\033[2J\033[H" + _render(state_path), flush=True)
            time.sleep(arguments.refresh)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
