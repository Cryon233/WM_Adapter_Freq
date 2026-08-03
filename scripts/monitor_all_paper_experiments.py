#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from wm_adapter.utils.reproducibility import project_root, resolve_path


ROOT = project_root()
DEFAULT_STATE = ROOT / "logs" / "paper_suite" / "state.json"
DEFAULT_SUITE = ROOT / "configs" / "experiment" / "icra2027_suite.yaml"
REFRESH = float(os.environ.get("MONITOR_REFRESH", "2"))
BAR_WIDTH = int(os.environ.get("MONITOR_BAR_WIDTH", "28"))
RECENT_COMPLETED = int(os.environ.get("MONITOR_RECENT_COMPLETED", "4"))

PROGRESS_PATTERN = re.compile(r"PLANNING_PROGRESS\s+(.*)")
FIELD_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s]+)")
STEP_PATTERN = re.compile(r"executing agent.*?(\d+)\s*/\s*(\d+)")
CEM_PATTERN = re.compile(
    r"Action optim at step\s+(\d+)\s+took\s+([0-9]+(?:\.[0-9]+)?)\s+seconds"
)
OFFLINE_PATTERN = re.compile(r"OFFLINE_PROGRESS completed=(\d+) total=(\d+)")
CACHE_PROGRESS_PATTERN = re.compile(
    r"CACHE_PROGRESS\s+completed=(\d+)\s+total=(\d+)\s+rate=([^\s]+)"
)
PROTOCOL_PROGRESS_PATTERN = re.compile(
    r"PROTOCOL_PROGRESS\s+task=([^\s]+)\s+completed=(\d+)\s+total=(\d+)"
)
TRAIN_PROGRESS_PATTERN = re.compile(r"TRAIN_PROGRESS\s+(.*)")
TRAIN_TQDM_PATTERN = re.compile(
    r"epoch\s+(\d+)\s*/\s*(\d+):[^\n]*?(\d+)\s*/\s*(\d+)"
)
TRAIN_SUMMARY_PATTERN = re.compile(
    r"epoch[= ](\d+)(?:/(\d+))?.*?total[=: ]+([0-9.eE+-]+)"
)
GENERIC_TQDM_PATTERNS = (
    re.compile(r"(\d{1,3})%\|[^\n]*?\|\s*(\d+)\s*/\s*(\d+)"),
    re.compile(r"(\d+)\s*/\s*(\d+)[^\n]*?(\d{1,3})%"),
)


@dataclass(frozen=True)
class Progress:
    percent: float
    state: str
    detail: str


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    label: str
    expected: int


def clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def bar(percent: float, width: int | None = None, active: bool = False) -> str:
    width = max(8, width or BAR_WIDTH)
    value = clamp(percent)
    filled = int(round(width * value / 100.0))
    if active and filled == 0:
        position = int(time.time() * 2) % width
        cells = ["-"] * width
        cells[position] = ">"
        return "".join(cells)
    return "#" * filled + "-" * (width - filled)


def state_marker(state: str) -> str:
    return {
        "DONE": "[OK]  ",
        "RUN": "[RUN] ",
        "WAIT": "[--]  ",
        "FAIL": "[FAIL]",
    }.get(state, f"[{state}]")


def human_duration(seconds: float | int) -> str:
    value = max(0, int(seconds))
    if value < 60:
        return f"{value}s"
    if value < 3600:
        return f"{value // 60}m{value % 60:02d}s"
    return f"{value // 3600}h{(value % 3600) // 60:02d}m"


def tail_text(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()
            data = handle.read()
        return data.decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return ""


def file_age(path: Path) -> int | None:
    try:
        return max(0, int(time.time() - path.stat().st_mtime))
    except OSError:
        return None


def log_heartbeat(path: Path) -> tuple[str, bool]:
    age = file_age(path)
    if age is None:
        return "no log", False
    if age > 120:
        return f"log unchanged {human_duration(age)}", True
    return f"log updated {human_duration(age)} ago", False


def latest_meaningful_line(text: str) -> str:
    ignored = (
        "Skipping incompatible RoboCasa object",
        "warnings.warn(",
        "Gym has been unmaintained",
        "Please upgrade to Gymnasium",
        "No OpenGL_accelerate",
        "Could not import robosuite_models",
        "Could not load the mink-based",
        "mimicgen environments not imported",
    )
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if line and not any(token in line for token in ignored):
            return line[:140]
    return "waiting for log output"


def tail_has_failure(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    fatal_tokens = (
        "RuntimeError:",
        "InterpolationKeyError:",
        "AttributeError:",
        "ModuleNotFoundError:",
        "FileNotFoundError:",
        "CUDA out of memory",
        "Fatal Python error:",
    )
    recent = lines[-30:]
    if any(token in recent[-1] for token in fatal_tokens):
        return True
    return any("Traceback (most recent call last)" in line for line in recent) and any(
        token in line for line in recent for token in fatal_tokens
    )


def parse_tqdm(text: str) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for pattern_index, pattern in enumerate(GENERIC_TQDM_PATTERNS):
        for match in pattern.finditer(text):
            if pattern_index == 0:
                percent, done, total = map(int, match.groups())
            else:
                done, total, percent = map(int, match.groups())
            if total <= 0 or done > total:
                continue
            candidates.append((match.start(), percent, done, total))
    if not candidates:
        return None
    _, percent, done, total = max(candidates, key=lambda item: item[0])
    return percent, done, total


def parse_planning(text: str) -> Progress | None:
    records = [
        dict(FIELD_PATTERN.findall(match.group(1)))
        for match in PROGRESS_PATTERN.finditer(text)
    ]
    if not records:
        return None
    latest = records[-1]
    total = int(latest.get("total", latest.get("total_episodes", 0)))
    completed = int(latest.get("completed", 0))
    success = int(latest.get("success_count", 0))
    current = int(latest.get("episode", min(completed + 1, total))) if total else 0
    episode_fraction = 0.0
    step_detail = ""
    if latest.get("phase") == "episode" and latest.get("status") == "started":
        step_matches = list(STEP_PATTERN.finditer(text))
        if step_matches:
            step, step_total = map(int, step_matches[-1].groups())
            if step_total > 0:
                episode_fraction = min(1.0, max(0.0, step / step_total))
                step_detail = f"step {step}/{step_total} | "
    percent = 100.0 * (completed + episode_fraction) / total if total else 0.0
    cem_matches = list(CEM_PATTERN.finditer(text))
    cem_detail = (
        f" | last CEM {float(cem_matches[-1].group(2)):.2f}s"
        if cem_matches
        else ""
    )
    durations = [
        float(record["elapsed_seconds"])
        for record in records
        if record.get("phase") == "episode"
        and record.get("status") == "completed"
        and "elapsed_seconds" in record
    ]
    timing = ""
    if durations:
        average = sum(durations) / len(durations)
        remaining = max(0.0, total - completed - episode_fraction)
        timing = (
            f" | last episode {durations[-1]:.1f}s"
            f" | average {average:.1f}s"
            f" | ETA≈{human_duration(average * remaining)}"
        )
    if latest.get("phase") == "job" and latest.get("status") == "completed":
        return Progress(
            100.0,
            "DONE",
            f"completed {completed}/{total} | success {success}/{max(total, 1)}",
        )
    detail = (
        f"episode {current}/{total} | {step_detail}"
        f"completed {completed}/{total}"
    )
    if completed:
        detail += f" | success {success}/{completed}"
    return Progress(percent, "RUN", detail + cem_detail + timing)


def parse_training(text: str, heartbeat: str) -> Progress | None:
    structured = [
        dict(FIELD_PATTERN.findall(match.group(1)))
        for match in TRAIN_PROGRESS_PATTERN.finditer(text)
    ]
    if structured:
        latest = structured[-1]
        step = int(latest["step"])
        total = int(latest["total"])
        recent = structured[-10:]
        if (
            len(recent) >= 2
            and "elapsed_seconds" in recent[0]
            and "elapsed_seconds" in recent[-1]
        ):
            first_step = int(recent[0]["step"])
            last_step = int(recent[-1]["step"])
            elapsed_delta = float(recent[-1]["elapsed_seconds"]) - float(
                recent[0]["elapsed_seconds"]
            )
            step_rate = max(last_step - first_step, 0) / max(elapsed_delta, 1.0e-9)
        else:
            step_rate = 0.0
        eta = (
            f" | ETA≈{human_duration((total - step) / step_rate)}"
            if step_rate > 0 and len(structured) >= 10
            else ""
        )
        detail = (
            f"step {step}/{total} | loss {latest.get('loss', 'n/a')} | "
            f"clean {latest.get('clean_mse', 'n/a')} | OOD {latest.get('ood_mse', 'n/a')} | "
            f"future {latest.get('future_mse', 'n/a')} | lr {latest.get('lr', 'n/a')} | "
            f"grad {latest.get('grad_norm', 'n/a')} | samples/s {latest.get('samples_per_sec', 'n/a')} | "
            f"core {latest.get('core_delta_ratio', 'n/a')} | spectral {latest.get('spectral_delta_ratio', 'n/a')}"
            f"{eta} | {heartbeat}"
        )
        return Progress(100.0 * step / max(total, 1), "RUN", detail)
    tqdm_matches = list(TRAIN_TQDM_PATTERN.finditer(text))
    if tqdm_matches:
        epoch, epochs, batch, batches = map(int, tqdm_matches[-1].groups())
        if epochs > 0 and batches > 0:
            percent = 100.0 * ((epoch - 1) + batch / batches) / epochs
            summaries = list(TRAIN_SUMMARY_PATTERN.finditer(text))
            loss = summaries[-1].group(3) if summaries else "n/a"
            return Progress(
                percent,
                "RUN",
                f"epoch {epoch}/{epochs}, batch {batch}/{batches} | loss {loss} | {heartbeat}",
            )
    summaries = list(TRAIN_SUMMARY_PATTERN.finditer(text))
    if summaries:
        epoch, epochs, loss = summaries[-1].groups()
        if epochs is not None and int(epochs) > 0:
            return Progress(
                100.0 * int(epoch) / int(epochs),
                "RUN",
                f"completed epoch {epoch}/{epochs} | loss {loss} | {heartbeat}",
            )
        return Progress(
            0.0,
            "RUN",
            f"completed epoch {epoch} | loss {loss} | {heartbeat}",
        )
    return None


def artifact_detail(entry: dict[str, Any]) -> str:
    artifact = entry.get("artifact")
    if not isinstance(artifact, dict):
        if entry.get("status") == "reused":
            return "existing artifact reused"
        return "artifact validated"
    if "success_count" in artifact:
        used = int(artifact.get("used_episodes", artifact.get("available_episodes", 0)))
        success = int(artifact["success_count"])
        return f"completed {used}/{used} | success {success}/{used}"
    if "window_count" in artifact:
        return f"{artifact['window_count']} windows complete"
    if "parameter_count" in artifact:
        steps = artifact.get("completed_optimizer_steps")
        suffix = f" | steps {steps}" if steps is not None else ""
        return f"checkpoint complete | params {artifact['parameter_count']}{suffix}"
    if "cache_fingerprint" in artifact:
        windows = artifact.get("available_windows", artifact.get("used_windows", "?"))
        return f"cache {windows} windows | fingerprint {str(artifact['cache_fingerprint'])[:12]}"
    if artifact.get("status") in {"passed", "not_applicable"}:
        sequence = artifact.get("sequence_replay") or {}
        detail = f"protocol {artifact['status']}"
        if sequence:
            detail += f" | qpos MAE {sequence.get('qpos_mae', 'n/a')}"
        return detail
    if "source_path" in artifact:
        return "existing result reused"
    return "artifact validated"


def job_progress(job_id: str, entry: dict[str, Any]) -> Progress:
    status = str(entry.get("status", "waiting"))
    if status in {"completed", "reused"}:
        return Progress(100.0, "DONE", artifact_detail(entry))
    if status == "failed":
        return Progress(0.0, "FAIL", str(entry.get("error", "job failed")))
    if status == "blocked":
        return Progress(0.0, "WAIT", "blocked after another job failed")
    if status != "running":
        return Progress(0.0, "WAIT", "not started")

    log_path_value = entry.get("log_path")
    log_path = Path(log_path_value) if log_path_value else Path()
    text = tail_text(log_path) if log_path_value else ""
    heartbeat, quiet = log_heartbeat(log_path) if log_path_value else ("no log", False)

    planning = parse_planning(text)
    if planning is not None:
        detail = f"{planning.detail} | {heartbeat}"
        if quiet:
            detail += " | RUN, log quiet"
        return Progress(planning.percent, "RUN", detail)

    offline_matches = list(OFFLINE_PATTERN.finditer(text))
    if offline_matches:
        completed, total = map(int, offline_matches[-1].groups())
        percent = 100.0 * completed / total if total else 0.0
        return Progress(
            percent,
            "RUN",
            f"offline windows {completed}/{total} | {heartbeat}",
        )

    cache_matches = list(CACHE_PROGRESS_PATTERN.finditer(text))
    if cache_matches:
        completed, total = map(int, cache_matches[-1].groups()[:2])
        rate = cache_matches[-1].group(3)
        return Progress(
            100.0 * completed / max(total, 1), "RUN",
            f"cache windows {completed}/{total} | {rate} windows/s | {heartbeat}",
        )

    protocol_matches = list(PROTOCOL_PROGRESS_PATTERN.finditer(text))
    if protocol_matches:
        task, completed, total = protocol_matches[-1].groups()
        return Progress(
            100.0 * int(completed) / max(int(total), 1), "RUN",
            f"protocol {task} {completed}/{total} | {heartbeat}",
        )

    training = parse_training(text, heartbeat)
    if training is not None:
        return training

    tqdm = parse_tqdm(text)
    if tqdm is not None:
        percent, done, total = tqdm
        return Progress(
            float(percent),
            "RUN",
            f"{done}/{total} | {heartbeat}",
        )

    if tail_has_failure(text):
        return Progress(0.0, "FAIL", latest_meaningful_line(text))
    detail = latest_meaningful_line(text)
    if quiet:
        detail += f" | {heartbeat}"
    return Progress(0.0, "RUN", detail)


def load_phase_specs(suite_path: Path) -> list[PhaseSpec]:
    try:
        suite = OmegaConf.load(suite_path)
        specs = [
            PhaseSpec("cache", "1. Feature cache", 1),
            PhaseSpec("training", "2. Main adapter training", 3),
            PhaseSpec(
                "ablation_training",
                "3. DCT ablation training",
                len(suite.dct_ablations),
            ),
            PhaseSpec(
                "offline",
                "4. Offline dynamics",
                4 + len(suite.dct_ablations),
            ),
            PhaseSpec(
                "planning_main",
                "5. Main planning",
                len(suite.tasks) * len(suite.methods) * len(suite.domains),
            ),
            PhaseSpec(
                "planning_multiseed",
                "6. Multi-seed planning",
                len(suite.multiseed.tasks)
                * len(suite.multiseed.methods)
                * len(suite.multiseed.domains)
                * len(suite.multiseed.seeds),
            ),
            PhaseSpec(
                "planning_severity",
                "7. OOD severity",
                len(suite.severity.tasks)
                * len(suite.severity.methods)
                * len(suite.severity.values),
            ),
            PhaseSpec(
                "planning_ablation",
                "8. Closed-loop ablations",
                len(suite.closed_loop_ablations) * len(suite.ablation_tasks),
            ),
            PhaseSpec("analysis", "9. Final analysis", 1),
        ]
        return specs
    except Exception:
        return [
            PhaseSpec("cache", "1. Feature cache", 1),
            PhaseSpec("training", "2. Main adapter training", 3),
            PhaseSpec("ablation_training", "3. DCT ablation training", 10),
            PhaseSpec("offline", "4. Offline dynamics", 14),
            PhaseSpec("planning_main", "5. Main planning", 32),
            PhaseSpec("planning_multiseed", "6. Multi-seed planning", 24),
            PhaseSpec("planning_severity", "7. OOD severity", 12),
            PhaseSpec("planning_ablation", "8. Closed-loop ablations", 8),
            PhaseSpec("analysis", "9. Final analysis", 1),
        ]


def phase_progress(
    spec: PhaseSpec,
    jobs: dict[str, dict[str, Any]],
) -> Progress:
    entries = [
        (job_id, entry)
        for job_id, entry in jobs.items()
        if str(entry.get("phase", "")) == spec.key
    ]
    completed = sum(
        str(entry.get("status")) in {"completed", "reused"}
        for _, entry in entries
    )
    running = [
        job_progress(job_id, entry)
        for job_id, entry in entries
        if str(entry.get("status")) == "running"
    ]
    failed = sum(str(entry.get("status")) == "failed" for _, entry in entries)
    blocked = sum(str(entry.get("status")) == "blocked" for _, entry in entries)
    partial = sum(progress.percent / 100.0 for progress in running)
    percent = 100.0 * (completed + partial) / max(spec.expected, 1)
    if failed:
        state = "FAIL"
    elif completed >= spec.expected:
        state = "DONE"
    elif running:
        state = "RUN"
    else:
        state = "WAIT"
    detail = (
        f"{completed}/{spec.expected} jobs complete"
        f" | running {len(running)}"
        f" | pending {max(0, spec.expected - completed - len(running) - failed - blocked)}"
    )
    if failed:
        detail += f" | failed {failed}"
    if blocked:
        detail += f" | blocked {blocked}"
    return Progress(percent, state, detail)


def overall_progress(
    specs: list[PhaseSpec],
    jobs: dict[str, dict[str, Any]],
) -> float:
    total = sum(spec.expected for spec in specs)
    if total <= 0:
        return 0.0
    completed_units = 0.0
    for spec in specs:
        phase = phase_progress(spec, jobs)
        completed_units += phase.percent / 100.0 * spec.expected
    return clamp(100.0 * completed_units / total)


def gpu_stats() -> dict[str, dict[str, int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    stats: dict[str, dict[str, int]] = {}
    for line in output.splitlines():
        try:
            index, util, used, total, temperature = [
                int(part.strip()) for part in line.split(",")
            ]
        except (ValueError, IndexError):
            continue
        stats[str(index)] = {
            "util": util,
            "used": used,
            "total": total,
            "temp": temperature,
        }
    return stats


def process_elapsed(entry: dict[str, Any]) -> int:
    started = float(entry.get("started_at_unix", time.time()))
    return max(0, int(time.time() - started))


def print_progress(label: str, progress: Progress, indent: str = "") -> None:
    active = progress.state == "RUN" and progress.percent == 0.0
    print(
        f"{indent}{label:<29} {state_marker(progress.state)} "
        f"[{bar(progress.percent, active=active)}] {clamp(progress.percent):6.2f}%  "
        f"{progress.detail}"
    )


def load_state(state_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("state root is not a mapping")
        return payload
    except FileNotFoundError:
        return {
            "suite": "icra2027",
            "status": "waiting",
            "jobs": {},
            "expected_jobs": 105,
        }
    except Exception as error:
        return {
            "suite": "icra2027",
            "status": "failed",
            "jobs": {},
            "expected_jobs": 105,
            "error": f"Cannot read state: {type(error).__name__}: {error}",
        }


def draw(state_path: Path, suite_path: Path) -> None:
    state = load_state(state_path)
    jobs: dict[str, dict[str, Any]] = state.get("jobs", {})
    specs = load_phase_specs(suite_path)
    overall = overall_progress(specs, jobs)
    stats = gpu_stats()
    running_by_gpu: dict[str, tuple[str, dict[str, Any]]] = {}
    running_jobs: list[tuple[str, dict[str, Any]]] = []
    for job_id, entry in jobs.items():
        if str(entry.get("status")) != "running":
            continue
        running_jobs.append((job_id, entry))
        gpu = entry.get("gpu")
        if gpu is not None:
            running_by_gpu[str(gpu)] = (job_id, entry)

    completed_count = sum(
        str(entry.get("status")) in {"completed", "reused"}
        for entry in jobs.values()
    )
    failed_count = sum(
        str(entry.get("status")) == "failed" for entry in jobs.values()
    )
    blocked_count = sum(
        str(entry.get("status")) == "blocked" for entry in jobs.values()
    )
    expected_jobs = int(state.get("expected_jobs", sum(spec.expected for spec in specs)))
    pending_count = max(
        0,
        expected_jobs
        - completed_count
        - len(running_jobs)
        - failed_count
        - blocked_count,
    )

    sys.stdout.write("\033[2J\033[H")
    print(f"WM Adapter ICRA Paper-Suite Dashboard  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {ROOT}")
    print(f"State:   {state_path}")
    print(f"Overall [{bar(overall, 42)}] {overall:6.2f}%")
    print(
        f"Jobs: {completed_count}/{expected_jobs} complete | "
        f"{len(running_jobs)} running | {pending_count} pending | "
        f"{failed_count} failed | {blocked_count} blocked"
    )
    print("=" * 140)

    print("\nPIPELINE")
    for spec in specs:
        print_progress(spec.label, phase_progress(spec, jobs))

    print("\nGPU TASKS")
    if not stats:
        print("  nvidia-smi unavailable")
    else:
        for gpu in sorted(stats, key=int):
            stat = stats[gpu]
            running = running_by_gpu.get(gpu)
            if running is None:
                task_label = "idle"
            else:
                job_id, entry = running
                task_label = (
                    f"{job_id} pid={entry.get('pid', '?')} "
                    f"time={human_duration(process_elapsed(entry))}"
                )
            print(
                f"  GPU {gpu} | util {stat['util']:3d}% | "
                f"mem {stat['used']:5d}/{stat['total']:5d} MiB | "
                f"temp {stat['temp']:2d}C | {task_label}"
            )

    print("\nRUNNING JOBS")
    if not running_jobs:
        print("  no active paper-suite jobs")
    else:
        for job_id, entry in sorted(
            running_jobs,
            key=lambda item: int(item[1].get("gpu", 999)),
        ):
            progress = job_progress(job_id, entry)
            gpu = entry.get("gpu", "?")
            label = f"GPU {gpu} {job_id}"
            print_progress(label, progress, indent="  ")

    failed_jobs = [
        (job_id, entry)
        for job_id, entry in jobs.items()
        if str(entry.get("status")) == "failed"
    ]
    if failed_jobs:
        print("\nFAILURES")
        for job_id, entry in failed_jobs:
            print(f"  [FAIL] {job_id}: {entry.get('error', 'unknown failure')}")

    recent = sorted(
        [
            (job_id, entry)
            for job_id, entry in jobs.items()
            if str(entry.get("status")) in {"completed", "reused"}
        ],
        key=lambda item: float(
            item[1].get("ended_at_unix", item[1].get("started_at_unix", 0.0))
        ),
        reverse=True,
    )[:RECENT_COMPLETED]
    if recent:
        print("\nRECENTLY COMPLETED")
        for job_id, entry in recent:
            status = "REUSED" if str(entry.get("status")) == "reused" else "DONE"
            elapsed = float(entry.get("elapsed_seconds", 0.0))
            suffix = f" | time {human_duration(elapsed)}" if elapsed > 0 else ""
            print(f"  [{status}] {job_id}{suffix} | {artifact_detail(entry)}")

    if state.get("status") == "failed" and state.get("error"):
        print(f"\nSUITE ERROR\n  {state['error']}")

    state_age = file_age(state_path)
    age_text = "state not created" if state_age is None else f"state updated {human_duration(state_age)} ago"
    print(
        "\nProgress uses the scheduler state plus structured training/offline/planning logs."
    )
    print(f"{age_text} | Ctrl+C to exit | refresh {REFRESH:.1f}s")
    sys.stdout.flush()


def main() -> None:
    global REFRESH
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--suite-config", default=str(DEFAULT_SUITE))
    parser.add_argument("--refresh", type=float, default=REFRESH)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh <= 0:
        raise ValueError(f"refresh must be positive, received {arguments.refresh}")
    REFRESH = arguments.refresh
    state_path = resolve_path(arguments.state)
    suite_path = resolve_path(arguments.suite_config)
    try:
        while True:
            draw(state_path, suite_path)
            if arguments.once:
                return
            time.sleep(arguments.refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
