#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
LOG_ROOT = ROOT / "logs" / "full_a100_experiment"
CACHE_PATH = ROOT / "storage" / "feature_cache" / "jepa_wm_droid_robocasa_pilot.h5"
CKPT_ROOT = ROOT / "checkpoints" / "jepa_wm_droid" / "robocasa"
OUTPUT_ROOT = ROOT / "outputs"
REFRESH = float(os.environ.get("MONITOR_REFRESH", "2"))
BAR_WIDTH = int(os.environ.get("MONITOR_BAR_WIDTH", "28"))

METHODS = ("dct_adapter", "token_mlp", "lora")
PLAN_METHODS = ("base", *METHODS)
DOMAINS = ("clean", "ood")


@dataclass
class Progress:
    percent: float
    state: str
    detail: str


@dataclass
class PlanningRecords:
    has_structured: bool = False
    phase: str = ""
    status: str = ""
    total: int = 0
    completed: int = 0
    current_episode: int = 0
    success_count: int = 0
    episode_active: bool = False
    current_step: int | None = None
    current_step_total: int | None = None
    current_step_fraction: float = 0.0
    current_step_percent: float | None = None
    last_cem_step: int | None = None
    last_cem_seconds: float | None = None
    episode_durations: list[float] = field(default_factory=list)


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


def read_config_int(section: str, key: str, default: int) -> int:
    path = ROOT / "configs" / "experiment" / "robocasa_pilot.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return default
    section_match = re.search(
        r"(?ms)^" + re.escape(section) + r":\s*$"
        r"(.*?)(?=^[A-Za-z_][A-Za-z0-9_]*:\s*$|\Z)",
        text,
    )
    if section_match is None:
        return default
    value_match = re.search(
        r"(?m)^\s+" + re.escape(key) + r":\s*(\d+)\s*$",
        section_match.group(1),
    )
    return int(value_match.group(1)) if value_match else default


TRAIN_EPOCHS = read_config_int("training", "epochs", 20)
PLAN_EPISODES = read_config_int("evaluation", "num_episodes", 50)


def clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def bar(percent: float, width: int | None = None, active: bool = False) -> str:
    width = width or BAR_WIDTH
    value = clamp(percent)
    filled = int(round(width * value / 100.0))
    if active and filled == 0:
        position = int(time.time() * 2) % width
        cells = [" "] * width
        cells[position] = ">"
        return "".join(cells)
    return "#" * filled + "-" * (width - filled)


def file_age(path: Path) -> int | None:
    try:
        return max(0, int(time.time() - path.stat().st_mtime))
    except OSError:
        return None


def human_duration(seconds: float | int) -> str:
    value = max(0, int(seconds))
    if value < 60:
        return f"{value}s"
    if value < 3600:
        return f"{value // 60}m{value % 60:02d}s"
    return f"{value // 3600}h{(value % 3600) // 60:02d}m"


def log_heartbeat(path: Path) -> tuple[str, bool]:
    age = file_age(path)
    if age is None:
        return "no planning log", False
    if age > 120:
        return f"log unchanged {human_duration(age)}", True
    return f"log updated {human_duration(age)} ago", False


def tail_has_failure(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    fatal_tokens = (
        "RuntimeError:",
        "InterpolationKeyError:",
        "AttributeError:",
        "ModuleNotFoundError:",
        "CUDA out of memory",
        "Fatal Python error:",
    )
    if any(token in lines[-1] for token in fatal_tokens):
        return True
    recent = lines[-20:]
    return any("Traceback (most recent call last)" in line for line in recent) and any(
        token in line for line in recent for token in fatal_tokens
    )


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
            return line[:120]
    return "waiting for log output"


def bad_object_count(text: str) -> int:
    return text.count("Skipping incompatible RoboCasa object")


def warning_detail(text: str) -> str | None:
    count = bad_object_count(text)
    return f"bad-object skips in log tail={count}" if count else None


def result_path(method: str, domain: str) -> Path | None:
    matches = sorted(OUTPUT_ROOT.glob(f"**/{method}/{domain}/results.json"))
    return matches[-1] if matches else None


def completed_result(method: str, domain: str) -> Progress | None:
    path = result_path(method, domain)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        total = int(payload["total_episodes"])
        success_count = int(payload["success_count"])
        detail = f"completed {total}/{total} | success {success_count}/{total}"
    except (OSError, ValueError, KeyError, TypeError):
        detail = "results.json complete"
    return Progress(100.0, "DONE", detail)


def parse_tqdm(text: str, expected_total: int | None = None) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    patterns = (
        r"(?P<pct>\d{1,3})%\|[^\n]*?\|\s*(?P<done>\d+)\s*/\s*(?P<total>\d+)",
        r"(?P<done>\d+)\s*/\s*(?P<total>\d+)[^\n]*?(?P<pct>\d{1,3})%",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            done = int(match.group("done"))
            total = int(match.group("total"))
            percent = int(match.group("pct"))
            if total <= 0 or done > total:
                continue
            if expected_total is not None and total != expected_total:
                continue
            candidates.append((match.start(), percent, done, total))
    if not candidates:
        return None
    _, percent, done, total = max(candidates, key=lambda item: item[0])
    return percent, done, total


def process_elapsed(pid: int) -> int:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "etimes=", "-p", str(pid)],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(output) if output else 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def list_active_processes() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    active: dict[str, dict[str, object]] = {}
    by_gpu: dict[str, dict[str, object]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return active, by_gpu
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        relevant = ("build_feature_cache.py", "train_adapter.py", "scripts/plan.py")
        if not any(token in command for token in relevant):
            continue
        gpu = "?"
        try:
            environment = (entry / "environ").read_bytes()
            for item in environment.split(b"\0"):
                if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    gpu = item.split(b"=", 1)[1].decode(errors="replace").split(",")[0]
                    break
        except OSError:
            pass
        method_match = re.search(r"(?:^|\s)method=([A-Za-z0-9_]+)", command)
        domain_match = re.search(r"(?:^|\s)domain=([A-Za-z0-9_]+)", command)
        method = method_match.group(1) if method_match else "unknown"
        domain = domain_match.group(1) if domain_match else "unknown"
        if "build_feature_cache.py" in command:
            key, label = "cache", "feature cache"
        elif "train_adapter.py" in command:
            key, label = f"train:{method}", f"train {method}"
        else:
            key, label = f"plan:{method}:{domain}", f"plan {method}/{domain}"
        info: dict[str, object] = {
            "pid": pid,
            "gpu": gpu,
            "label": label,
            "elapsed": process_elapsed(pid),
        }
        active[key] = info
        if gpu != "?":
            by_gpu[gpu] = info
    return active, by_gpu


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


def assets_status() -> Progress:
    object_root = ROOT / "third_party" / "robocasa" / "robocasa" / "models" / "assets" / "objects"
    if object_root.is_dir():
        return Progress(100.0, "DONE", "RoboCasa object assets present")
    return Progress(0.0, "WAIT", "RoboCasa object assets missing")


def cache_status(active: dict[str, dict[str, object]]) -> Progress:
    if CACHE_PATH.is_file() and CACHE_PATH.stat().st_size > 0 and "cache" not in active:
        return Progress(100.0, "DONE", "shared cache exists")
    log = LOG_ROOT / "build_feature_cache.log"
    text = tail_text(log)
    match = parse_tqdm(text)
    if match:
        percent, done, total = match
        return Progress(float(percent), "RUN", f"{done}/{total}")
    if "cache" in active:
        return Progress(0.0, "RUN", latest_meaningful_line(text))
    if tail_has_failure(text):
        return Progress(0.0, "FAIL", latest_meaningful_line(text))
    return Progress(0.0, "WAIT", "not started")


def training_status(method: str, active: dict[str, dict[str, object]]) -> Progress:
    checkpoint = CKPT_ROOT / f"{method}_final.pt"
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return Progress(100.0, "DONE", "checkpoint complete")
    log = LOG_ROOT / f"train_{method}.log"
    text = tail_text(log)
    active_key = f"train:{method}"
    matches = list(re.finditer(r"(?i)epoch\s+(\d+)\s*/\s*(\d+)[^\n]*?(\d{1,3})%", text))
    if matches:
        epoch, total, inner = map(int, matches[-1].groups())
        overall = 100.0 * ((epoch - 1) + inner / 100.0) / max(total, 1)
        return Progress(overall, "RUN", f"epoch {epoch}/{total}, batch {inner}%")
    completed = [int(match.group(1)) for match in re.finditer(r"(?i)\bepoch=(\d+)\b", text)]
    if completed:
        epoch = max(completed)
        state = "RUN" if active_key in active else "WAIT"
        return Progress(100.0 * epoch / max(TRAIN_EPOCHS, 1), state, f"completed epoch {epoch}/{TRAIN_EPOCHS}")
    if active_key in active:
        return Progress(0.0, "RUN", latest_meaningful_line(text))
    if tail_has_failure(text):
        return Progress(0.0, "FAIL", latest_meaningful_line(text))
    return Progress(0.0, "WAIT", "not started")


def parse_progress_fields(line: str) -> dict[str, str]:
    marker = line.find("PLANNING_PROGRESS")
    if marker < 0:
        return {}
    payload = line[marker + len("PLANNING_PROGRESS") :]
    return dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", payload))


def parse_planning_records(text: str, method: str, domain: str) -> PlanningRecords:
    records = PlanningRecords()
    for line in text.splitlines():
        fields = parse_progress_fields(line)
        if fields:
            if fields.get("method", method) != method or fields.get("domain", domain) != domain:
                continue
            records.has_structured = True
            records.phase = fields.get("phase", records.phase)
            records.status = fields.get("status", records.status)
            phase = fields.get("phase")
            status = fields.get("status")
            if "total_episodes" in fields:
                records.total = int(fields["total_episodes"])
            if "total" in fields:
                records.total = int(fields["total"])
            if "completed" in fields:
                records.completed = int(fields["completed"])
            if "success_count" in fields:
                records.success_count = int(fields["success_count"])
            if phase == "episode" and status == "started":
                records.current_episode = int(fields["episode"])
                records.episode_active = True
                records.current_step = None
                records.current_step_total = None
                records.current_step_fraction = 0.0
                records.current_step_percent = None
                records.last_cem_step = None
                records.last_cem_seconds = None
            elif phase == "episode" and status == "completed":
                records.current_episode = int(fields["episode"])
                records.episode_active = False
                records.current_step = None
                records.current_step_total = None
                records.current_step_fraction = 0.0
                records.current_step_percent = None
                if "elapsed_seconds" in fields:
                    records.episode_durations.append(float(fields["elapsed_seconds"]))
            elif phase == "job" and status == "completed":
                records.episode_active = False
            continue

        if records.episode_active and "executing agent" in line:
            counter = re.search(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", line)
            if counter:
                step = int(counter.group(1))
                total = int(counter.group(2))
                if total > 0 and 0 <= step <= total:
                    records.current_step = step
                    records.current_step_total = total
                    records.current_step_fraction = step / total
                    records.current_step_percent = 100.0 * records.current_step_fraction
            else:
                percentage = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)%", line)
                if percentage:
                    value = clamp(float(percentage.group(1)))
                    records.current_step_fraction = value / 100.0
                    records.current_step_percent = value

        if records.episode_active:
            action_optim = re.search(
                r"Action optim at step\s+(\d+)\s+took\s+([0-9]+(?:\.[0-9]+)?)\s+seconds",
                line,
            )
            if action_optim:
                records.last_cem_step = int(action_optim.group(1))
                records.last_cem_seconds = float(action_optim.group(2))
    return records


def legacy_planning_stage(text: str) -> str:
    stages = (
        (("Loading checkpoint from local path",), "loading world model"),
        (("loaded pretrained predictor",), "world model loaded"),
        (("Loaded 14 trajectories", "Total dataset:"), "dataset loaded"),
        (("Initializing GC_Agent",), "agent initialization"),
        (("Loading controller configuration",), "controller initialization"),
        (("Wrapped RoboCasa environment",), "environment initialization"),
        (("Resetting from provided model XML",), "environment reset"),
        (("Finished resetting from provided model XML",), "environment ready"),
    )
    stage = "planning job initialization"
    for tokens, label in stages:
        if any(token in text for token in tokens):
            stage = label
    return stage


def eta_detail(records: PlanningRecords) -> str | None:
    if records.completed < 1 or not records.episode_durations:
        return None
    average = sum(records.episode_durations) / len(records.episode_durations)
    fraction = records.current_step_fraction if records.episode_active else 0.0
    remaining = max(0.0, records.total - records.completed - fraction)
    return f"ETA≈{human_duration(average * remaining)}"


def planning_status(
    method: str,
    domain: str,
    active: dict[str, dict[str, object]],
) -> Progress:
    active_key = f"plan:{method}:{domain}"
    is_active = active_key in active
    if not is_active:
        completed = completed_result(method, domain)
        if completed is not None:
            return completed
    log = LOG_ROOT / f"plan_{method}_{domain}.log"
    text = tail_text(log)
    heartbeat, quiet = log_heartbeat(log)
    warning = warning_detail(text)
    records = parse_planning_records(text, method, domain)

    if not records.has_structured:
        if not is_active and tail_has_failure(text):
            return Progress(0.0, "FAIL", f"{latest_meaningful_line(text)} | {heartbeat}")
        state = "RUN" if is_active else "WAIT"
        stage = legacy_planning_stage(text) if text else "not started"
        parts = [stage, "legacy log: exact episode progress unavailable", heartbeat]
        if is_active and quiet:
            parts.append("RUN, log quiet")
        if warning:
            parts.append(warning)
        return Progress(0.0, state, " | ".join(parts))

    if not is_active and tail_has_failure(text):
        return Progress(0.0, "FAIL", f"{latest_meaningful_line(text)} | {heartbeat}")

    total = records.total or PLAN_EPISODES
    if records.phase == "job" and records.status == "completed":
        detail = f"completed {records.completed}/{total} | success {records.success_count}/{total}"
        if warning:
            detail += f" | {warning}"
        return Progress(100.0, "DONE", detail)

    fraction = records.current_step_fraction if records.episode_active else 0.0
    percent = clamp(100.0 * (records.completed + fraction) / max(total, 1))
    state = "RUN" if is_active else "WAIT"
    if records.phase == "environment" and records.status == "started":
        parts = ["environment initialization"]
    elif records.episode_active:
        parts = [f"episode {records.current_episode}/{total}"]
        if records.current_step is not None and records.current_step_total is not None:
            parts.append(f"step {records.current_step}/{records.current_step_total}")
        elif records.current_step_percent is not None:
            parts.append(f"step {records.current_step_percent:.0f}%")
        parts.append(f"completed {records.completed}/{total}")
        if records.completed:
            parts.append(f"success {records.success_count}/{records.completed}")
        if records.last_cem_seconds is not None:
            parts.append(f"last CEM {records.last_cem_seconds:.2f}s")
    elif records.completed:
        parts = [
            f"completed {records.completed}/{total}",
            f"success {records.success_count}/{records.completed}",
        ]
    else:
        parts = ["planning job initialization"]

    if records.episode_durations:
        average = sum(records.episode_durations) / len(records.episode_durations)
        parts.append(f"last episode {records.episode_durations[-1]:.1f}s")
        parts.append(f"average {average:.1f}s")
    eta = eta_detail(records)
    if eta:
        parts.append(eta)
    parts.append(heartbeat)
    if is_active and quiet:
        parts.append("RUN, log quiet")
    if warning:
        parts.append(warning)
    return Progress(percent, state, " | ".join(parts))


def state_marker(state: str) -> str:
    return {
        "DONE": "[OK]  ",
        "RUN": "[RUN] ",
        "WAIT": "[--]  ",
        "FAIL": "[FAIL]",
    }.get(state, f"[{state}]")


def print_progress(label: str, progress: Progress, indent: str = "") -> None:
    active = progress.state == "RUN" and progress.percent == 0.0
    print(
        f"{indent}{label:<24} {state_marker(progress.state)} "
        f"[{bar(progress.percent, active=active)}] {clamp(progress.percent):6.2f}%  "
        f"{progress.detail}"
    )


def draw() -> None:
    active, by_gpu = list_active_processes()
    stats = gpu_stats()
    assets = assets_status()
    cache = cache_status(active)
    training = {method: training_status(method, active) for method in METHODS}
    planning = {
        (method, domain): planning_status(method, domain, active)
        for method in PLAN_METHODS
        for domain in DOMAINS
    }
    train_average = sum(item.percent for item in training.values()) / len(training)
    plan_average = sum(item.percent for item in planning.values()) / len(planning)
    overall = 0.05 * assets.percent + 0.10 * cache.percent + 0.25 * train_average + 0.60 * plan_average

    sys.stdout.write("\033[2J\033[H")
    print(f"WM Adapter Full-Pipeline Dashboard  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {ROOT}")
    print(f"Overall [{bar(overall, 42)}] {overall:6.2f}%")
    print("=" * 132)

    print("\nPIPELINE")
    print_progress("1. Assets", assets)
    print_progress("2. Feature cache", cache)
    training_state = "DONE" if train_average >= 100.0 else (
        "RUN" if any(item.state == "RUN" for item in training.values()) else "WAIT"
    )
    print_progress(
        "3. Adapter training",
        Progress(
            train_average,
            training_state,
            f"{sum(item.percent >= 100.0 for item in training.values())}/{len(training)} methods complete",
        ),
    )
    planning_state = "DONE" if plan_average >= 100.0 else (
        "RUN" if any(item.state == "RUN" for item in planning.values()) else "WAIT"
    )
    print_progress(
        "4. Planning",
        Progress(
            plan_average,
            planning_state,
            f"{sum(item.percent >= 100.0 for item in planning.values())}/8 jobs complete",
        ),
    )

    print("\nGPU TASKS")
    if not stats:
        print("  nvidia-smi unavailable")
    for gpu in sorted(stats, key=int):
        stat = stats[gpu]
        task = by_gpu.get(gpu)
        if task:
            label = (
                f"{task['label']} pid={task['pid']} "
                f"time={human_duration(int(task['elapsed']))}"
            )
        else:
            label = "idle"
        print(
            f"  GPU {gpu} | util {stat['util']:3d}% | mem {stat['used']:5d}/{stat['total']:5d} MiB | "
            f"temp {stat['temp']:2d}C | {label}"
        )

    print("\nTRAINING")
    for method in METHODS:
        print_progress(method, training[method], indent="  ")

    print("\nPLANNING")
    for method in PLAN_METHODS:
        for domain in DOMAINS:
            print_progress(f"{method}/{domain}", planning[(method, domain)], indent="  ")

    print("\nPlanning percentages use structured completed episodes and explicit 'executing agent' steps only.")
    print(f"Ctrl+C to exit | refresh {REFRESH:.1f}s")
    sys.stdout.flush()


def main() -> None:
    try:
        while True:
            draw()
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
