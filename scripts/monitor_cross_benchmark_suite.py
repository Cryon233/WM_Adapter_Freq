#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import json
import os
import signal
import time
import textwrap
from pathlib import Path
from typing import Any

import monitor_all_paper_experiments as progress_core

from wm_adapter.experiments.cross_benchmark import PHASES, load_suite_config
from wm_adapter.experiments.cross_jobs import build_job_graph
from wm_adapter.utils.reproducibility import project_root, resolve_path


DEFAULT_STATE = project_root() / "logs/cross_benchmark_v1/state.json"
DEFAULT_SUITE = project_root() / "configs/experiment/cross_benchmark_v1.yaml"
RUNNER_COMMAND = "scripts/run_cross_benchmark_suite.py"


def _state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("state root is not a mapping")
        return value
    except FileNotFoundError:
        return {"suite": "cross_benchmark_v1", "status": "waiting", "jobs": {}}
    except Exception as error:
        return {
            "suite": "cross_benchmark_v1", "status": "failed", "jobs": {},
            "error": f"Cannot read state: {type(error).__name__}: {error}",
        }


def _normalized(entry: dict[str, Any]) -> dict[str, Any]:
    value = dict(entry)
    value["artifact"] = value.get("artifact_validation")
    value["started_at_unix"] = value.get("start_time")
    value["ended_at_unix"] = value.get("end_time")
    return value


def _job_progress(job_id: str, entry: dict[str, Any]) -> progress_core.Progress:
    return progress_core.job_progress(job_id, _normalized(entry))


def _running_job_sort_key(
    value: tuple[str, dict[str, Any]],
) -> tuple[int, int, str]:
    job_id, entry = value
    gpu = entry.get("gpu")
    if gpu is None:
        return (1, 0, job_id)
    try:
        return (0, int(gpu), job_id)
    except (TypeError, ValueError):
        return (1, 0, job_id)


def _phase_progress(
    phase: str, expected: int, jobs: dict[str, dict[str, Any]]
) -> progress_core.Progress:
    if expected == 0:
        return progress_core.Progress(100.0, "DONE", "not part of this run")
    entries = [(key, value) for key, value in jobs.items() if value.get("phase") == phase]
    completed = sum(value.get("status") in {"completed", "reused"} for _, value in entries)
    running = [_job_progress(key, value) for key, value in entries if value.get("status") == "running"]
    failed = sum(value.get("status") == "failed" for _, value in entries)
    blocked = sum(value.get("status") == "blocked" for _, value in entries)
    partial = sum(value.percent / 100.0 for value in running)
    percent = 100.0 * (completed + partial) / max(expected, 1)
    marker = "FAIL" if failed else "DONE" if completed >= expected else "RUN" if running else "WAIT"
    detail = (
        f"{completed}/{expected} complete | running {len(running)} | "
        f"pending {max(0, expected - completed - len(running) - failed - blocked)}"
    )
    if failed:
        detail += f" | failed {failed}"
    if blocked:
        detail += f" | blocked {blocked}"
    return progress_core.Progress(percent, marker, detail)


def _progress_line(label: str, progress: progress_core.Progress, width: int) -> str:
    active = progress.state == "RUN" and progress.percent == 0.0
    bar_width = max(8, min(28, width - 67))
    return (
        f"{label:<35} {progress_core.state_marker(progress.state)} "
        f"[{progress_core.bar(progress.percent, bar_width, active)}] "
        f"{progress_core.clamp(progress.percent):6.2f}%  {progress.detail}"
    )


def _wrap_dashboard_lines(lines: list[str], width: int) -> list[str]:
    usable = max(8, width - 1)
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        if set(line) == {"="}:
            wrapped.append("=" * usable)
            continue
        leading = len(line) - len(line.lstrip(" "))
        continuation = " " * min(leading + 2, max(0, usable - 1))
        pieces = textwrap.wrap(
            line,
            width=usable,
            subsequent_indent=continuation,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        wrapped.extend(pieces or [""])
    return wrapped


def _expected(suite_path: Path, state: dict[str, Any]) -> dict[str, int]:
    if state.get("phase_summary"):
        return {
            phase: int(state["phase_summary"].get(phase, {}).get("total", 0))
            for phase in PHASES
        }
    suite = load_suite_config(suite_path)
    jobs = build_job_graph(suite, self_test=bool(state.get("self_test", False)))
    return {phase: sum(job.phase == phase for job in jobs) for phase in PHASES}


def dashboard_lines(
    state_path: Path, suite_path: Path, width: int, height: int, refresh: float
) -> list[str]:
    state = _state(state_path)
    jobs: dict[str, dict[str, Any]] = dict(state.get("jobs", {}))
    expected = _expected(suite_path, state)
    phases = [_phase_progress(phase, expected[phase], jobs) for phase in PHASES]
    total = sum(expected.values())
    overall = sum(item.percent / 100.0 * expected[phase] for phase, item in zip(PHASES, phases))
    overall = 100.0 * overall / max(total, 1)
    running = [(key, value) for key, value in jobs.items() if value.get("status") == "running"]
    completed = sum(value.get("status") in {"completed", "reused"} for value in jobs.values())
    failed = sum(value.get("status") == "failed" for value in jobs.values())
    blocked = sum(value.get("status") == "blocked" for value in jobs.values())
    pending = max(0, total - completed - len(running) - failed - blocked)
    lines = [
        f"WM Adapter RoboCasa + LIBERO Dashboard  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Project: {project_root()}",
        f"State:   {state_path}",
        f"Overall [{progress_core.bar(overall, max(12, min(42, width - 22)))}] {overall:6.2f}%",
        f"Jobs: {completed}/{total} complete | {len(running)} running | {pending} pending | {failed} failed | {blocked} blocked",
        "=" * max(1, width - 1), "", "PIPELINE",
    ]
    lines.extend(
        _progress_line(f"{index}. {phase}", progress, width)
        for index, (phase, progress) in enumerate(zip(PHASES, phases), start=1)
    )
    lines.extend(["", "GPU TASKS"])
    gpu = progress_core.gpu_stats()
    running_gpu = {
        str(value.get("gpu")): (key, value)
        for key, value in running if value.get("gpu") is not None
    }
    if not gpu:
        lines.append("  nvidia-smi unavailable")
    for index in sorted(gpu, key=int):
        stat = gpu[index]
        active = running_gpu.get(index)
        detail = "idle"
        if active is not None:
            job_id, entry = active
            started = float(entry.get("start_time") or time.time())
            detail = (
                f"{job_id} pid={entry.get('pid', '?')} "
                f"time={progress_core.human_duration(time.time() - started)}"
            )
        lines.append(
            f"  GPU {index} | util {stat['util']:3d}% | mem {stat['used']:5d}/{stat['total']:5d} MiB | "
            f"temp {stat['temp']:2d}C | {detail}"
        )
    lines.extend(["", "RUNNING JOBS"])
    if not running:
        lines.append("  no active cross-benchmark jobs")
    for job_id, entry in sorted(running, key=_running_job_sort_key):
        gpu_label = entry.get("gpu")
        gpu_text = "--" if gpu_label is None else str(gpu_label)
        lines.append(
            "  " + _progress_line(
                f"GPU {gpu_text} {job_id}", _job_progress(job_id, entry), width - 2
            )
        )
    failures = [(key, value) for key, value in jobs.items() if value.get("status") == "failed"]
    if failures:
        lines.extend(["", "FAILURES"])
        lines.extend(f"  [FAIL] {key}: {value.get('error', 'unknown failure')}" for key, value in failures)
    available = height - len(lines) - 3
    recent = sorted(
        [(key, value) for key, value in jobs.items() if value.get("status") in {"completed", "reused"}],
        key=lambda value: float(value[1].get("end_time") or value[1].get("start_time") or 0.0),
        reverse=True,
    )
    if recent and available >= 3:
        lines.extend(["", "RECENTLY COMPLETED"])
        for key, value in recent[: min(4, available - 2)]:
            tag = "REUSED" if value.get("status") == "reused" else "DONE"
            detail = progress_core.artifact_detail(_normalized(value))
            lines.append(f"  [{tag}] {key} | {detail}")
    age = progress_core.file_age(state_path)
    age_text = "state not created" if age is None else f"state updated {progress_core.human_duration(age)} ago"
    lines.extend([
        "",
        f"{age_text} | refresh {refresh:.1f}s | "
        "q/Ctrl+C detach | X terminate all suite processes and exit",
    ])
    return _wrap_dashboard_lines(lines, width)


def _read_runner_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None
    return value if value > 1 else None


def _process_group_alive(process_group: int) -> bool:
    proc = Path("/proc")
    if proc.is_dir():
        for candidate in proc.iterdir():
            if not candidate.name.isdigit():
                continue
            try:
                stat = (candidate / "stat").read_text(encoding="utf-8")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            suffix = stat[stat.rfind(")") + 2 :].split()
            if len(suffix) >= 3:
                state, group = suffix[0], suffix[2]
                if int(group) == process_group and state != "Z":
                    return True
        return False
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _runner_pid_path(suite_path: Path, state: dict[str, Any]) -> Path:
    suite = load_suite_config(suite_path)
    if bool(state.get("self_test", False)):
        return resolve_path(str(suite.self_test.roots.pid_path))
    return resolve_path(str(suite.pid_path))


def _mark_stopped(state_path: Path) -> None:
    state = _state(state_path)
    stopped = time.time()
    state.update(
        status="stopped",
        stopped_at_unix=stopped,
        error="suite terminated explicitly from the Dashboard",
    )
    for entry in state.get("jobs", {}).values():
        if entry.get("status") == "running":
            entry.update(
                status="stopped",
                end_time=stopped,
                error="terminated explicitly from the Dashboard",
            )
    from wm_adapter.benchmarks.base import atomic_json

    atomic_json(state_path, state)


def _terminate_suite(state_path: Path, suite_path: Path) -> str:
    state = _state(state_path)
    pid_path = _runner_pid_path(suite_path, state)
    pid = _read_runner_pid(pid_path)
    if pid is None:
        return f"No active runner PID was found at {pid_path}"
    command_path = Path(f"/proc/{pid}/cmdline")
    try:
        command = command_path.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except FileNotFoundError:
        pid_path.unlink(missing_ok=True)
        return f"Runner PID {pid} is no longer active"
    if RUNNER_COMMAND not in command:
        raise RuntimeError(
            "Refusing to terminate a PID that is not the cross-benchmark runner: "
            f"pid={pid}, command={command!r}, pid_file={pid_path}"
        )
    process_group = os.getpgid(pid)
    os.killpg(process_group, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _process_group_alive(process_group):
        time.sleep(0.1)
    if _process_group_alive(process_group):
        os.killpg(process_group, signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and _process_group_alive(process_group):
            time.sleep(0.05)
    if _process_group_alive(process_group):
        raise RuntimeError(
            f"Runner process group remains active after SIGKILL: pgid={process_group}"
        )
    pid_path.unlink(missing_ok=True)
    _mark_stopped(state_path)
    return f"Terminated runner process group {process_group}"


def _add(screen: Any, row: int, line: str, width: int, attr: int = 0) -> None:
    if row < 0 or width <= 1:
        return
    try:
        screen.addnstr(row, 0, line, width - 1, attr)
    except curses.error:
        return


def _curses(screen: Any, state: Path, suite: Path, refresh: float) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        curses_visibility_unavailable = True
    else:
        curses_visibility_unavailable = False
    del curses_visibility_unavailable
    screen.timeout(max(100, int(refresh * 1000)))
    while True:
        height, width = screen.getmaxyx()
        lines = dashboard_lines(state, suite, width, height, refresh)
        screen.erase()
        for row, line in enumerate(lines[:height]):
            heading = line in {"PIPELINE", "GPU TASKS", "RUNNING JOBS", "FAILURES", "RECENTLY COMPLETED"}
            _add(screen, row, line, width, curses.A_BOLD if row == 0 or heading else 0)
        if len(lines) > height:
            _add(
                screen,
                height - 1,
                "Terminal too small; enlarge to see more | q detach | X terminate suite",
                width,
                curses.A_REVERSE,
            )
        screen.noutrefresh()
        curses.doupdate()
        key = screen.getch()
        if key in {ord("q"), ord("Q"), 3}:
            return
        if key == ord("X"):
            try:
                message = _terminate_suite(state, suite)
            except Exception as error:
                message = f"Termination refused: {type(error).__name__}: {error}"
                _add(screen, height - 1, message, width, curses.A_REVERSE)
                screen.refresh()
                time.sleep(2.0)
                continue
            screen.erase()
            _add(screen, 0, message, width, curses.A_BOLD)
            screen.refresh()
            time.sleep(0.5)
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--suite-config", default=str(DEFAULT_SUITE))
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    state = resolve_path(args.state)
    suite = resolve_path(args.suite_config)
    if args.once:
        print("\n".join(dashboard_lines(state, suite, 160, 200, args.refresh)))
        return
    curses.wrapper(_curses, state, suite, args.refresh)


if __name__ == "__main__":
    main()
