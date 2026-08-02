#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import time
from pathlib import Path
from typing import Any

import monitor_all_paper_experiments as core
from wm_adapter.utils.reproducibility import resolve_path


def _progress_line(label: str, progress: core.Progress, width: int) -> str:
    active = progress.state == "RUN" and progress.percent == 0.0
    bar_width = max(8, min(core.BAR_WIDTH, width - 61))
    return (
        f"{label:<29} {core.state_marker(progress.state)} "
        f"[{core.bar(progress.percent, bar_width, active)}] "
        f"{core.clamp(progress.percent):6.2f}%  {progress.detail}"
    )


def _dashboard_lines(
    state_path: Path,
    suite_path: Path,
    width: int,
    height: int,
    refresh: float,
) -> list[str]:
    state = core.load_state(state_path)
    jobs: dict[str, dict[str, Any]] = state.get("jobs", {})
    specs = core.load_phase_specs(suite_path)
    overall = core.overall_progress(specs, jobs)
    running_jobs = [
        (job_id, entry)
        for job_id, entry in jobs.items()
        if str(entry.get("status")) == "running"
    ]
    running_by_gpu = {
        str(entry.get("gpu")): (job_id, entry)
        for job_id, entry in running_jobs
        if entry.get("gpu") is not None
    }
    completed = sum(
        str(entry.get("status")) in {"completed", "reused"}
        for entry in jobs.values()
    )
    failed = sum(
        str(entry.get("status")) == "failed" for entry in jobs.values()
    )
    blocked = sum(
        str(entry.get("status")) == "blocked" for entry in jobs.values()
    )
    expected = int(
        state.get("expected_jobs", sum(spec.expected for spec in specs))
    )
    pending = max(
        0,
        expected - completed - len(running_jobs) - failed - blocked,
    )
    overall_width = max(12, min(42, width - 22))

    lines = [
        f"WM Adapter ICRA Paper-Suite Dashboard  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Project: {core.ROOT}",
        f"Overall [{core.bar(overall, overall_width)}] {overall:6.2f}%",
        (
            f"Jobs: {completed}/{expected} complete | "
            f"{len(running_jobs)} running | {pending} pending | "
            f"{failed} failed | {blocked} blocked"
        ),
        "=" * max(1, width - 1),
        "",
        "PIPELINE",
    ]
    lines.extend(
        _progress_line(spec.label, core.phase_progress(spec, jobs), width)
        for spec in specs
    )

    lines.extend(["", "GPU TASKS"])
    stats = core.gpu_stats()
    if not stats:
        lines.append("  nvidia-smi unavailable")
    else:
        for gpu in sorted(stats, key=int):
            stat = stats[gpu]
            running = running_by_gpu.get(gpu)
            if running is None:
                task = "idle"
            else:
                job_id, entry = running
                task = (
                    f"{job_id} pid={entry.get('pid', '?')} "
                    f"time={core.human_duration(core.process_elapsed(entry))}"
                )
            lines.append(
                f"  GPU {gpu} | util {stat['util']:3d}% | "
                f"mem {stat['used']:5d}/{stat['total']:5d} MiB | "
                f"temp {stat['temp']:2d}C | {task}"
            )

    lines.extend(["", "RUNNING JOBS"])
    if not running_jobs:
        lines.append("  no active paper-suite jobs")
    else:
        for job_id, entry in sorted(
            running_jobs,
            key=lambda item: int(item[1].get("gpu", 999)),
        ):
            label = f"GPU {entry.get('gpu', '?')} {job_id}"
            lines.append(
                "  "
                + _progress_line(
                    label,
                    core.job_progress(job_id, entry),
                    width - 2,
                )
            )

    failed_jobs = [
        (job_id, entry)
        for job_id, entry in jobs.items()
        if str(entry.get("status")) == "failed"
    ]
    if failed_jobs:
        lines.extend(["", "FAILURES"])
        lines.extend(
            f"  [FAIL] {job_id}: {entry.get('error', 'unknown failure')}"
            for job_id, entry in failed_jobs
        )

    remaining = height - len(lines) - 3
    recent = sorted(
        [
            (job_id, entry)
            for job_id, entry in jobs.items()
            if str(entry.get("status")) in {"completed", "reused"}
        ],
        key=lambda item: float(
            item[1].get(
                "ended_at_unix",
                item[1].get("started_at_unix", 0.0),
            )
        ),
        reverse=True,
    )
    if recent and remaining >= 3:
        lines.extend(["", "RECENTLY COMPLETED"])
        for job_id, entry in recent[: min(core.RECENT_COMPLETED, remaining - 2)]:
            status = (
                "REUSED"
                if str(entry.get("status")) == "reused"
                else "DONE"
            )
            elapsed = float(entry.get("elapsed_seconds", 0.0))
            suffix = (
                f" | time {core.human_duration(elapsed)}"
                if elapsed > 0
                else ""
            )
            lines.append(
                f"  [{status}] {job_id}{suffix} | "
                f"{core.artifact_detail(entry)}"
            )

    age = core.file_age(state_path)
    age_text = (
        "state not created"
        if age is None
        else f"state updated {core.human_duration(age)} ago"
    )
    lines.extend(
        [
            "",
            (
                f"{age_text} | q/Ctrl+C exit | "
                f"refresh {refresh:.1f}s | single-screen live refresh"
            ),
        ]
    )
    return lines


def _safe_add(
    screen: Any,
    row: int,
    text: str,
    width: int,
    attr: int = 0,
) -> None:
    if row < 0 or width <= 1:
        return
    try:
        screen.addnstr(row, 0, text, width - 1, attr)
    except curses.error:
        pass


def _run_curses(
    screen: Any,
    state_path: Path,
    suite_path: Path,
    refresh: float,
) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.leaveok(True)
    screen.timeout(max(100, int(refresh * 1000)))

    while True:
        height, width = screen.getmaxyx()
        lines = _dashboard_lines(
            state_path,
            suite_path,
            width,
            height,
            refresh,
        )
        screen.erase()
        for row, line in enumerate(lines[:height]):
            heading = line in {
                "PIPELINE",
                "GPU TASKS",
                "RUNNING JOBS",
                "FAILURES",
                "RECENTLY COMPLETED",
            }
            attr = curses.A_BOLD if row == 0 or heading else 0
            _safe_add(screen, row, line, width, attr)
        if len(lines) > height:
            _safe_add(
                screen,
                height - 1,
                (
                    f"Terminal too short: showing {height}/{len(lines)} rows; "
                    "enlarge terminal | q exit"
                ),
                width,
                curses.A_REVERSE,
            )
        screen.noutrefresh()
        curses.doupdate()
        key = screen.getch()
        if key in {ord("q"), ord("Q"), 3}:
            return


def _print_once(
    state_path: Path,
    suite_path: Path,
    refresh: float,
) -> None:
    for line in _dashboard_lines(
        state_path,
        suite_path,
        width=160,
        height=200,
        refresh=refresh,
    ):
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default=str(core.DEFAULT_STATE))
    parser.add_argument("--suite-config", default=str(core.DEFAULT_SUITE))
    parser.add_argument("--refresh", type=float, default=core.REFRESH)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.refresh <= 0:
        raise ValueError(f"refresh must be positive, received {args.refresh}")
    state_path = resolve_path(args.state)
    suite_path = resolve_path(args.suite_config)
    if args.once:
        _print_once(state_path, suite_path, args.refresh)
        return
    curses.wrapper(
        _run_curses,
        state_path,
        suite_path,
        args.refresh,
    )


if __name__ == "__main__":
    main()
