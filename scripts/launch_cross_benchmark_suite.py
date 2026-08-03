#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from wm_adapter.experiments.cross_benchmark import load_suite_config
from wm_adapter.utils.reproducibility import project_root, resolve_path


DEFAULT_CONFIG = "configs/experiment/cross_benchmark_v1.yaml"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch or attach to cross_benchmark_v1")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stop", action="store_true")
    action.add_argument("--attach", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _paths(suite: Any, self_test: bool) -> tuple[Path, Path, Path]:
    if self_test:
        roots = suite.self_test.roots
        return (
            resolve_path(str(roots.pid_path)),
            resolve_path(str(roots.state_path)),
            resolve_path(str(roots.log_root)).parent / "runner.log",
        )
    return (
        resolve_path(str(suite.pid_path)),
        resolve_path(str(suite.state_path)),
        resolve_path(str(suite.runner_log)),
    )


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except PermissionError:
        stat = ""
    if stat:
        suffix = stat[stat.rfind(")") + 2 :].split()
        if suffix and suffix[0] == "Z":
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _monitor(state: Path, config: str, *, once: bool) -> int:
    command = [
        sys.executable, "scripts/monitor_cross_benchmark_suite.py",
        "--state", str(state), "--suite-config", config,
    ]
    if once:
        command.append("--once")
    return subprocess.call(command, cwd=project_root())


def main() -> None:
    args = _args()
    suite = load_suite_config(args.config)
    pid_path, state_path, runner_log = _paths(suite, args.self_test)
    pid = _read_pid(pid_path)
    running = _alive(pid)
    if pid is not None and not running:
        pid_path.unlink(missing_ok=True)
        pid = None
    if args.dry_run:
        raise SystemExit(
            subprocess.call(
                [sys.executable, "scripts/run_cross_benchmark_suite.py", "--config", args.config, "--dry-run"],
                cwd=project_root(),
            )
        )
    if args.status:
        raise SystemExit(_monitor(state_path, args.config, once=True))
    if args.stop:
        if not running or pid is None:
            print("cross_benchmark_v1 runner is not active")
            return
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        for _ in range(50):
            if not _alive(pid):
                break
            time.sleep(0.1)
        if _alive(pid):
            raise RuntimeError(f"Runner process group did not stop after SIGTERM: pid={pid}")
        pid_path.unlink(missing_ok=True)
        print(f"Stopped cross_benchmark_v1 runner pid={pid}")
        return
    if args.attach:
        if not running:
            print("cross_benchmark_v1 runner is not active; showing the latest state")
        raise SystemExit(_monitor(state_path, args.config, once=False))
    if not running:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        runner_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_fd = os.open(pid_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            competing_pid = _read_pid(pid_path)
            if _alive(competing_pid):
                print(
                    f"cross_benchmark_v1 was started concurrently with pid={competing_pid}; "
                    "attaching Dashboard"
                )
                raise SystemExit(_monitor(state_path, args.config, once=False))
            raise RuntimeError(f"Stale or incomplete runner lock appeared concurrently: {pid_path}")
        command = [
            sys.executable, "scripts/run_cross_benchmark_suite.py", "--config", args.config,
        ]
        if args.self_test:
            command.append("--self-test")
        try:
            with runner_log.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, cwd=project_root(), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, text=True,
                )
            os.write(lock_fd, f"{process.pid}\n".encode("ascii"))
        except BaseException:
            os.close(lock_fd)
            pid_path.unlink(missing_ok=True)
            raise
        os.close(lock_fd)
        print(f"Started cross_benchmark_v1 runner pid={process.pid}; log={runner_log}")
    else:
        print(f"cross_benchmark_v1 is already running with pid={pid}; attaching Dashboard")
    raise SystemExit(_monitor(state_path, args.config, once=False))


if __name__ == "__main__":
    main()
