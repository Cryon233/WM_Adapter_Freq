from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wm_adapter.benchmarks.base import atomic_json
from wm_adapter.utils.reproducibility import resolve_path


RUNNER_COMMAND = "scripts/run_cross_benchmark_suite.py"
_TERMINAL_SUITE_STATES = {"completed", "completed_with_failures"}


@dataclass(frozen=True)
class TerminationResult:
    runner_pid: int | None
    process_group: int | None
    signal_used: str | None
    pid_file_removed: bool
    state_updated: bool
    message: str


def _read_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None
    return value if value > 1 else None


def _process_state(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    suffix = stat[stat.rfind(")") + 2 :].split()
    return suffix[0] if suffix else None


def _process_alive(pid: int) -> bool:
    state = _process_state(pid)
    if state is None or state == "Z":
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


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
            if len(suffix) >= 3 and suffix[0] != "Z" and int(suffix[2]) == process_group:
                return True
        return False
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False


def _read_command(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError) as error:
        raise RuntimeError(f"Cannot read runner command line for pid={pid}") from error
    return [item.decode("utf-8", errors="replace") for item in raw.split(b"\0") if item]


def _command_config(arguments: list[str]) -> Path | None:
    for index, value in enumerate(arguments):
        if value == "--config" and index + 1 < len(arguments):
            return resolve_path(arguments[index + 1])
        if value.startswith("--config="):
            return resolve_path(value.split("=", 1)[1])
    return None


def _load_state(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"Suite state root is not a mapping: {path}")
    return value


def _update_stopped_state(path: Path, reason: str) -> bool:
    state = _load_state(path)
    if state is None:
        return False
    if str(state.get("status")) in _TERMINAL_SUITE_STATES:
        return False
    stopped = time.time()
    state.update(
        status="stopped",
        stopped_at_unix=stopped,
        stop_reason=reason,
    )
    for entry in state.get("jobs", {}).values():
        if entry.get("status") == "running":
            entry.update(
                status="stopped",
                end_time=stopped,
                elapsed_seconds=(
                    stopped - float(entry["start_time"])
                    if entry.get("start_time") is not None
                    else None
                ),
                gpu=None,
                pid=None,
                error=reason,
            )
    atomic_json(path, state)
    return True


def _wait_for_group(process_group: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_group_alive(process_group):
            return True
        time.sleep(0.05)
    return not _process_group_alive(process_group)


def terminate_suite(
    *,
    pid_path: str | Path,
    state_path: str | Path,
    suite_config_path: str | Path,
    reason: str,
    self_test: bool | None = None,
    sigterm_timeout_seconds: float = 5.0,
    sigkill_timeout_seconds: float = 2.0,
) -> TerminationResult:
    pid_file = resolve_path(pid_path)
    state_file = resolve_path(state_path)
    target_config = resolve_path(suite_config_path)
    pid = _read_pid(pid_file)
    if pid is None or not _process_alive(pid):
        removed = pid_file.exists()
        pid_file.unlink(missing_ok=True)
        updated = _update_stopped_state(state_file, reason)
        return TerminationResult(
            pid, None, None, removed, updated,
            f"No active runner; stale PID state cleaned for {target_config}",
        )

    try:
        command = _read_command(pid)
    except RuntimeError:
        if not _process_alive(pid):
            pid_file.unlink(missing_ok=True)
            updated = _update_stopped_state(state_file, reason)
            return TerminationResult(
                pid,
                None,
                None,
                True,
                updated,
                f"Runner pid={pid} exited during termination; stale PID cleaned",
            )
        raise
    if not any(RUNNER_COMMAND in value for value in command):
        raise RuntimeError(
            "Refusing to terminate a PID that is not the cross-benchmark runner: "
            f"pid={pid}, command={command!r}, pid_file={pid_file}"
        )
    command_is_self_test = "--self-test" in command
    if self_test is not None and command_is_self_test != self_test:
        raise RuntimeError(
            "Refusing to terminate the other suite lifecycle: "
            f"pid={pid}, requested_self_test={self_test}, "
            f"command_self_test={command_is_self_test}, pid_file={pid_file}"
        )
    state = _load_state(state_file) or {}
    state_config_value = state.get("suite_config_path")
    state_config = resolve_path(str(state_config_value)) if state_config_value else None
    command_config = _command_config(command)
    if command_config != target_config and state_config != target_config:
        raise RuntimeError(
            "Refusing to terminate a runner for a different suite config: "
            f"pid={pid}, requested={target_config}, command_config={command_config}, "
            f"state_config={state_config}"
        )

    try:
        process_group = os.getpgid(pid)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        updated = _update_stopped_state(state_file, reason)
        return TerminationResult(
            pid,
            None,
            None,
            True,
            updated,
            f"Runner pid={pid} exited during termination; stale PID cleaned",
        )
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        updated = _update_stopped_state(state_file, reason)
        return TerminationResult(
            pid,
            process_group,
            None,
            True,
            updated,
            f"Runner process group {process_group} exited before SIGTERM",
        )
    signal_used = "SIGTERM"
    if not _wait_for_group(process_group, sigterm_timeout_seconds):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        signal_used = "SIGKILL"
        if not _wait_for_group(process_group, sigkill_timeout_seconds):
            raise RuntimeError(
                f"Runner process group remains active after SIGKILL: pgid={process_group}"
            )
    pid_file.unlink(missing_ok=True)
    updated = _update_stopped_state(state_file, reason)
    return TerminationResult(
        pid, process_group, signal_used, True, updated,
        f"Terminated runner process group {process_group} with {signal_used}",
    )
