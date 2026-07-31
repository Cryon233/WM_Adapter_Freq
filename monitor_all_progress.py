#!/usr/bin/env python3
from __future__ import print_function

import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", os.getcwd())).resolve()
LOG_ROOT = ROOT / "logs" / "full_a100_experiment"
CACHE_PATH = ROOT / "storage" / "feature_cache" / "jepa_wm_droid_robocasa_pilot.h5"
CKPT_ROOT = ROOT / "checkpoints" / "jepa_wm_droid" / "robocasa"
OUTPUT_ROOT = ROOT / "outputs"
REFRESH = float(os.environ.get("MONITOR_REFRESH", "2"))
BAR_WIDTH = int(os.environ.get("MONITOR_BAR_WIDTH", "28"))

METHODS = ("dct_adapter", "token_mlp", "lora")
PLAN_METHODS = ("base", "dct_adapter", "token_mlp", "lora")
DOMAINS = ("clean", "ood")


def tail_text(path, max_bytes=2 * 1024 * 1024):
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()
            data = handle.read()
        return data.decode("utf-8", errors="replace").replace("\r", "\n")
    except (OSError, IOError):
        return ""


def read_config_int(section, key, default):
    path = ROOT / "configs" / "experiment" / "robocasa_pilot.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return default

    pattern = (
        r"(?ms)^" + re.escape(section) + r":\s*$"
        r"(.*?)(?=^[A-Za-z_][A-Za-z0-9_]*:\s*$|\Z)"
    )
    match = re.search(pattern, text)
    if not match:
        return default

    value_match = re.search(
        r"(?m)^\s+" + re.escape(key) + r":\s*(\d+)\s*$",
        match.group(1),
    )
    return int(value_match.group(1)) if value_match else default


TRAIN_EPOCHS = read_config_int("training", "epochs", 20)
PLAN_EPISODES = read_config_int("evaluation", "num_episodes", 50)


def clamp(value):
    return max(0.0, min(100.0, float(value)))


def bar(percent, width=None, active=False):
    width = width or BAR_WIDTH
    percent = clamp(percent)
    filled = int(round(width * percent / 100.0))

    if active and filled == 0:
        position = int(time.time() * 2) % width
        chars = [" "] * width
        chars[position] = ">"
        return "".join(chars)

    return "#" * filled + "-" * (width - filled)


def result_exists(method, domain):
    pattern = str(OUTPUT_ROOT / "**" / method / domain / "results.json")
    return any(Path(path).is_file() for path in glob.iglob(pattern, recursive=True))


def file_age(path):
    try:
        return max(0, int(time.time() - path.stat().st_mtime))
    except OSError:
        return None


def human_age(seconds):
    if seconds is None:
        return "no log"
    if seconds < 60:
        return "%ds ago" % seconds
    if seconds < 3600:
        return "%dm%02ds ago" % (seconds // 60, seconds % 60)
    return "%dh%02dm ago" % (
        seconds // 3600,
        (seconds % 3600) // 60,
    )


def parse_tqdm(text, expected_total=None):
    candidates = []
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

            candidates.append(
                (match.start(), percent, done, total)
            )

    return max(
        candidates,
        default=None,
        key=lambda item: item[0],
    )


def latest_episode_number(text, total):
    candidates = []

    paired_patterns = (
        r"(?i)\bepisode(?:_idx| index)?\s*[=:#!]?\s*(\d+)\s*/\s*(\d+)",
        r"(?i)\bevaluat(?:e|ing|ion)[^\n]*?(\d+)\s*/\s*(\d+)",
        r"(?i)\brollout\s*[=:#!]?\s*(\d+)\s*/\s*(\d+)",
    )

    for pattern in paired_patterns:
        for match in re.finditer(pattern, text):
            done = int(match.group(1))
            found_total = int(match.group(2))
            if found_total == total and 0 <= done <= found_total:
                candidates.append((match.start(), done))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    single_patterns = (
        r"(?i)\bepisode(?:_idx| index)?\s*[=:#!]\s*(\d+)\b",
        r"(?i)\bfinished episode\s+(\d+)\b",
        r"(?i)\bevaluating episode\s+(\d+)\b",
    )

    for pattern in single_patterns:
        for match in re.finditer(pattern, text):
            number = int(match.group(1))
            if 0 <= number <= total:
                candidates.append((match.start(), number))

    return max(
        candidates,
        default=(0, None),
        key=lambda item: item[0],
    )[1]


def recent_warning_count(text):
    return text.count("Skipping incompatible RoboCasa object")


def latest_meaningful_line(text):
    ignored = (
        "Skipping incompatible RoboCasa object",
        "warnings.warn(",
        "Gym has been unmaintained",
        "Please upgrade to Gymnasium",
        "See the migration guide",
        "No OpenGL_accelerate",
        "Could not import robosuite_models",
        "Could not load the mink-based",
        "mimicgen environments not imported",
    )

    structural = re.compile(
        r"^\s*(\(|\)|[A-Za-z0-9_]+\):|[A-Za-z0-9_]+\(|"
        r"\(.*\):|\d+-\d+\):)"
    )

    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        if any(token in line for token in ignored):
            continue
        if line.startswith("/data/") and "RuntimeWarning:" in line:
            continue
        if structural.match(line) and len(line) < 100:
            continue
        return line[:120]

    return "waiting for log output"


def tail_has_failure(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    last = "\n".join(lines[-60:])

    if "Traceback (most recent call last)" in last:
        return True

    fatal_tokens = (
        "RuntimeError:",
        "InterpolationKeyError:",
        "AttributeError:",
        "ModuleNotFoundError:",
        "CUDA out of memory",
    )
    return any(token in last for token in fatal_tokens)


def process_elapsed(pid):
    try:
        output = subprocess.check_output(
            ["ps", "-o", "etimes=", "-p", str(pid)],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
        return int(output) if output else 0
    except Exception:
        return 0


def human_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm%02ds" % (
            seconds // 60,
            seconds % 60,
        )
    return "%dh%02dm" % (
        seconds // 3600,
        (seconds % 3600) // 60,
    )


def list_active_processes():
    active = {}
    by_gpu = {}

    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return active, by_gpu

    for entry in entries:
        if not entry.name.isdigit():
            continue

        pid = int(entry.name)

        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (OSError, IOError):
            continue

        relevant = (
            "build_feature_cache.py",
            "train_adapter.py",
            "scripts/plan.py",
            "run_all_a100_experiments.sh",
        )
        if not any(token in command for token in relevant):
            continue

        gpu = "?"
        try:
            environment = (entry / "environ").read_bytes()
            for item in environment.split(b"\0"):
                if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    gpu = (
                        item.split(b"=", 1)[1]
                        .decode("utf-8", errors="replace")
                        .split(",")[0]
                    )
                    break
        except (OSError, IOError):
            pass

        method_match = re.search(
            r"(?:^|\s)method=([A-Za-z0-9_]+)",
            command,
        )
        domain_match = re.search(
            r"(?:^|\s)domain=([A-Za-z0-9_]+)",
            command,
        )

        method = method_match.group(1) if method_match else None
        domain = domain_match.group(1) if domain_match else None

        if "build_feature_cache.py" in command:
            key = "cache"
            label = "feature cache"
        elif "train_adapter.py" in command:
            key = "train:%s" % (method or "unknown")
            label = "train %s" % (method or "unknown")
        elif "scripts/plan.py" in command:
            key = "plan:%s:%s" % (
                method or "unknown",
                domain or "unknown",
            )
            label = "plan %s/%s" % (
                method or "unknown",
                domain or "unknown",
            )
        else:
            key = "launcher"
            label = "launcher"

        info = {
            "pid": pid,
            "gpu": gpu,
            "label": label,
            "elapsed": process_elapsed(pid),
        }
        active[key] = info

        if gpu != "?":
            by_gpu[gpu] = info

    return active, by_gpu


def gpu_stats():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,"
                "memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except Exception:
        return {}

    stats = {}

    for line in output.splitlines():
        try:
            index, util, used, total, temperature = [
                int(part.strip())
                for part in line.split(",")
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


def assets_status():
    object_root = (
        ROOT
        / "third_party"
        / "robocasa"
        / "robocasa"
        / "models"
        / "assets"
        / "objects"
    )

    if object_root.is_dir():
        return 100.0, "DONE", "RoboCasa object assets present"

    return 0.0, "WAIT", "RoboCasa object assets missing"


def cache_status(active):
    if (
        CACHE_PATH.is_file()
        and CACHE_PATH.stat().st_size > 0
        and "cache" not in active
    ):
        return 100.0, "DONE", "shared cache exists"

    log = LOG_ROOT / "build_feature_cache.log"
    text = tail_text(log)
    match = parse_tqdm(text)

    if match:
        _, percent, done, total = match
        return (
            percent,
            "RUN",
            "%d/%d | %s"
            % (done, total, human_age(file_age(log))),
        )

    if "cache" in active:
        return (
            1.0,
            "RUN",
            "%s | %s"
            % (
                latest_meaningful_line(text),
                human_age(file_age(log)),
            ),
        )

    if tail_has_failure(text):
        return 0.0, "FAIL", latest_meaningful_line(text)

    return 0.0, "WAIT", "not started"


def training_status(method, active):
    checkpoint = CKPT_ROOT / ("%s_final.pt" % method)

    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return 100.0, "DONE", "checkpoint complete"

    log = LOG_ROOT / ("train_%s.log" % method)
    text = tail_text(log)
    active_key = "train:%s" % method

    pattern = (
        r"(?i)epoch\s+(\d+)\s*/\s*(\d+)"
        r"[^\n]*?(\d{1,3})%"
    )
    matches = list(re.finditer(pattern, text))

    if matches:
        match = matches[-1]
        epoch = int(match.group(1))
        total = int(match.group(2))
        inner = int(match.group(3))
        overall = (
            100.0
            * ((epoch - 1) + inner / 100.0)
            / max(total, 1)
        )
        return (
            overall,
            "RUN",
            "epoch %d/%d, batch %d%% | %s"
            % (
                epoch,
                total,
                inner,
                human_age(file_age(log)),
            ),
        )

    completed = [
        int(match.group(1))
        for match in re.finditer(
            r"(?i)\bepoch=(\d+)\b",
            text,
        )
    ]

    if completed:
        epoch = max(completed)
        percent = (
            100.0
            * epoch
            / max(TRAIN_EPOCHS, 1)
        )
        state = "RUN" if active_key in active else "WAIT"
        return (
            percent,
            state,
            "completed epoch %d/%d | %s"
            % (
                epoch,
                TRAIN_EPOCHS,
                human_age(file_age(log)),
            ),
        )

    if active_key in active:
        return (
            0.5,
            "RUN",
            "%s | %s"
            % (
                latest_meaningful_line(text),
                human_age(file_age(log)),
            ),
        )

    if tail_has_failure(text):
        return 0.0, "FAIL", latest_meaningful_line(text)

    return 0.0, "WAIT", "not started"


def planning_milestone(text):
    milestones = (
        (
            3.0,
            ("Loading checkpoint from local path",),
            "loading world model",
        ),
        (
            6.0,
            ("loaded pretrained predictor",),
            "world model loaded",
        ),
        (
            9.0,
            ("Loaded 14 trajectories", "Total dataset:"),
            "dataset loaded",
        ),
        (
            12.0,
            ("Initializing GC_Agent",),
            "agent initialization",
        ),
        (
            15.0,
            ("Loading controller configuration",),
            "controller initialization",
        ),
        (
            18.0,
            ("Wrapped RoboCasa environment",),
            "environment created",
        ),
        (
            20.0,
            ("Resetting from provided model XML",),
            "environment reset",
        ),
        (
            22.0,
            ("Finished resetting from provided model XML",),
            "environment ready",
        ),
        (
            24.0,
            ("CEM", "planning", "PlanEvaluator"),
            "planning loop",
        ),
    )

    best = (1.0, "process initialization")

    for percent, tokens, label in milestones:
        if any(token in text for token in tokens):
            best = (percent, label)

    return best


def planning_status(method, domain, active):
    if result_exists(method, domain):
        return 100.0, "DONE", "results.json complete"

    log = LOG_ROOT / (
        "plan_%s_%s.log" % (method, domain)
    )
    text = tail_text(log)
    key = "plan:%s:%s" % (method, domain)
    is_active = key in active

    tqdm_match = parse_tqdm(text, PLAN_EPISODES)

    if tqdm_match:
        _, percent, done, total = tqdm_match
        warning_count = recent_warning_count(text)
        detail = "episode %d/%d | %s" % (
            done,
            total,
            human_age(file_age(log)),
        )
        if warning_count:
            detail += (
                " | recent bad-object skips=%d"
                % warning_count
            )
        return (
            percent,
            "RUN" if is_active else "WAIT",
            detail,
        )

    episode = latest_episode_number(
        text,
        PLAN_EPISODES,
    )

    if episode is not None:
        percent = (
            100.0
            * episode
            / max(PLAN_EPISODES, 1)
        )
        warning_count = recent_warning_count(text)
        detail = "episode %d/%d | %s" % (
            episode,
            PLAN_EPISODES,
            human_age(file_age(log)),
        )
        if warning_count:
            detail += (
                " | recent bad-object skips=%d"
                % warning_count
            )
        return (
            percent,
            "RUN" if is_active else "WAIT",
            detail,
        )

    if is_active:
        percent, phase = planning_milestone(text)
        warning_count = recent_warning_count(text)

        detail = "%s | %s | %s" % (
            phase,
            human_age(file_age(log)),
            latest_meaningful_line(text),
        )

        if warning_count:
            detail += (
                " | recent bad-object skips=%d"
                % warning_count
            )

        return percent, "RUN", detail

    if tail_has_failure(text):
        return 0.0, "FAIL", latest_meaningful_line(text)

    if text:
        percent, phase = planning_milestone(text)
        return (
            percent,
            "WAIT",
            "%s | process inactive | %s"
            % (
                phase,
                latest_meaningful_line(text),
            ),
        )

    return 0.0, "WAIT", "not started"


def state_marker(state):
    return {
        "DONE": "[OK]  ",
        "RUN": "[RUN] ",
        "WAIT": "[--]  ",
        "FAIL": "[FAIL]",
    }.get(state, "[%s]" % state)


def print_progress(
    label,
    percent,
    state,
    detail,
    indent="",
):
    active = state == "RUN" and percent < 1.0

    print(
        "%s%-24s %s [%s] %6.2f%%  %s"
        % (
            indent,
            label,
            state_marker(state),
            bar(percent, active=active),
            clamp(percent),
            detail,
        )
    )


def draw():
    active, by_gpu = list_active_processes()
    stats = gpu_stats()

    asset_pct, asset_state, asset_detail = (
        assets_status()
    )
    cache_pct, cache_state, cache_detail = (
        cache_status(active)
    )

    training = {}
    for method in METHODS:
        training[method] = training_status(
            method,
            active,
        )

    planning = {}
    for method in PLAN_METHODS:
        for domain in DOMAINS:
            planning[(method, domain)] = (
                planning_status(
                    method,
                    domain,
                    active,
                )
            )

    train_avg = (
        sum(item[0] for item in training.values())
        / len(training)
    )
    plan_avg = (
        sum(item[0] for item in planning.values())
        / len(planning)
    )

    overall = (
        0.05 * asset_pct
        + 0.10 * cache_pct
        + 0.25 * train_avg
        + 0.60 * plan_avg
    )

    sys.stdout.write("\033[2J\033[H")
    print(
        "WM Adapter Full-Pipeline Dashboard  %s"
        % time.strftime("%Y-%m-%d %H:%M:%S")
    )
    print("Project: %s" % ROOT)
    print(
        "Overall [%s] %6.2f%%"
        % (bar(overall, 42), overall)
    )
    print("=" * 126)

    print("\nPIPELINE")
    print_progress(
        "1. Assets",
        asset_pct,
        asset_state,
        asset_detail,
    )
    print_progress(
        "2. Feature cache",
        cache_pct,
        cache_state,
        cache_detail,
    )

    training_state = (
        "DONE"
        if train_avg >= 100
        else (
            "RUN"
            if any(
                value[1] == "RUN"
                for value in training.values()
            )
            else "WAIT"
        )
    )
    print_progress(
        "3. Adapter training",
        train_avg,
        training_state,
        "%d/%d methods complete"
        % (
            sum(
                1
                for value in training.values()
                if value[0] >= 100
            ),
            len(training),
        ),
    )

    planning_state = (
        "DONE"
        if plan_avg >= 100
        else (
            "RUN"
            if any(
                value[1] == "RUN"
                for value in planning.values()
            )
            else "WAIT"
        )
    )
    print_progress(
        "4. Planning",
        plan_avg,
        planning_state,
        "%d/8 jobs complete | %d episodes/job"
        % (
            sum(
                1
                for value in planning.values()
                if value[0] >= 100
            ),
            PLAN_EPISODES,
        ),
    )

    print("\nGPU TASKS")
    if not stats:
        print("  nvidia-smi unavailable")

    for gpu in sorted(
        stats,
        key=lambda value: int(value),
    ):
        stat = stats[gpu]
        task = by_gpu.get(gpu)

        if task:
            label = "%s pid=%d time=%s" % (
                task["label"],
                task["pid"],
                human_duration(task["elapsed"]),
            )
        else:
            label = "idle"

        print(
            "  GPU %s | util %3d%% | "
            "mem %5d/%5d MiB | temp %2dC | %s"
            % (
                gpu,
                stat["util"],
                stat["used"],
                stat["total"],
                stat["temp"],
                label,
            )
        )

    print("\nTRAINING")
    for method in METHODS:
        percent, state, detail = training[method]
        print_progress(
            method,
            percent,
            state,
            detail,
            indent="  ",
        )

    print("\nPLANNING")
    for method in PLAN_METHODS:
        for domain in DOMAINS:
            percent, state, detail = planning[
                (method, domain)
            ]
            print_progress(
                "%s/%s" % (method, domain),
                percent,
                state,
                detail,
                indent="  ",
            )

    print(
        "\nPlanning uses exact episode progress when "
        "upstream writes it to the log."
    )
    print(
        "Before the first episode it shows initialization "
        "milestones, log heartbeat and bad-object skips."
    )
    print(
        "Ctrl+C to exit | refresh %.1fs"
        % REFRESH
    )
    sys.stdout.flush()


def main():
    try:
        while True:
            draw()
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
