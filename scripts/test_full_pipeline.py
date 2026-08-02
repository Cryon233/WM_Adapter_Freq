from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from wm_adapter.experiments.paper_suite import (
    atomic_write_json,
    preflight_resources,
    validate_feature_cache,
    validate_method_checkpoint,
    validate_offline_result,
    validate_planning_result,
)
from wm_adapter.utils.reproducibility import project_root, resolve_path


CONFIG = "configs/experiment/robocasa_self_test.yaml"
TRAINING_COMPLETE_PATTERN = re.compile(r"TRAINING_COMPLETE final_losses=(\{.*\})")


def _report_markdown(report: dict[str, Any]) -> str:
    lines = ["# Full pipeline self-test", "", f"Overall: **{report['status']}**", ""]
    for name, stage in report["stages"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Status: {stage['status']}",
                f"- Elapsed: {stage.get('elapsed_seconds', 0.0):.3f}s",
            ]
        )
        if stage.get("artifacts"):
            lines.append(f"- Artifacts: `{json.dumps(stage['artifacts'], sort_keys=True)}`")
        if stage.get("error"):
            lines.append(f"- Error: `{stage['error']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_report(report: dict[str, Any]) -> None:
    output = resolve_path("outputs/self_test")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "self_test_report.json", report)
    temporary = output / "self_test_report.md.tmp"
    temporary.write_text(_report_markdown(report), encoding="utf-8")
    temporary.replace(output / "self_test_report.md")


def _clean_isolated_directories() -> None:
    root = project_root()
    for relative in (
        "storage/self_test",
        "checkpoints/self_test",
        "outputs/self_test",
        "logs/self_test",
    ):
        path = resolve_path(relative)
        if path.parent == root or root not in path.parents or path.name != "self_test":
            raise RuntimeError(f"Refusing unsafe self-test cleanup path: {path}")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _run(command: list[str], log_path: Path, gpu: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=project_root(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
        raise RuntimeError(
            f"Self-test command failed with code {completed.returncode}; log={log_path}; "
            f"command={command}\nChild process log tail:\n{log_tail}"
        )


def _finite_training_losses(log_path: Path) -> dict[str, float]:
    matches = TRAINING_COMPLETE_PATTERN.findall(log_path.read_text(encoding="utf-8"))
    if not matches:
        raise RuntimeError(f"Training log has no final loss record: {log_path}")
    losses = {key: float(value) for key, value in json.loads(matches[-1]).items()}
    if set(losses) != {"total", "canonical", "dynamics"} or not all(
        math.isfinite(value) for value in losses.values()
    ):
        raise RuntimeError(f"Training produced invalid final losses: {losses}")
    return losses


def _validate_action_shuffle_metrics(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    action_shuffle = payload.get("action_shuffle")
    if not isinstance(action_shuffle, dict):
        raise RuntimeError(f"Offline metrics lack action_shuffle metadata: {metrics_path}")
    permutation = action_shuffle.get("permutation")
    if (
        not isinstance(permutation, list)
        or len(permutation) != 3
        or sorted(permutation) != [0, 1, 2]
        or permutation == [0, 1, 2]
    ):
        raise RuntimeError(
            f"Offline action shuffle is not a non-identity permutation: {action_shuffle}"
        )
    for domain, metrics in payload.get("domains", {}).items():
        for key in ("shuffled_action_mse", "action_shuffle_gap"):
            value = float(metrics.get(key, float("nan")))
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Offline {domain} metric {key} is not finite: {value}"
                )
    return action_shuffle


def main() -> None:
    _clean_isolated_directories()
    report: dict[str, Any] = {
        "schema_version": "jepa_wm_full_pipeline_self_test_v1",
        "status": "RUNNING",
        "started_at_unix": time.time(),
        "stages": {},
    }
    current_stage = "preflight"
    stage_started = time.time()
    try:
        preflight = preflight_resources(CONFIG)
        report["stages"][current_stage] = {
            "status": "PASS",
            "elapsed_seconds": time.time() - stage_started,
            "artifacts": preflight,
        }
        gpu = 0

        current_stage = "feature_cache"
        stage_started = time.time()
        cache_path = resolve_path("storage/self_test/feature_cache.h5")
        _run(
            [sys.executable, "scripts/build_feature_cache.py", "--config", CONFIG],
            resolve_path("logs/self_test/build_feature_cache.log"),
            gpu,
        )
        cache_info = validate_feature_cache(cache_path, 16)
        report["stages"][current_stage] = {
            "status": "PASS",
            "elapsed_seconds": time.time() - stage_started,
            "artifacts": cache_info,
        }

        current_stage = "training"
        stage_started = time.time()
        checkpoints: dict[str, dict[str, Any]] = {}
        for method in ("dct_adapter", "token_mlp", "lora"):
            checkpoint = resolve_path(f"checkpoints/self_test/{method}_final.pt")
            training_log = resolve_path(f"logs/self_test/train_{method}.log")
            _run(
                [
                    sys.executable,
                    "scripts/train_adapter.py",
                    "--config",
                    CONFIG,
                    f"method={method}",
                ],
                training_log,
                gpu,
            )
            checkpoints[method] = validate_method_checkpoint(
                checkpoint,
                method,
                str(cache_info["cache_fingerprint"]),
            )
            checkpoints[method]["final_training_losses"] = _finite_training_losses(
                training_log
            )
        report["stages"][current_stage] = {
            "status": "PASS",
            "elapsed_seconds": time.time() - stage_started,
            "artifacts": checkpoints,
        }

        current_stage = "offline"
        stage_started = time.time()
        offline_artifacts: dict[str, Any] = {}
        for method in ("base", "dct_adapter", "token_mlp", "lora"):
            output = resolve_path(f"outputs/self_test/offline/{method}")
            command = [
                sys.executable,
                "scripts/evaluate_offline_dynamics.py",
                "--config",
                CONFIG,
                f"method={method}",
                "domain=both",
                f"offline.output_directory={output}",
            ]
            _run(command, resolve_path(f"logs/self_test/offline_{method}.log"), gpu)
            metrics_path = output / "metrics.json"
            offline_artifacts[method] = validate_offline_result(metrics_path, 4)
            offline_artifacts[method]["action_shuffle"] = (
                _validate_action_shuffle_metrics(metrics_path)
            )
        report["stages"][current_stage] = {
            "status": "PASS",
            "elapsed_seconds": time.time() - stage_started,
            "artifacts": offline_artifacts,
        }

        current_stage = "planning"
        stage_started = time.time()
        planning_artifacts: dict[str, Any] = {}
        for method in ("base", "dct_adapter", "token_mlp", "lora"):
            for domain in ("clean", "ood"):
                output = resolve_path(f"outputs/self_test/planning/{method}/{domain}")
                _run(
                    [
                        sys.executable,
                        "scripts/plan.py",
                        "--config",
                        CONFIG,
                        f"method={method}",
                        f"domain={domain}",
                        f"output.run_directory={output}",
                    ],
                    resolve_path(f"logs/self_test/plan_{method}_{domain}.log"),
                    gpu,
                )
                planning_artifacts[f"{method}/{domain}"] = validate_planning_result(
                    output / "results.json", 1
                )
        report["stages"][current_stage] = {
            "status": "PASS",
            "elapsed_seconds": time.time() - stage_started,
            "artifacts": planning_artifacts,
        }

        current_stage = "analysis"
        stage_started = time.time()
        analysis = resolve_path("outputs/self_test/analysis")
        _run(
            [
                sys.executable,
                "scripts/analyze_paper_suite.py",
                "--self-test-root",
                str(resolve_path("outputs/self_test")),
                "--output",
                str(analysis),
            ],
            resolve_path("logs/self_test/analysis.log"),
            gpu,
        )
        required = [analysis / "main_results.csv", analysis / "statistics.json", analysis / "paper_summary.md"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Self-test analysis did not create required artifacts: {missing}")
        report["stages"][current_stage] = {
            "status": "PASS",
            "elapsed_seconds": time.time() - stage_started,
            "artifacts": [str(path) for path in required],
        }
        report["status"] = "PASS"
    except Exception as error:
        report["stages"][current_stage] = {
            "status": "FAIL",
            "elapsed_seconds": time.time() - stage_started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        report["status"] = "FAIL"
        raise
    finally:
        report["completed_at_unix"] = time.time()
        report["elapsed_seconds"] = report["completed_at_unix"] - report["started_at_unix"]
        _write_report(report)
    print(resolve_path("outputs/self_test/self_test_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
