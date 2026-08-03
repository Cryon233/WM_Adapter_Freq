from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wm_adapter.benchmarks.base import atomic_json
from wm_adapter.experiments.cross_benchmark import load_suite_config
from wm_adapter.utils.checkpoints import sha256_file
from wm_adapter.utils.reproducibility import resolve_path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _wilson(success: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def _source(entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    validation = dict(entry.get("artifact_validation") or {})
    path = resolve_path(str(validation.get("path") or entry.get("reuse_source") or entry["artifact_path"]))
    return path, validation


def _planning_record(entry: dict[str, Any]) -> dict[str, Any]:
    path, validation = _source(entry)
    payload = json.loads(path.read_text(encoding="utf-8"))
    used = int(entry["required_count"])
    values = [bool(value) for value in payload["per_episode_success"][:used]]
    ids = list(validation.get("evaluation_instance_ids") or payload.get("evaluation_instance_ids", []))[:used]
    if len(ids) != used:
        raise RuntimeError(f"Planning artifact lacks {used} paired instance IDs: {path}")
    source_ids = list(
        validation.get("source_trajectory_ids")
        or payload.get("source_trajectory_ids", [])
    )[:used]
    if len(source_ids) != used:
        source_ids = ids
    elapsed = float(payload.get("elapsed_seconds", 0.0))
    available = len(payload["per_episode_success"])
    if available <= 0:
        raise RuntimeError(f"Planning artifact has no source episodes: {path}")
    low, high = _wilson(sum(values), used)
    return {
        "benchmark": entry["benchmark"], "task": entry["task"],
        "method": entry["method"], "domain": entry["domain"],
        "seed": entry["seed"], "severity": entry["severity"],
        "variant": entry.get("variant"), "success_count": sum(values),
        "n": used, "success_rate": sum(values) / used,
        "wilson_low": low, "wilson_high": high,
        "successes": values, "instance_ids": ids, "source_ids": source_ids,
        "source_path": str(path), "source_sha256": sha256_file(path),
        "source_elapsed_seconds": elapsed,
        "source_available_episodes": available, "used_episodes": used,
        "elapsed_seconds_per_episode": elapsed / available,
        "peak_cuda_memory_bytes": int(payload.get("peak_cuda_memory_bytes", 0)),
        "parameter_count": int(payload.get("method_parameter_count", 0)),
        "source_checkpoint_fingerprint": payload.get("method_checkpoint_sha256"),
        "goal_base_latent_fingerprints": list(
            payload.get("goal_base_latent_fingerprint", [])
        )[:used],
    }


def _exact_mcnemar(left: list[bool], right: list[bool]) -> float:
    b = sum(a and not c for a, c in zip(left, right))
    c = sum(not a and c for a, c in zip(left, right))
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_bootstrap(
    left: list[bool], right: list[bool], *, samples: int, seed: int,
    clusters: list[str] | None = None,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    estimates = np.empty(samples, dtype=np.float64)
    if clusters is None:
        for index in range(samples):
            selection = generator.integers(0, len(differences), size=len(differences))
            estimates[index] = float(differences[selection].mean())
    else:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, cluster in enumerate(clusters):
            grouped[cluster].append(index)
        names = sorted(grouped)
        for index in range(samples):
            chosen = generator.integers(0, len(names), size=len(names))
            values = np.concatenate([differences[grouped[names[int(item)]]] for item in chosen])
            estimates[index] = float(values.mean())
    return {
        "mean": float(differences.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def _holm(records: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(records), key=lambda item: float(item[1]["mcnemar_p"]))
    adjusted = [1.0] * len(records)
    running = 0.0
    count = len(records)
    for rank, (index, record) in enumerate(ordered):
        value = min(1.0, (count - rank) * float(record["mcnemar_p"]))
        running = max(running, value)
        adjusted[index] = running
    for record, value in zip(records, adjusted):
        record["holm_adjusted_p"] = value


def _markdown_main(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Benchmark | Task | Method | Clean success (95% Wilson CI) | OOD success (95% Wilson CI) | Clean→OOD drop |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['task']} | {row['method']} | "
            f"{row['clean_success_count']}/{row['clean_n']} ({row['clean_success_rate']:.3f}; "
            f"[{row['clean_wilson_low']:.3f}, {row['clean_wilson_high']:.3f}]) | "
            f"{row['ood_success_count']}/{row['ood_n']} ({row['ood_success_rate']:.3f}; "
            f"[{row['ood_wilson_low']:.3f}, {row['ood_wilson_high']:.3f}]) | "
            f"{row['clean_to_ood_drop']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _main_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(item["task"], item["method"], item["domain"]): item for item in records}
    rows: list[dict[str, Any]] = []
    for task, method in sorted({(item["task"], item["method"]) for item in records}):
        clean = indexed[(task, method, "clean")]
        ood = indexed[(task, method, "ood")]
        rows.append({
            "benchmark": clean["benchmark"], "task": task, "method": method,
            "clean_success_count": clean["success_count"], "clean_n": clean["n"],
            "clean_success_rate": clean["success_rate"],
            "clean_wilson_low": clean["wilson_low"], "clean_wilson_high": clean["wilson_high"],
            "ood_success_count": ood["success_count"], "ood_n": ood["n"],
            "ood_success_rate": ood["success_rate"],
            "ood_wilson_low": ood["wilson_low"], "ood_wilson_high": ood["wilson_high"],
            "clean_to_ood_drop": clean["success_rate"] - ood["success_rate"],
        })
    return rows


def _offline_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job_id, entry in state["jobs"].items():
        if entry.get("kind") != "offline" or entry.get("status") not in {"completed", "reused"}:
            continue
        path, _ = _source(entry)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for domain, metrics in payload["domains"].items():
            row = {
                "job_id": job_id, "benchmark": entry["benchmark"], "task": entry["task"],
                "method": entry["method"], "variant": entry.get("variant"),
                "domain": domain, "window_count": payload["window_count"],
                "source_path": str(path),
            }
            row.update(metrics)
            rows.append(row)
    return rows


def _write_analysis(suite: Any, state: dict[str, Any], output: Path) -> None:
    is_v2 = str(suite.suite_name) == "cross_benchmark_v2"
    planning: dict[str, dict[str, Any]] = {}
    for job_id, entry in state["jobs"].items():
        if entry.get("kind") == "planning" and entry.get("status") in {"completed", "reused"}:
            planning[job_id] = _planning_record(entry)
    main_records = [
        value for key, value in planning.items() if key.startswith("planning/main/")
    ]
    goal_groups: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for record in main_records:
        values = tuple(record["goal_base_latent_fingerprints"])
        if len(values) != int(record["n"]):
            raise RuntimeError(
                "Planning result lacks frozen-Base goal fingerprints: "
                f"task={record['task']}, method={record['method']}, domain={record['domain']}"
            )
        goal_groups[record["task"]].add(values)
    inconsistent_goals = {
        task: len(values)
        for task, values in goal_groups.items()
        if len(values) != 1
    }
    if inconsistent_goals:
        raise RuntimeError(
            "Frozen-Base goal fingerprints differ across methods: "
            f"{inconsistent_goals}"
        )
    expected_main = len(suite.tasks) * len(suite.methods) * len(suite.domains)
    if len(main_records) != expected_main:
        raise RuntimeError(
            f"Main analysis expects {expected_main} conditions, found {len(main_records)}"
        )
    main_rows = _main_rows(main_records)
    fields = list(main_rows[0])
    _write_csv(output / "main_results.csv", main_rows, fields)
    markdown = _markdown_main(main_rows)
    _atomic_text(output / "main_results.md", markdown)
    latex_lines = ["\\begin{tabular}{lllrrr}", "Benchmark & Task & Method & Clean & OOD & Drop \\\\", "\\hline"]
    for row in main_rows:
        latex_lines.append(
            f"{row['benchmark']} & {row['task']} & {row['method']} & "
            f"{row['clean_success_count']}/{row['clean_n']} "
            f"[{row['clean_wilson_low']:.3f},{row['clean_wilson_high']:.3f}] & "
            f"{row['ood_success_count']}/{row['ood_n']} "
            f"[{row['ood_wilson_low']:.3f},{row['ood_wilson_high']:.3f}] & "
            f"{row['clean_to_ood_drop']:.3f} \\\\"
        )
    latex_lines.append("\\end{tabular}")
    _atomic_text(output / "main_results.tex", "\n".join(latex_lines) + "\n")

    macro_rows: list[dict[str, Any]] = []
    for method in suite.methods:
        method_rows = [row for row in main_rows if row["method"] == str(method)]
        for benchmark in sorted({row["benchmark"] for row in method_rows}):
            selected = [row for row in method_rows if row["benchmark"] == benchmark]
            macro_rows.append({
                "scope": benchmark, "method": str(method),
                "clean_macro_average": float(np.mean([row["clean_success_rate"] for row in selected])),
                "ood_macro_average": float(np.mean([row["ood_success_rate"] for row in selected])),
                "task_count": len(selected),
            })
        macro_rows.append({
            "scope": "overall_task_macro", "method": str(method),
            "clean_macro_average": float(np.mean([row["clean_success_rate"] for row in method_rows])),
            "ood_macro_average": float(np.mean([row["ood_success_rate"] for row in method_rows])),
            "task_count": len(method_rows),
        })
    _write_csv(output / "benchmark_macro_average.csv", macro_rows, list(macro_rows[0]))

    comparisons: list[dict[str, Any]] = []
    indexed = {(item["task"], item["method"], item["domain"]): item for item in main_records}
    focal_method = "hfra" if is_v2 else "dct_adapter"
    comparators = (
        ("base", "dct_adapter", "token_mlp", "lora")
        if is_v2
        else ("base", "token_mlp", "lora")
    )
    for task in suite.tasks:
        for domain in suite.domains:
            focal = indexed[(str(task), focal_method, str(domain))]
            for comparator in comparators:
                other = indexed[(str(task), comparator, str(domain))]
                if focal["instance_ids"] != other["instance_ids"]:
                    raise RuntimeError(f"Paired instance IDs differ for {task}/{domain}/{comparator}")
                bootstrap = _paired_bootstrap(
                    focal["successes"], other["successes"],
                    samples=int(suite.offline.bootstrap_samples), seed=42,
                )
                record = {
                    "task": str(task), "domain": str(domain),
                    "comparison": f"{focal_method}_vs_{comparator}",
                    "paired_success_difference": bootstrap["mean"],
                    "bootstrap_ci_low": bootstrap["ci_low"],
                    "bootstrap_ci_high": bootstrap["ci_high"],
                    "mcnemar_p": _exact_mcnemar(focal["successes"], other["successes"]),
                    "focal_only_success": sum(a and not b for a, b in zip(focal["successes"], other["successes"])),
                    "comparator_only_success": sum(b and not a for a, b in zip(focal["successes"], other["successes"])),
                }
                if focal["benchmark"] == "robocasa":
                    cluster = _paired_bootstrap(
                        focal["successes"], other["successes"],
                        samples=int(suite.offline.bootstrap_samples), seed=42,
                        clusters=focal["source_ids"],
                    )
                    record.update({
                        "cluster_bootstrap_ci_low": cluster["ci_low"],
                        "cluster_bootstrap_ci_high": cluster["ci_high"],
                    })
                comparisons.append(record)
    _holm(comparisons)
    paired_name = "paired_tests.json" if is_v2 else "statistics.json"
    atomic_json(output / paired_name, {"comparisons": comparisons})

    offline = _offline_rows(state)
    _write_csv(output / "offline_metrics.csv", offline, sorted({key for row in offline for key in row}))
    family_files = {
        "planning/stability/": "stability.csv",
        "planning/severity/": "severity.csv",
        "planning/ablation/": "ablations.csv",
    }
    for prefix, filename in family_files.items():
        rows = [
            {key: value for key, value in record.items() if key not in {"successes", "instance_ids", "source_ids"}}
            for job_id, record in planning.items() if job_id.startswith(prefix)
        ]
        _write_csv(output / filename, rows, sorted({key for row in rows for key in row}))
    efficiency = [
        {
            "task": record["task"], "method": record["method"], "domain": record["domain"],
            "source_elapsed_seconds": record["source_elapsed_seconds"],
            "source_available_episodes": record["source_available_episodes"],
            "used_episodes": record["used_episodes"],
            "elapsed_seconds_per_episode": record["elapsed_seconds_per_episode"],
            "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
            "parameter_count": record["parameter_count"], "source_path": record["source_path"],
        }
        for record in main_records
    ]
    _write_csv(output / "efficiency.csv", efficiency, list(efficiency[0]))
    parameter_rows = [
        {
            "method": method,
            "trainable_parameter_count": next(
                (
                    int(record["parameter_count"])
                    for record in main_records
                    if record["method"] == method
                ),
                0,
            ),
        }
        for method in [str(value) for value in suite.methods]
    ]
    _write_csv(output / "parameter_counts.csv", parameter_rows, list(parameter_rows[0]))
    atomic_json(output / "run_manifest.json", {
        "suite": str(suite.suite_name), "protocol": str(suite.protocol),
        "state_path": state.get("state_path"),
        "suite_config_path": state.get("suite_config_path"),
        "suite_config_sha256": state.get("suite_config_sha256"),
        "git": state.get("git"), "jobs": state["jobs"],
    })
    resolved_tasks = []
    for task_key in suite.tasks:
        validation = state["jobs"][f"preflight/{task_key}"]["artifact_validation"]
        task_payload = validation["report"]["task"]
        language = task_payload.get("language_instruction")
        resolved_tasks.append(
            f"{task_key}: {task_payload['task_name']}"
            + (f" ({language})" if language else "")
        )
    summary = [
        f"# {suite.suite_name} paper summary", "",
        "The four resolved main tasks are: " + "; ".join(resolved_tasks) + ".",
        "", "Each main method/domain condition reports n=20 paired closed-loop rollouts. Stability, severity, and closed-loop ablations report n=10.",
        "", "Training/evaluation partitions are deterministic trajectory-level 80/20 splits (seed 42), and every method/domain uses the same immutable evaluation manifest.",
        "", "RoboCasa and LIBERO are separate benchmark/task stacks, but they are not claimed to be independent physical engines because both use MuJoCo/robosuite-family simulation components.",
        "", "Reused artifacts remain at their source paths and are represented through SHA256 source references; no source result is rewritten.",
        "", "When a source contains more episodes than the paper uses, success uses the manifest-aligned prefix while efficiency is normalized by the complete source episode count.",
        "", "Statistical language must follow paired bootstrap, exact McNemar, and Holm-adjusted results above; an unsupported comparison is not described as a significant improvement.",
        "", markdown,
    ]
    _atomic_text(output / "paper_summary.md", "\n".join(summary))
    if is_v2:
        _atomic_text(output / "summary.md", "\n".join(summary))


def main() -> None:
    args = _args()
    suite = load_suite_config(args.suite_config)
    state_path = resolve_path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        failures = [
            job_id for job_id, entry in state.get("jobs", {}).items()
            if entry.get("status") in {"failed", "blocked"}
        ]
        if failures:
            raise RuntimeError(f"Cross-benchmark self-test has failed jobs: {failures}")
        incomplete = [
            job_id
            for job_id, entry in state.get("jobs", {}).items()
            if job_id != "analysis/self_test"
            and entry.get("status") not in {"completed", "reused"}
        ]
        if incomplete:
            raise RuntimeError(
                f"Cross-benchmark self-test has incomplete jobs: {incomplete}"
            )
        report = output.parent / "self_test_report.md"
        _atomic_text(report, f"# {suite.suite_name} self-test\n\nOverall: PASS\n")
        return
    _write_analysis(suite, state, output)


if __name__ == "__main__":
    main()
