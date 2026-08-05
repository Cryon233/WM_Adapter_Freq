from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wm_adapter.utils.reproducibility import resolve_path


BOOTSTRAP_REPLICATES = 10000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Artifact root must be a mapping: {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: Iterable[str],
) -> None:
    fieldnames = list(fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _rate(payload: dict[str, Any]) -> float:
    values = [
        bool(value) for value in payload["per_episode_success"]
    ]
    if not values:
        raise RuntimeError("Planning artifact has no episodes")
    return sum(values) / len(values)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return (
        statistics.mean(values),
        statistics.stdev(values)
        if len(values) > 1
        else 0.0,
    )


def _comparison_seed(key: tuple[str, ...]) -> int:
    digest = hashlib.sha256(
        "|".join(key).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _records_by_seed(
    main_payloads: dict[
        tuple[str, str, str, int, str],
        dict[str, Any],
    ],
    *,
    backend: str,
    task: str,
    method: str,
    domain: str,
) -> dict[int, dict[str, Any]]:
    return {
        seed: record
        for (
            record_backend,
            record_task,
            record_method,
            seed,
            record_domain,
        ), record in main_payloads.items()
        if record_backend == backend
        and record_task == task
        and record_method == method
        and record_domain == domain
    }


def _paired_hierarchical_bootstrap(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    *,
    key: tuple[str, ...],
) -> tuple[float, float, float, int]:
    if set(left) != set(right) or not left:
        raise RuntimeError(
            "Paired comparison requires matching non-empty seeds: "
            f"left={sorted(left)}, right={sorted(right)}, key={key}"
        )
    seeds = sorted(left)
    per_seed_differences: list[np.ndarray] = []
    for seed in seeds:
        left_success = np.asarray(
            left[seed]["per_episode_success"],
            dtype=np.float64,
        )
        right_success = np.asarray(
            right[seed]["per_episode_success"],
            dtype=np.float64,
        )
        if (
            left_success.ndim != 1
            or left_success.shape != right_success.shape
            or left_success.size == 0
        ):
            raise RuntimeError(
                "Paired success arrays are incompatible: "
                f"key={key}, seed={seed}, "
                f"left={left_success.shape}, "
                f"right={right_success.shape}"
            )
        left_ids = left[seed].get(
            "evaluation_instance_ids", []
        )
        right_ids = right[seed].get(
            "evaluation_instance_ids", []
        )
        if left_ids and right_ids and left_ids != right_ids:
            raise RuntimeError(
                "Paired planning artifacts use different "
                f"evaluation instances: key={key}, seed={seed}"
            )
        per_seed_differences.append(
            left_success - right_success
        )

    observed = float(
        np.mean(
            [
                float(values.mean())
                for values in per_seed_differences
            ]
        )
    )
    rng = np.random.default_rng(_comparison_seed(key))
    replicates = np.empty(
        BOOTSTRAP_REPLICATES,
        dtype=np.float64,
    )
    seed_count = len(per_seed_differences)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_seeds = rng.integers(
            0,
            seed_count,
            size=seed_count,
        )
        sampled_seed_means: list[float] = []
        for selected_seed in selected_seeds:
            values = per_seed_differences[
                int(selected_seed)
            ]
            selected_episodes = rng.integers(
                0,
                values.size,
                size=values.size,
            )
            sampled_seed_means.append(
                float(values[selected_episodes].mean())
            )
        replicates[replicate] = float(
            np.mean(sampled_seed_means)
        )
    lower, upper = np.quantile(
        replicates,
        [0.025, 0.975],
    )
    return (
        observed,
        float(lower),
        float(upper),
        seed_count,
    )


def _format_mean_std(
    records: list[dict[str, Any]],
) -> str:
    mean, std = _mean_std(
        [float(record["success_rate"]) for record in records]
    )
    return f"{mean:.3f} ± {std:.3f}"


def main() -> None:
    args = _arguments()
    state = _json(args.state)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = state.get("jobs", {})
    if not isinstance(jobs, dict):
        raise TypeError("Suite state jobs must be a mapping")

    main_payloads: dict[
        tuple[str, str, str, int, str],
        dict[str, Any],
    ] = {}
    ablation_payloads: list[dict[str, Any]] = []
    offline_payloads: list[dict[str, Any]] = []
    efficiency: list[dict[str, Any]] = []

    for job_id, entry in jobs.items():
        if entry.get("status") not in {
            "completed",
            "reused",
        }:
            continue
        artifact = (
            entry.get("reuse_source")
            or entry.get("artifact_path")
        )
        if (
            not artifact
            or not resolve_path(str(artifact)).is_file()
        ):
            continue

        if entry.get("kind") == "planning":
            payload = _json(str(artifact))
            successes = [
                bool(value)
                for value in payload[
                    "per_episode_success"
                ]
            ]
            record = {
                "backend": payload["backend"],
                "task": payload["task"],
                "method": payload["method"],
                "seed": int(
                    payload["seeds"]["training"]
                ),
                "domain": payload["domain"],
                "success_count": int(
                    payload["success_count"]
                ),
                "episodes": int(
                    payload["total_episodes"]
                ),
                "success_rate": _rate(payload),
                "per_episode_success": successes,
                "evaluation_instance_ids": list(
                    payload.get(
                        "evaluation_instance_ids",
                        [],
                    )
                ),
                "source_path": str(
                    resolve_path(str(artifact))
                ),
            }
            if str(job_id).startswith(
                "planning/main/"
            ):
                key = (
                    str(record["backend"]),
                    str(record["task"]),
                    str(record["method"]),
                    int(record["seed"]),
                    str(record["domain"]),
                )
                if key in main_payloads:
                    raise RuntimeError(
                        f"Duplicate main planning result: {key}"
                    )
                main_payloads[key] = record
            else:
                ablation_payloads.append(
                    record | {"variant": "core_only"}
                )

            elapsed = float(
                payload.get(
                    "elapsed_seconds",
                    payload.get(
                        "runtime_seconds",
                        0.0,
                    ),
                )
            )
            available = int(
                payload["total_episodes"]
            )
            efficiency.append(
                record
                | {
                    "source_elapsed_seconds": elapsed,
                    "source_available_episodes": available,
                    "used_episodes": available,
                    "elapsed_seconds_per_episode": (
                        elapsed / available
                    ),
                    "peak_cuda_memory_bytes": int(
                        payload.get(
                            "peak_cuda_memory_bytes",
                            0,
                        )
                    ),
                    "parameter_count": int(
                        payload.get(
                            "method_parameter_count",
                            0,
                        )
                    ),
                }
            )
        elif entry.get("kind") == "offline":
            offline_payloads.append(
                _json(str(artifact))
            )

    long_rows = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "per_episode_success",
                "evaluation_instance_ids",
            }
        }
        for _, record in sorted(main_payloads.items())
    ]
    long_fields = [
        "backend",
        "task",
        "method",
        "seed",
        "domain",
        "success_count",
        "episodes",
        "success_rate",
        "source_path",
    ]
    _csv(
        output / "planning_results_long.csv",
        long_rows,
        long_fields,
    )

    paired_rows: list[dict[str, Any]] = []
    identities = sorted(
        {
            (
                backend,
                task,
                method,
                seed,
            )
            for (
                backend,
                task,
                method,
                seed,
                _domain,
            ) in main_payloads
        }
    )
    for backend, task, method, seed in identities:
        clean = main_payloads.get(
            (
                backend,
                task,
                method,
                seed,
                "clean",
            )
        )
        ood = main_payloads.get(
            (
                backend,
                task,
                method,
                seed,
                "ood",
            )
        )
        paired_rows.append(
            {
                "backend": backend,
                "task": task,
                "method": method,
                "seed": seed,
                "clean_success_count": (
                    None
                    if clean is None
                    else clean["success_count"]
                ),
                "clean_n": (
                    None
                    if clean is None
                    else clean["episodes"]
                ),
                "clean_success_rate": (
                    None
                    if clean is None
                    else clean["success_rate"]
                ),
                "ood_success_count": (
                    None
                    if ood is None
                    else ood["success_count"]
                ),
                "ood_n": (
                    None
                    if ood is None
                    else ood["episodes"]
                ),
                "ood_success_rate": (
                    None
                    if ood is None
                    else ood["success_rate"]
                ),
                "clean_to_ood_drop": (
                    None
                    if clean is None or ood is None
                    else (
                        float(clean["success_rate"])
                        - float(ood["success_rate"])
                    )
                ),
            }
        )

    paired_fields = [
        "backend",
        "task",
        "method",
        "seed",
        "clean_success_count",
        "clean_n",
        "clean_success_rate",
        "ood_success_count",
        "ood_n",
        "ood_success_rate",
        "clean_to_ood_drop",
    ]
    _csv(
        output / "jepa_wm_main_results.csv",
        [
            row
            for row in paired_rows
            if row["backend"] == "jepa_wm_droid"
        ],
        paired_fields,
    )
    _csv(
        output / "dino_wm_main_results.csv",
        [
            row
            for row in paired_rows
            if row["backend"] == "dino_wm_droid"
        ],
        paired_fields,
    )

    grouped_domain: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for record in main_payloads.values():
        grouped_domain[
            (
                str(record["backend"]),
                str(record["task"]),
                str(record["method"]),
                str(record["domain"]),
            )
        ].append(record)

    markdown = [
        "# Cross-backend adapter results",
        "",
        "The primary confirmatory matrix is OOD-only "
        "with three seeds. Clean evaluation is a "
        "single-seed Base/HFRA guardrail and is not "
        "used as a three-seed superiority test.",
        "",
        "Backends are reported separately; raw latent "
        "MSE is never averaged across latent spaces.",
        "",
        "## Primary OOD results",
        "",
        "| backend | task | method | OOD success "
        "mean ± std | seeds | episodes/seed |",
        "|---|---|---|---:|---:|---:|",
    ]
    for (
        backend,
        task,
        method,
        domain,
    ), records in sorted(grouped_domain.items()):
        if domain != "ood":
            continue
        episode_counts = {
            int(record["episodes"])
            for record in records
        }
        if len(episode_counts) != 1:
            raise RuntimeError(
                "OOD aggregation contains inconsistent "
                f"episode counts: {(backend, task, method)} "
                f"-> {sorted(episode_counts)}"
            )
        markdown.append(
            f"| {backend} | {task} | {method} | "
            f"{_format_mean_std(records)} | "
            f"{len(records)} | "
            f"{next(iter(episode_counts))} |"
        )

    comparison_rows: list[dict[str, Any]] = []
    markdown.extend(
        [
            "",
            "## Paired OOD HFRA improvements",
            "",
            "Intervals use a deterministic hierarchical "
            "paired bootstrap: seeds are resampled first "
            "and manifest-aligned episodes are resampled "
            "within each selected seed.",
            "",
            "| backend | task | comparison | Δ success | "
            "95% CI | seeds |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    backend_tasks = sorted(
        {
            (backend, task)
            for backend, task, _, _, domain
            in main_payloads
            if domain == "ood"
        }
    )
    for backend, task in backend_tasks:
        hfra = _records_by_seed(
            main_payloads,
            backend=backend,
            task=task,
            method="hfra",
            domain="ood",
        )
        for baseline in ("base", "lora"):
            baseline_records = _records_by_seed(
                main_payloads,
                backend=backend,
                task=task,
                method=baseline,
                domain="ood",
            )
            delta, lower, upper, seed_count = (
                _paired_hierarchical_bootstrap(
                    hfra,
                    baseline_records,
                    key=(
                        backend,
                        task,
                        "ood",
                        "hfra",
                        baseline,
                    ),
                )
            )
            comparison_rows.append(
                {
                    "backend": backend,
                    "task": task,
                    "domain": "ood",
                    "left_method": "hfra",
                    "right_method": baseline,
                    "success_rate_difference": delta,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "seed_count": seed_count,
                    "bootstrap_replicates": (
                        BOOTSTRAP_REPLICATES
                    ),
                }
            )
            markdown.append(
                f"| {backend} | {task} | "
                f"HFRA−{baseline} | {delta:+.3f} | "
                f"[{lower:+.3f}, {upper:+.3f}] | "
                f"{seed_count} |"
            )

    markdown.extend(
        [
            "",
            "## Clean guardrail",
            "",
            "Clean results use seed 42 only. The paired "
            "episode bootstrap interval does not represent "
            "cross-training-seed uncertainty.",
            "",
            "| backend | task | Base | HFRA | HFRA−Base | "
            "95% paired episode CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for backend, task in backend_tasks:
        hfra_clean = _records_by_seed(
            main_payloads,
            backend=backend,
            task=task,
            method="hfra",
            domain="clean",
        )
        base_clean = _records_by_seed(
            main_payloads,
            backend=backend,
            task=task,
            method="base",
            domain="clean",
        )
        delta, lower, upper, seed_count = (
            _paired_hierarchical_bootstrap(
                hfra_clean,
                base_clean,
                key=(
                    backend,
                    task,
                    "clean",
                    "hfra",
                    "base",
                ),
            )
        )
        base_rate = statistics.mean(
            float(record["success_rate"])
            for record in base_clean.values()
        )
        hfra_rate = statistics.mean(
            float(record["success_rate"])
            for record in hfra_clean.values()
        )
        comparison_rows.append(
            {
                "backend": backend,
                "task": task,
                "domain": "clean",
                "left_method": "hfra",
                "right_method": "base",
                "success_rate_difference": delta,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "seed_count": seed_count,
                "bootstrap_replicates": (
                    BOOTSTRAP_REPLICATES
                ),
            }
        )
        markdown.append(
            f"| {backend} | {task} | "
            f"{base_rate:.3f} | {hfra_rate:.3f} | "
            f"{delta:+.3f} | "
            f"[{lower:+.3f}, {upper:+.3f}] |"
        )

    _csv(
        output / "paired_success_comparisons.csv",
        comparison_rows,
        [
            "backend",
            "task",
            "domain",
            "left_method",
            "right_method",
            "success_rate_difference",
            "ci95_lower",
            "ci95_upper",
            "seed_count",
            "bootstrap_replicates",
        ],
    )
    _atomic_text(
        output / "main_results.md",
        "\n".join(markdown) + "\n",
    )

    mse_rows: list[dict[str, Any]] = []
    for payload in offline_payloads:
        for domain, metrics in payload[
            "domains"
        ].items():
            mse_rows.append(
                {
                    "backend": payload["backend"],
                    "task": payload["task"],
                    "method": payload["method"],
                    "seed": payload.get(
                        "training_seed"
                    ),
                    "domain": domain,
                    **metrics,
                }
            )
    mse_fields = [
        "backend",
        "task",
        "method",
        "seed",
        "domain",
        "h1_autoregressive_latent_mse",
        "h2_autoregressive_latent_mse",
        "h3_autoregressive_latent_mse",
        "future_mean_mse",
        "terminal_mse",
        "unified_6frame_trajectory_mse",
    ]
    _csv(
        output / "mse_results.csv",
        mse_rows,
        mse_fields,
    )
    mse_md = [
        "# Offline autoregressive MSE",
        "",
        "Raw values are only comparable within a backend "
        "latent space. Base is evaluated once; learned "
        "methods report mean ± sample standard deviation "
        "over the three training seeds.",
        "",
        "| backend | task | method | domain | H1 | H2 | "
        "H3 | future mean | trajectory |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    metric_names = [
        "h1_autoregressive_latent_mse",
        "h2_autoregressive_latent_mse",
        "h3_autoregressive_latent_mse",
        "future_mean_mse",
        "unified_6frame_trajectory_mse",
    ]
    grouped_mse: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in mse_rows:
        grouped_mse[
            (
                row["backend"],
                row["task"],
                row["method"],
                row["domain"],
            )
        ].append(row)
    for key, rows in sorted(grouped_mse.items()):
        rendered: list[str] = []
        for metric in metric_names:
            mean, std = _mean_std(
                [
                    float(row[metric])
                    for row in rows
                ]
            )
            rendered.append(
                f"{mean:.6g}"
                if key[2] == "base"
                else f"{mean:.6g} ± {std:.3g}"
            )
        mse_md.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{key[3]} | "
            + " | ".join(rendered)
            + " |"
        )

    mse_summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped_mse.items()):
        backend, task, method, domain = key
        base_rows = grouped_mse.get(
            (backend, task, "base", domain),
            [],
        )
        if len(base_rows) != 1:
            raise RuntimeError(
                "Offline MSE aggregation requires one "
                "deterministic Base row: "
                f"backend={backend}, task={task}, "
                f"domain={domain}, found={len(base_rows)}"
            )
        for metric in metric_names:
            mean, std = _mean_std(
                [
                    float(row[metric])
                    for row in rows
                ]
            )
            base_value = float(
                base_rows[0][metric]
            )
            relative = (
                (base_value - mean) / base_value
                if base_value > 0.0
                else None
            )
            mse_summary_rows.append(
                {
                    "backend": backend,
                    "task": task,
                    "method": method,
                    "domain": domain,
                    "metric": metric,
                    "seed_count": len(rows),
                    "mean": mean,
                    "std": std,
                    "base_value": base_value,
                    "relative_improvement_over_base": (
                        relative
                    ),
                }
            )
    _csv(
        output / "mse_results_summary.csv",
        mse_summary_rows,
        [
            "backend",
            "task",
            "method",
            "domain",
            "metric",
            "seed_count",
            "mean",
            "std",
            "base_value",
            "relative_improvement_over_base",
        ],
    )
    _atomic_text(
        output / "mse_results.md",
        "\n".join(mse_md) + "\n",
    )

    # Add full HFRA rows by reference to the primary OOD
    # results; no extra full-HFRA ablation jobs are created.
    for (
        backend,
        task,
        method,
        seed,
        domain,
    ), record in sorted(main_payloads.items()):
        if (
            backend == "jepa_wm_droid"
            and task
            in {
                "robocasa_place",
                "libero_goal_0",
            }
            and method == "hfra"
            and domain == "ood"
        ):
            ablation_payloads.append(
                {
                    "backend": backend,
                    "task": task,
                    "method": "hfra",
                    "variant": "full",
                    "seed": seed,
                    "domain": "ood",
                    "success_count": record[
                        "success_count"
                    ],
                    "episodes": record["episodes"],
                    "success_rate": record[
                        "success_rate"
                    ],
                    "source_path": (
                        "primary OOD result reuse"
                    ),
                }
            )

    ablation_fields = [
        "backend",
        "task",
        "variant",
        "method",
        "seed",
        "domain",
        "success_count",
        "episodes",
        "success_rate",
        "source_path",
    ]
    _csv(
        output / "ablation_results.csv",
        ablation_payloads,
        ablation_fields,
    )
    grouped_ablation: dict[
        tuple[str, str, str],
        list[float],
    ] = defaultdict(list)
    for row in ablation_payloads:
        grouped_ablation[
            (
                row["backend"],
                row["task"],
                row["variant"],
            )
        ].append(float(row["success_rate"]))
    ablation_markdown = [
        "# HFRA closed-loop ablation",
        "",
        "Full HFRA rows reuse the corresponding primary "
        "OOD result.",
        "",
        "| backend | task | variant | OOD success "
        "mean ± std | seeds |",
        "|---|---|---|---:|---:|",
    ]
    for key, values in sorted(
        grouped_ablation.items()
    ):
        mean, std = _mean_std(values)
        ablation_markdown.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{mean:.3f} ± {std:.3f} | "
            f"{len(values)} |"
        )
    _atomic_text(
        output / "ablation_results.md",
        "\n".join(ablation_markdown) + "\n",
    )
    _csv(
        output / "efficiency.csv",
        efficiency,
        [
            "backend",
            "task",
            "method",
            "seed",
            "domain",
            "source_elapsed_seconds",
            "source_available_episodes",
            "used_episodes",
            "elapsed_seconds_per_episode",
            "peak_cuda_memory_bytes",
            "parameter_count",
            "source_path",
        ],
    )


if __name__ == "__main__":
    main()
