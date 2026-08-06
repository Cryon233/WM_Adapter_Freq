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
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Artifact root must be a mapping: {path}")
    return payload


def _csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _seed(key: tuple[str, ...]) -> int:
    digest = hashlib.sha256("|".join(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _records_by_seed(
    records: dict[tuple[str, str, str, int, str], dict[str, Any]],
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
        ), record in records.items()
        if record_backend == backend
        and record_task == task
        and record_method == method
        and record_domain == domain
    }


def _paired_bootstrap(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    *,
    key: tuple[str, ...],
) -> tuple[float, float, float]:
    if set(left) != set(right) or not left:
        raise RuntimeError(
            f"Paired seeds differ for {key}: "
            f"left={sorted(left)}, right={sorted(right)}"
        )
    differences: list[np.ndarray] = []
    for seed in sorted(left):
        left_values = np.asarray(
            left[seed]["per_episode_success"],
            dtype=np.float64,
        )
        right_values = np.asarray(
            right[seed]["per_episode_success"],
            dtype=np.float64,
        )
        if left_values.shape != right_values.shape:
            raise RuntimeError(
                f"Paired episode counts differ for {key}, seed={seed}"
            )
        if (
            left[seed]["evaluation_instance_ids"]
            != right[seed]["evaluation_instance_ids"]
        ):
            raise RuntimeError(
                f"Paired instances differ for {key}, seed={seed}"
            )
        differences.append(left_values - right_values)

    observed = float(np.mean([values.mean() for values in differences]))
    rng = np.random.default_rng(_seed(key))
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    seed_count = len(differences)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_means: list[float] = []
        for chosen_seed in chosen_seeds:
            values = differences[int(chosen_seed)]
            chosen_episodes = rng.integers(
                0,
                values.size,
                size=values.size,
            )
            seed_means.append(float(values[chosen_episodes].mean()))
        replicates[replicate] = float(np.mean(seed_means))
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return observed, float(lower), float(upper)


def main() -> None:
    args = _arguments()
    state = _json(args.state)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = state.get("jobs", {})
    if not isinstance(jobs, dict):
        raise TypeError("Suite state jobs must be a mapping")

    main_records: dict[
        tuple[str, str, str, int, str],
        dict[str, Any],
    ] = {}
    ablation_records: dict[
        tuple[str, str, str, int, str],
        dict[str, Any],
    ] = {}
    offline_payloads: list[dict[str, Any]] = []

    for job_id, entry in jobs.items():
        if entry.get("status") not in {"completed", "reused"}:
            continue
        artifact = entry.get("reuse_source") or entry.get("artifact_path")
        if not artifact or not resolve_path(str(artifact)).is_file():
            continue

        if entry.get("kind") == "planning":
            payload = _json(str(artifact))
            successes = [
                bool(value) for value in payload["per_episode_success"]
            ]
            record = {
                "backend": str(payload["backend"]),
                "task": str(payload["task"]),
                "method": str(payload["method"]),
                "seed": int(payload["seeds"]["training"]),
                "domain": str(payload["domain"]),
                "success_count": int(payload["success_count"]),
                "episodes": int(payload["total_episodes"]),
                "success_rate": sum(successes) / len(successes),
                "per_episode_success": successes,
                "evaluation_instance_ids": list(
                    payload.get("evaluation_instance_ids", [])
                ),
                "source_path": str(resolve_path(str(artifact))),
            }
            key = (
                record["backend"],
                record["task"],
                record["method"],
                record["seed"],
                record["domain"],
            )
            target = (
                main_records
                if str(job_id).startswith("planning/main/")
                else ablation_records
            )
            if key in target:
                raise RuntimeError(f"Duplicate planning result: {key}")
            target[key] = record
        elif entry.get("kind") == "offline":
            offline_payloads.append(_json(str(artifact)))

    long_rows = [
        {
            key: value
            for key, value in record.items()
            if key not in {
                "per_episode_success",
                "evaluation_instance_ids",
            }
        }
        for _, record in sorted(main_records.items())
    ]
    _csv(
        output / "planning_results_long.csv",
        long_rows,
        [
            "backend",
            "task",
            "method",
            "seed",
            "domain",
            "success_count",
            "episodes",
            "success_rate",
            "source_path",
        ],
    )

    paired_rows: list[dict[str, Any]] = []
    identities = sorted(
        {
            (backend, task, method, seed)
            for backend, task, method, seed, _domain in main_records
        }
    )
    for backend, task, method, seed in identities:
        clean = main_records[(backend, task, method, seed, "clean")]
        ood = main_records[(backend, task, method, seed, "ood")]
        if (
            clean["evaluation_instance_ids"]
            != ood["evaluation_instance_ids"]
        ):
            raise RuntimeError(
                "Clean/OOD evaluation instances differ: "
                f"{(backend, task, method, seed)}"
            )
        clean_rate = float(clean["success_rate"])
        ood_rate = float(ood["success_rate"])
        paired_rows.append(
            {
                "backend": backend,
                "task": task,
                "method": method,
                "seed": seed,
                "clean_success_rate": clean_rate,
                "ood_success_rate": ood_rate,
                "absolute_drop": clean_rate - ood_rate,
                "relative_retention": (
                    ood_rate / clean_rate if clean_rate > 0.0 else None
                ),
                "episodes": clean["episodes"],
            }
        )

    _csv(
        output / "clean_ood_pairs.csv",
        paired_rows,
        [
            "backend",
            "task",
            "method",
            "seed",
            "clean_success_rate",
            "ood_success_rate",
            "absolute_drop",
            "relative_retention",
            "episodes",
        ],
    )

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in paired_rows:
        grouped[
            (row["backend"], row["task"], row["method"])
        ].append(row)

    markdown = [
        "# Clean-trained adapter robustness",
        "",
        "All trainable adapters are optimized from clean observations only. "
        "Each checkpoint is evaluated on paired clean and photometric-OOD "
        "instances using identical environment and CEM seeds.",
        "",
        "## Clean and OOD success",
        "",
        "| backend | task | method | Clean mean ± std | "
        "OOD mean ± std | drop mean ± std | retention |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for (backend, task, method), rows in sorted(grouped.items()):
        clean_mean, clean_std = _mean_std(
            [float(row["clean_success_rate"]) for row in rows]
        )
        ood_mean, ood_std = _mean_std(
            [float(row["ood_success_rate"]) for row in rows]
        )
        drop_mean, drop_std = _mean_std(
            [float(row["absolute_drop"]) for row in rows]
        )
        retention_values = [
            float(row["relative_retention"])
            for row in rows
            if row["relative_retention"] is not None
        ]
        retention = (
            statistics.mean(retention_values)
            if retention_values
            else float("nan")
        )
        markdown.append(
            f"| {backend} | {task} | {method} | "
            f"{clean_mean:.3f} ± {clean_std:.3f} | "
            f"{ood_mean:.3f} ± {ood_std:.3f} | "
            f"{drop_mean:.3f} ± {drop_std:.3f} | "
            f"{retention:.3f} |"
        )

    comparison_rows: list[dict[str, Any]] = []
    markdown.extend(
        [
            "",
            "## Paired HFRA comparisons",
            "",
            "| backend | task | domain | comparison | Δ success | 95% CI |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    backend_tasks = sorted(
        {(backend, task) for backend, task, _, _, _ in main_records}
    )
    for backend, task in backend_tasks:
        for domain in ("clean", "ood"):
            hfra = _records_by_seed(
                main_records,
                backend=backend,
                task=task,
                method="hfra",
                domain=domain,
            )
            for baseline in ("base", "lora"):
                baseline_records = _records_by_seed(
                    main_records,
                    backend=backend,
                    task=task,
                    method=baseline,
                    domain=domain,
                )
                if not hfra or not baseline_records:
                    continue
                delta, lower, upper = _paired_bootstrap(
                    hfra,
                    baseline_records,
                    key=(
                        backend,
                        task,
                        domain,
                        "hfra",
                        baseline,
                    ),
                )
                comparison_rows.append(
                    {
                        "backend": backend,
                        "task": task,
                        "domain": domain,
                        "left_method": "hfra",
                        "right_method": baseline,
                        "success_rate_difference": delta,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    }
                )
                markdown.append(
                    f"| {backend} | {task} | {domain} | "
                    f"HFRA−{baseline} | {delta:+.3f} | "
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
            "bootstrap_replicates",
        ],
    )

    ablation_rows: list[dict[str, Any]] = []
    ablation_identities = sorted(
        {
            (backend, task, method, seed)
            for backend, task, method, seed, _domain in ablation_records
        }
    )
    for backend, task, method, seed in ablation_identities:
        clean = ablation_records[(backend, task, method, seed, "clean")]
        ood = ablation_records[(backend, task, method, seed, "ood")]
        clean_rate = float(clean["success_rate"])
        ood_rate = float(ood["success_rate"])
        ablation_rows.append(
            {
                "backend": backend,
                "task": task,
                "method": method,
                "seed": seed,
                "clean_success_rate": clean_rate,
                "ood_success_rate": ood_rate,
                "absolute_drop": clean_rate - ood_rate,
                "relative_retention": (
                    ood_rate / clean_rate if clean_rate > 0.0 else None
                ),
            }
        )
    _csv(
        output / "ablation_results.csv",
        ablation_rows,
        [
            "backend",
            "task",
            "method",
            "seed",
            "clean_success_rate",
            "ood_success_rate",
            "absolute_drop",
            "relative_retention",
        ],
    )

    mse_rows: list[dict[str, Any]] = []
    for payload in offline_payloads:
        for domain, metrics in payload["domains"].items():
            mse_rows.append(
                {
                    "backend": payload["backend"],
                    "task": payload["task"],
                    "method": payload["method"],
                    "seed": payload.get("training_seed"),
                    "domain": domain,
                    **metrics,
                }
            )
    _csv(
        output / "mse_results.csv",
        mse_rows,
        [
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
        ],
    )

    _atomic_text(
        output / "main_results.md",
        "\n".join(markdown) + "\n",
    )


if __name__ == "__main__":
    main()
