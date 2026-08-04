from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from wm_adapter.utils.reproducibility import resolve_path


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


def _csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _rate(payload: dict[str, Any]) -> float:
    values = [bool(value) for value in payload["per_episode_success"]]
    return sum(values) / len(values)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = _arguments()
    state = _json(args.state)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = state.get("jobs", {})
    if not isinstance(jobs, dict):
        raise TypeError("Suite state jobs must be a mapping")

    main_payloads: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    ablation_payloads: list[dict[str, Any]] = []
    offline_payloads: list[dict[str, Any]] = []
    efficiency: list[dict[str, Any]] = []
    for job_id, entry in jobs.items():
        if entry.get("status") not in {"completed", "reused"}:
            continue
        artifact = entry.get("reuse_source") or entry.get("artifact_path")
        if not artifact or not resolve_path(str(artifact)).is_file():
            continue
        if entry.get("kind") == "planning":
            payload = _json(str(artifact))
            record = {
                "backend": payload["backend"],
                "task": payload["task"],
                "method": payload["method"],
                "seed": int(payload["seeds"]["training"]),
                "domain": payload["domain"],
                "success_count": int(payload["success_count"]),
                "episodes": int(payload["total_episodes"]),
                "success_rate": _rate(payload),
                "source_path": str(resolve_path(str(artifact))),
            }
            if str(job_id).startswith("planning/main/"):
                key = (
                    record["backend"],
                    record["task"],
                    record["method"],
                    record["seed"],
                    record["domain"],
                )
                main_payloads[key] = record
            else:
                ablation_payloads.append(record | {"variant": "core_only"})
            elapsed = float(payload.get("elapsed_seconds", payload.get("runtime_seconds", 0.0)))
            available = int(payload["total_episodes"])
            efficiency.append(
                record
                | {
                    "source_elapsed_seconds": elapsed,
                    "source_available_episodes": available,
                    "used_episodes": available,
                    "elapsed_seconds_per_episode": elapsed / available,
                    "peak_cuda_memory_bytes": int(payload.get("peak_cuda_memory_bytes", 0)),
                    "parameter_count": int(payload.get("method_parameter_count", 0)),
                }
            )
        elif entry.get("kind") == "offline":
            offline_payloads.append(_json(str(artifact)))

    main_rows: list[dict[str, Any]] = []
    keys = sorted({key[:4] for key in main_payloads})
    for backend, task, method, seed in keys:
        clean = main_payloads.get((backend, task, method, seed, "clean"))
        ood = main_payloads.get((backend, task, method, seed, "ood"))
        if clean is None or ood is None:
            raise RuntimeError(
                f"Main planning pair is incomplete: {(backend, task, method, seed)}"
            )
        main_rows.append(
            {
                "backend": backend,
                "task": task,
                "method": method,
                "seed": seed,
                "clean_success_count": clean["success_count"],
                "clean_n": clean["episodes"],
                "clean_success_rate": clean["success_rate"],
                "ood_success_count": ood["success_count"],
                "ood_n": ood["episodes"],
                "ood_success_rate": ood["success_rate"],
                "clean_to_ood_drop": clean["success_rate"] - ood["success_rate"],
            }
        )
    main_fields = [
        "backend", "task", "method", "seed",
        "clean_success_count", "clean_n", "clean_success_rate",
        "ood_success_count", "ood_n", "ood_success_rate", "clean_to_ood_drop",
    ]
    _csv(
        output / "jepa_wm_main_results.csv",
        [row for row in main_rows if row["backend"] == "jepa_wm_droid"],
        main_fields,
    )
    _csv(
        output / "dino_wm_main_results.csv",
        [row for row in main_rows if row["backend"] == "dino_wm_droid"],
        main_fields,
    )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        grouped[(row["backend"], row["task"], row["method"])].append(row)
    markdown = [
        "# Cross-backend adapter results",
        "",
        "Backends are reported separately; raw latent MSE is never averaged across latent spaces.",
        "",
        "| backend | task | method | Clean mean ± std | OOD mean ± std | drop |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key, values in sorted(grouped.items()):
        clean_mean, clean_std = _mean_std([float(row["clean_success_rate"]) for row in values])
        ood_mean, ood_std = _mean_std([float(row["ood_success_rate"]) for row in values])
        markdown.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{clean_mean:.3f} ± {clean_std:.3f} | {ood_mean:.3f} ± {ood_std:.3f} | "
            f"{clean_mean - ood_mean:.3f} |"
        )
    markdown.extend(
        [
            "",
            "## HFRA improvements within each backend",
            "",
            "Improvements are computed within a backend/task/domain; no backend "
            "scores are pooled.",
            "",
            "| backend | task | domain | HFRA−Base | HFRA−LoRA |",
            "|---|---|---|---:|---:|",
        ]
    )
    aggregate_rates: dict[tuple[str, str, str, str], float] = {}
    for (backend, task, method), values in grouped.items():
        aggregate_rates[(backend, task, method, "clean")] = statistics.mean(
            float(row["clean_success_rate"]) for row in values
        )
        aggregate_rates[(backend, task, method, "ood")] = statistics.mean(
            float(row["ood_success_rate"]) for row in values
        )
    for backend, task, method in sorted(grouped):
        if method != "hfra":
            continue
        for domain in ("clean", "ood"):
            hfra = aggregate_rates[(backend, task, "hfra", domain)]
            base = aggregate_rates[(backend, task, "base", domain)]
            lora = aggregate_rates[(backend, task, "lora", domain)]
            markdown.append(
                f"| {backend} | {task} | {domain} | "
                f"{hfra - base:+.3f} | {hfra - lora:+.3f} |"
            )
    _atomic_text(output / "main_results.md", "\n".join(markdown) + "\n")

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
    mse_fields = [
        "backend", "task", "method", "seed", "domain",
        "h1_autoregressive_latent_mse", "h2_autoregressive_latent_mse",
        "h3_autoregressive_latent_mse", "future_mean_mse", "terminal_mse",
        "unified_6frame_trajectory_mse",
    ]
    _csv(output / "mse_results.csv", mse_rows, mse_fields)
    mse_md = [
        "# Offline autoregressive MSE",
        "",
        "Raw values are only comparable within a backend latent space. Base is "
        "evaluated once; learned methods report mean ± sample standard deviation "
        "over the three training seeds.",
        "",
        "| backend | task | method | domain | H1 | H2 | H3 | future mean | trajectory |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    metric_names = [
        "h1_autoregressive_latent_mse",
        "h2_autoregressive_latent_mse",
        "h3_autoregressive_latent_mse",
        "future_mean_mse",
        "unified_6frame_trajectory_mse",
    ]
    grouped_mse: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mse_rows:
        grouped_mse[(row["backend"], row["task"], row["method"], row["domain"])].append(row)
    for key, rows in sorted(grouped_mse.items()):
        rendered: list[str] = []
        for metric in metric_names:
            mean, std = _mean_std([float(row[metric]) for row in rows])
            rendered.append(
                f"{mean:.6g}" if key[2] == "base" else f"{mean:.6g} ± {std:.3g}"
            )
        mse_md.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | "
            + " | ".join(rendered)
            + " |"
        )
    mse_summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped_mse.items()):
        backend, task, method, domain = key
        base_rows = grouped_mse.get((backend, task, "base", domain), [])
        if len(base_rows) != 1:
            raise RuntimeError(
                "Offline MSE aggregation requires one deterministic Base row: "
                f"backend={backend}, task={task}, domain={domain}, found={len(base_rows)}"
            )
        for metric in metric_names:
            mean, std = _mean_std([float(row[metric]) for row in rows])
            base_value = float(base_rows[0][metric])
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
                    "relative_improvement_over_base": relative,
                }
            )
    _csv(
        output / "mse_results_summary.csv",
        mse_summary_rows,
        [
            "backend", "task", "method", "domain", "metric", "seed_count",
            "mean", "std", "base_value", "relative_improvement_over_base",
        ],
    )
    _atomic_text(output / "mse_results.md", "\n".join(mse_md) + "\n")

    # Add full HFRA rows by reference to the main OOD results; no extra full
    # ablation planning jobs are created.
    for row in main_rows:
        if (
            row["backend"] == "jepa_wm_droid"
            and row["task"] in {"robocasa_place", "libero_goal_0"}
            and row["method"] == "hfra"
        ):
            ablation_payloads.append(
                {
                    "backend": row["backend"],
                    "task": row["task"],
                    "method": "hfra",
                    "variant": "full",
                    "seed": row["seed"],
                    "domain": "ood",
                    "success_count": row["ood_success_count"],
                    "episodes": row["ood_n"],
                    "success_rate": row["ood_success_rate"],
                    "source_path": "main result reuse",
                }
            )
    ablation_fields = [
        "backend", "task", "variant", "method", "seed", "domain",
        "success_count", "episodes", "success_rate", "source_path",
    ]
    _csv(output / "ablation_results.csv", ablation_payloads, ablation_fields)
    grouped_ablation: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in ablation_payloads:
        grouped_ablation[(row["backend"], row["task"], row["variant"])].append(
            float(row["success_rate"])
        )
    ablation_markdown = [
        "# HFRA closed-loop ablation",
        "",
        "Full HFRA rows reuse the corresponding main OOD result.",
        "",
        "| backend | task | variant | OOD success mean ± std | seeds |",
        "|---|---|---|---:|---:|",
    ]
    for key, values in sorted(grouped_ablation.items()):
        mean, std = _mean_std(values)
        ablation_markdown.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{mean:.3f} ± {std:.3f} | {len(values)} |"
        )
    _atomic_text(
        output / "ablation_results.md",
        "\n".join(ablation_markdown) + "\n",
    )
    _csv(
        output / "efficiency.csv",
        efficiency,
        [
            "backend", "task", "method", "seed", "domain",
            "source_elapsed_seconds", "source_available_episodes", "used_episodes",
            "elapsed_seconds_per_episode", "peak_cuda_memory_bytes",
            "parameter_count", "source_path",
        ],
    )


if __name__ == "__main__":
    main()
