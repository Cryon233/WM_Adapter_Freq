from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import OmegaConf

from wm_adapter.experiments.paper_suite import atomic_write_json, validate_planning_result
from wm_adapter.utils.checkpoints import sha256_file
from wm_adapter.utils.reproducibility import resolve_path


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError(f"Wilson interval requires positive sample count, received {total}")
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_difference(
    left: list[bool], right: list[bool], samples: int, seed: int = 42
) -> tuple[float, float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("Paired bootstrap requires equally sized, non-empty outcomes")
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indices = generator.integers(0, len(difference), size=(stop - start, len(difference)))
        estimates[start:stop] = difference[indices].mean(axis=1)
    return float(difference.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _mcnemar_exact(left: list[bool], right: list[bool]) -> tuple[int, int, float]:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    probability = min(1.0, 2.0 * tail / (2**discordant))
    return left_only, right_only, probability


def _holm(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No records._\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for key in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def _load_planning(path: Path, episodes: int, metadata: dict[str, Any]) -> dict[str, Any]:
    validation = validate_planning_result(path, episodes)
    payload = json.loads(path.read_text(encoding="utf-8"))
    outcomes = [bool(value) for value in payload["per_episode_success"][:episodes]]
    identities = payload.get("environment_seeds", [])[:episodes]
    if len(identities) != episodes:
        raise RuntimeError(f"Planning result lacks paired episode identities: {path}")
    lower, upper = _wilson(sum(outcomes), episodes)
    return {
        **metadata,
        "episodes": episodes,
        "success_count": sum(outcomes),
        "success_rate": sum(outcomes) / episodes,
        "wilson_low": lower,
        "wilson_high": upper,
        "outcomes": outcomes,
        "identities": [int(value) for value in identities],
        "source_path": str(path),
        "source_sha256": validation["sha256"],
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
        "peak_cuda_memory_bytes": int(payload.get("peak_cuda_memory_bytes", 0)),
        "parameter_count": int(payload.get("method_parameter_count", 0)),
    }


def _public_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key not in {"outcomes", "identities"}} for row in rows]


def _latex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def _paired_statistics(rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    indexed = {
        (row.get("family"), row.get("task"), row.get("domain"), row.get("seed"), row.get("method")): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    for key, candidate in indexed.items():
        family, task, domain, seed, method = key
        if method == "base":
            continue
        base = indexed.get((family, task, domain, seed, "base"))
        if base is None:
            continue
        if candidate["identities"] != base["identities"]:
            raise RuntimeError(
                "Refusing paired statistics because episode identities differ: "
                f"family={family}, task={task}, domain={domain}, seed={seed}, method={method}"
            )
        difference, low, high = _bootstrap_difference(
            candidate["outcomes"], base["outcomes"], bootstrap_samples
        )
        candidate_only, base_only, p_value = _mcnemar_exact(
            candidate["outcomes"], base["outcomes"]
        )
        comparisons.append(
            {
                "family": family,
                "task": task,
                "domain": domain,
                "seed": seed,
                "method": method,
                "reference": "base",
                "success_rate_difference": difference,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "candidate_only_successes": candidate_only,
                "base_only_successes": base_only,
                "mcnemar_exact_p": p_value,
            }
        )
    adjusted = _holm([row["mcnemar_exact_p"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = value
    return comparisons


def _formal_rows(suite: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    main: list[dict[str, Any]] = []
    for task in suite.tasks:
        for method in suite.methods:
            for domain in suite.domains:
                path = resolve_path(
                    f"outputs/jepa_wm_droid/robocasa/{suite.protocol}/{task}/seed_{suite.main.seed}/{method}/{domain}/results.json"
                )
                main.append(
                    _load_planning(
                        path,
                        int(suite.main.episodes),
                        {"family": "main", "task": str(task), "method": str(method), "domain": str(domain), "seed": int(suite.main.seed)},
                    )
                )
    multiseed: list[dict[str, Any]] = []
    for task in suite.multiseed.tasks:
        for method in suite.multiseed.methods:
            for domain in suite.multiseed.domains:
                for seed in suite.multiseed.seeds:
                    if int(seed) == int(suite.main.seed):
                        path = resolve_path(
                            f"outputs/jepa_wm_droid/robocasa/{suite.protocol}/{task}/seed_{seed}/{method}/{domain}/results.json"
                        )
                    else:
                        path = resolve_path(
                            f"outputs/paper_suite/{suite.protocol}/multiseed/{task}/seed_{seed}/{method}/{domain}/results.json"
                        )
                    multiseed.append(
                        _load_planning(path, int(suite.multiseed.episodes), {"family": "multiseed", "task": str(task), "method": str(method), "domain": str(domain), "seed": int(seed)})
                    )
    severity: list[dict[str, Any]] = []
    for task in suite.severity.tasks:
        for method in suite.severity.methods:
            for value in suite.severity.values:
                if float(value) == 1.0:
                    path = resolve_path(
                        f"outputs/jepa_wm_droid/robocasa/{suite.protocol}/{task}/seed_{suite.severity.seed}/{method}/ood/results.json"
                    )
                else:
                    name = str(float(value)).replace(".", "p")
                    path = resolve_path(
                        f"outputs/paper_suite/{suite.protocol}/severity/{task}/seed_{suite.severity.seed}/{method}/ood_severity_{name}/results.json"
                    )
                severity.append(
                    _load_planning(path, int(suite.severity.episodes), {"family": "severity", "task": str(task), "method": str(method), "domain": "ood", "severity": float(value), "seed": int(suite.severity.seed)})
                )
    ablations: list[dict[str, Any]] = []
    for variant in suite.closed_loop_ablations:
        for task in suite.ablation_tasks:
            if str(variant) == "full":
                path = resolve_path(
                    f"outputs/jepa_wm_droid/robocasa/{suite.protocol}/{task}/seed_42/dct_adapter/ood/results.json"
                )
            else:
                path = resolve_path(
                    f"outputs/paper_suite/{suite.protocol}/ablations/{task}/seed_42/{variant}/ood/results.json"
                )
            ablations.append(
                _load_planning(path, int(suite.ablation_episodes), {"family": "ablation", "task": str(task), "method": "dct_adapter", "variant": str(variant), "domain": "ood", "seed": 42})
            )
    return main, multiseed, severity, ablations


def _offline_rows(suite: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_identities: list[list[int]] | None = None
    paths: list[tuple[str, Path]] = [
        (method, resolve_path(f"outputs/paper_suite/offline/main/{method}/metrics.json"))
        for method in ("base", "dct_adapter", "token_mlp", "lora")
    ]
    for variant in suite.dct_ablations:
        source = (
            resolve_path("outputs/paper_suite/offline/main/dct_adapter/metrics.json")
            if str(variant) in {"full", "rank8"}
            else resolve_path(f"outputs/paper_suite/offline/ablations/{variant}/metrics.json")
        )
        paths.append((f"dct_{variant}", source))
    for label, path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identities = payload.get("window_identities")
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise RuntimeError(
                f"Offline window identities differ for {label}: {path}"
            )
        for domain, values in payload["domains"].items():
            rows.append(
                {
                    "method_or_variant": label,
                    "domain": domain,
                    "window_count": int(payload["window_count"]),
                    **values,
                    "source_path": str(path),
                    "source_sha256": sha256_file(path),
                }
            )
    return rows


def _self_test_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    planning: list[dict[str, Any]] = []
    for method in ("base", "dct_adapter", "token_mlp", "lora"):
        for domain in ("clean", "ood"):
            path = root / "planning" / method / domain / "results.json"
            planning.append(
                _load_planning(path, 1, {"family": "self_test", "task": "place", "method": method, "domain": domain, "seed": 42})
            )
    offline: list[dict[str, Any]] = []
    reference_identities: list[list[int]] | None = None
    for method in ("base", "dct_adapter", "token_mlp", "lora"):
        path = root / "offline" / method / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        identities = payload.get("window_identities")
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise RuntimeError(
                f"Self-test offline window identities differ for {method}: {path}"
            )
        for domain, values in payload["domains"].items():
            offline.append({"method_or_variant": method, "domain": domain, "window_count": payload["window_count"], **values, "source_path": str(path), "source_sha256": sha256_file(path)})
    return planning, offline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", default="configs/experiment/icra2027_suite.yaml")
    parser.add_argument("--self-test-root")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    output = resolve_path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    if arguments.self_test_root:
        main_rows, offline_rows = _self_test_rows(resolve_path(arguments.self_test_root))
        multiseed_rows: list[dict[str, Any]] = []
        severity_rows: list[dict[str, Any]] = []
        ablation_rows: list[dict[str, Any]] = []
        bootstrap_samples = 10000
    else:
        suite = OmegaConf.load(resolve_path(arguments.suite_config))
        main_rows, multiseed_rows, severity_rows, ablation_rows = _formal_rows(suite)
        offline_rows = _offline_rows(suite)
        bootstrap_samples = int(suite.offline.bootstrap_samples)
    main_public = _public_rows(main_rows)
    main_columns = ["family", "task", "method", "domain", "seed", "episodes", "success_count", "success_rate", "wilson_low", "wilson_high", "source_path", "source_sha256"]
    _write_csv(output / "main_results.csv", main_public, main_columns)
    (output / "main_results.md").write_text(_markdown(main_public, main_columns), encoding="utf-8")
    latex_lines = ["\\begin{tabular}{llllrr}", "Task & Method & Domain & Seed & Success & Rate \\\\", "\\hline"]
    for row in main_public:
        latex_lines.append(
            f"{_latex(row['task'])} & {_latex(row['method'])} & "
            f"{_latex(row['domain'])} & {row['seed']} & "
            f"{row['success_count']}/{row['episodes']} & "
            f"{row['success_rate']:.3f} \\\\"
        )
    latex_lines.append("\\end{tabular}")
    (output / "main_results.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    simple_columns = ["family", "task", "method", "domain", "seed", "severity", "variant", "episodes", "success_count", "success_rate", "source_path", "source_sha256"]
    _write_csv(output / "multiseed.csv", _public_rows(multiseed_rows), simple_columns)
    _write_csv(output / "severity.csv", _public_rows(severity_rows), simple_columns)
    _write_csv(output / "ablations.csv", _public_rows(ablation_rows), simple_columns)
    offline_columns = sorted({key for row in offline_rows for key in row})
    _write_csv(output / "offline_metrics.csv", offline_rows, offline_columns)
    efficiency_rows = [
        {key: row[key] for key in ("task", "method", "domain", "elapsed_seconds", "peak_cuda_memory_bytes", "parameter_count", "source_path")}
        for row in main_public
    ]
    _write_csv(output / "efficiency.csv", efficiency_rows, ["task", "method", "domain", "elapsed_seconds", "peak_cuda_memory_bytes", "parameter_count", "source_path"])
    paired = _paired_statistics(main_rows + multiseed_rows, bootstrap_samples)
    seed_summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in multiseed_rows:
        grouped.setdefault((row["task"], row["method"], row["domain"]), []).append(row["success_rate"])
    for (task, method, domain), values in grouped.items():
        seed_summary.append({"task": task, "method": method, "domain": domain, "seed_mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "seed_count": len(values)})
    statistics = {"bootstrap_samples": bootstrap_samples, "paired_comparisons": paired, "multiseed_summary": seed_summary}
    atomic_write_json(output / "statistics.json", statistics)
    summary = ["# ICRA 2027 paper suite summary", "", f"Main planning conditions: {len(main_rows)}", f"Offline metric rows: {len(offline_rows)}", f"Paired comparisons: {len(paired)}", "", "All reused records retain source paths and SHA256 fingerprints.", ""]
    (output / "paper_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Paper-suite analysis written: {output}")


if __name__ == "__main__":
    main()
