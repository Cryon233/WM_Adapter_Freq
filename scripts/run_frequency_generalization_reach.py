from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiment/cross_backend_adapter/robocasa_reach.yaml"
SEEDS = (42, 7, 2026)
MAIN_METHODS = ("base", "lora", "token_mlp", "hfra")
ABLATION_METHODS = ("hfra_core_only",)
UNSEEN_FAMILIES = (
    "gaussian_blur",
    "gaussian_noise",
    "dct_compression",
)


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    log_path: Path
    artifact_path: Path
    artifact_kind: str
    required_episodes: int | None = None
    family: str | None = None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the focused JEPA-WM RoboCasa Reach frequency-"
            "generalization experiment"
        )
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--gpus",
        default=os.environ.get("GPUS", "0,1,2,3"),
    )
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checkpoint(method: str, seed: int) -> Path:
    return ROOT / (
        "checkpoints/cross_backend_adapter_v1/jepa_wm_droid/"
        f"robocasa_reach/{method}/seed_{seed}_final.pt"
    )


def manifest(seed: int) -> Path:
    return ROOT / (
        "outputs/cross_backend_adapter_v1/manifests/evaluation/"
        f"robocasa_reach/seed_{seed}.json"
    )


def cache() -> Path:
    return ROOT / (
        "storage/feature_cache/cross_backend_adapter_v1/"
        "jepa_wm_droid/robocasa_reach/cache.h5"
    )


def result_path(family: str, method: str, seed: int) -> Path:
    return ROOT / (
        "outputs/frequency_generalization_reach/robocasa_reach/"
        f"{family}/seed_{seed}/{method}/results.json"
    )


def valid_checkpoint(path: Path, method: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        import torch

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        return (
            str(payload.get("method_name")) == method
            and int(payload.get("training_seed", -1)) == seed
            and int(payload.get("completed_optimizer_steps", -1)) == 2000
        )
    except Exception:
        return False


def valid_result(
    path: Path, *, family: str, episodes: int
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_family = str(
            payload.get("evaluation_family")
            or payload.get("appearance_metadata", {}).get("family", "")
        )
        return (
            int(payload.get("total_episodes", -1)) == episodes
            and len(payload.get("per_episode_success", [])) == episodes
            and actual_family == family
        )
    except Exception:
        return False


def archive_stale(path: Path) -> None:
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = path.with_name(f"{path.name}.stale-{stamp}")
    path.rename(destination)


def training_jobs(log_root: Path) -> list[Job]:
    jobs: list[Job] = []
    for method in (*MAIN_METHODS, *ABLATION_METHODS):
        if method == "base":
            continue
        for seed in SEEDS:
            artifact = checkpoint(method, seed)
            if valid_checkpoint(artifact, method, seed):
                continue
            if artifact.exists():
                archive_stale(artifact)
            command = (
                sys.executable,
                "scripts/train_adapter.py",
                "--config",
                CONFIG,
                f"method={method}",
                f"training.seed={seed}",
                f"evaluation.eval_seed={seed}",
                f"paths.feature_cache={cache()}",
                f"paths.evaluation_manifest={manifest(seed)}",
                f"paths.method_checkpoint={artifact}",
            )
            jobs.append(
                Job(
                    name=f"train-{method}-seed-{seed}",
                    command=command,
                    log_path=log_root / f"train-{method}-seed-{seed}.log",
                    artifact_path=artifact,
                    artifact_kind="checkpoint",
                )
            )
    return jobs


def planning_jobs(
    log_root: Path, *, episodes: int, include_clean: bool
) -> list[Job]:
    families = list(UNSEEN_FAMILIES)
    if include_clean:
        families.insert(0, "identity")
    jobs: list[Job] = []
    for family in families:
        domain = "clean" if family == "identity" else "ood"
        for method in (*MAIN_METHODS, *ABLATION_METHODS):
            for seed in SEEDS:
                artifact = result_path(family, method, seed)
                if valid_result(
                    artifact, family=family, episodes=episodes
                ):
                    continue
                if artifact.exists():
                    artifact.unlink()
                run_directory = artifact.parent
                command = [
                    sys.executable,
                    "scripts/plan.py",
                    "--config",
                    CONFIG,
                    f"method={method}",
                    f"domain={domain}",
                    f"training.seed={seed}",
                    f"evaluation.eval_seed={seed}",
                    f"evaluation.num_episodes={episodes}",
                    f"paths.feature_cache={cache()}",
                    f"paths.evaluation_manifest={manifest(seed)}",
                    f"appearance.evaluation_family={family}",
                    "appearance.severity=1.0",
                    f"output.run_directory={run_directory}",
                    "suite.family=frequency_generalization",
                    (
                        "suite.variant=core_only"
                        if method == "hfra_core_only"
                        else "suite.variant=full"
                    ),
                ]
                if method != "base":
                    command.append(
                        f"paths.method_checkpoint={checkpoint(method, seed)}"
                    )
                jobs.append(
                    Job(
                        name=(
                            f"plan-{family}-{method}-seed-{seed}"
                        ),
                        command=tuple(command),
                        log_path=(
                            log_root
                            / f"plan-{family}-{method}-seed-{seed}.log"
                        ),
                        artifact_path=artifact,
                        artifact_kind="planning",
                        required_episodes=episodes,
                        family=family,
                    )
                )
    return jobs


def run_phase(jobs: list[Job], gpu_ids: list[int]) -> None:
    pending = list(jobs)
    running: dict[int, tuple[Job, subprocess.Popen[Any], Any]] = {}
    while pending or running:
        for gpu in gpu_ids:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = job.log_path.open("w", encoding="utf-8")
            handle.write("COMMAND " + shlex.join(job.command) + "\n")
            handle.flush()
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                job.command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu] = (job, process, handle)
            print(f"START gpu={gpu} {job.name}", flush=True)
        time.sleep(2.0)
        for gpu, (job, process, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[gpu]
            if code != 0:
                for _, other, other_handle in running.values():
                    other.terminate()
                    other_handle.close()
                raise RuntimeError(
                    f"Job failed: {job.name}; log={job.log_path}"
                )
            if job.artifact_kind == "checkpoint":
                method = job.name.split("-seed-")[0].removeprefix("train-")
                seed = int(job.name.rsplit("-", 1)[-1])
                valid = valid_checkpoint(job.artifact_path, method, seed)
            else:
                valid = valid_result(
                    job.artifact_path,
                    family=str(job.family),
                    episodes=int(job.required_episodes or 0),
                )
            if not valid:
                raise RuntimeError(
                    f"Job artifact failed validation: {job.name}; "
                    f"artifact={job.artifact_path}"
                )
            print(f"DONE  gpu={gpu} {job.name}", flush=True)


def write_summary(*, episodes: int, include_clean: bool) -> Path:
    families = list(UNSEEN_FAMILIES)
    if include_clean:
        families.insert(0, "identity")
    summary: dict[str, Any] = {
        "episodes_per_job": episodes,
        "seeds": list(SEEDS),
        "families": families,
        "methods": [*MAIN_METHODS, *ABLATION_METHODS],
        "results": {},
    }
    for family in families:
        family_values: dict[str, Any] = {}
        for method in (*MAIN_METHODS, *ABLATION_METHODS):
            successes = 0
            total = 0
            per_seed: dict[str, Any] = {}
            for seed in SEEDS:
                payload = json.loads(
                    result_path(family, method, seed).read_text(
                        encoding="utf-8"
                    )
                )
                count = int(payload["success_count"])
                number = int(payload["total_episodes"])
                successes += count
                total += number
                per_seed[str(seed)] = {
                    "successes": count,
                    "episodes": number,
                    "success_rate": count / number,
                }
            family_values[method] = {
                "successes": successes,
                "episodes": total,
                "success_rate": successes / total,
                "per_seed": per_seed,
            }
        summary["results"][family] = family_values
    destination = (
        ROOT
        / "outputs/frequency_generalization_reach/summary.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    args = arguments()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    gpu_ids = [
        int(value) for value in str(args.gpus).split(",") if value.strip()
    ]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Invalid GPU list: {args.gpus!r}")
    required = [cache(), *(manifest(seed) for seed in SEEDS)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required cache/manifests are missing: " + ", ".join(missing)
        )
    log_root = ROOT / "logs/frequency_generalization_reach"
    train = training_jobs(log_root)
    plan = planning_jobs(
        log_root,
        episodes=args.episodes,
        include_clean=args.include_clean,
    )
    print(
        f"training_jobs={len(train)} planning_jobs={len(plan)} "
        f"episodes={args.episodes} gpus={gpu_ids}"
    )
    if args.dry_run:
        for job in (*train, *plan):
            print(job.name, shlex.join(job.command))
        return
    run_phase(train, gpu_ids)
    run_phase(plan, gpu_ids)
    summary = write_summary(
        episodes=args.episodes,
        include_clean=args.include_clean,
    )
    print(f"SUMMARY {summary}")


if __name__ == "__main__":
    main()
