from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from wm_adapter.adapters.base import BaseMethod, PEFTMethod
from wm_adapter.adapters.factory import build_method
from wm_adapter.adapters.lora import LastBlockAttentionLoRA
from wm_adapter.appearance.composed_photometric import ComposedPhotometricShift
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.data.robocasa_windows import (
    RoboCasaWindowDataset,
    build_robocasa_dataset,
    select_episode_balanced_windows,
    split_episode_indices,
)
from wm_adapter.utils.checkpoints import load_method_checkpoint, sha256_file
from wm_adapter.utils.reproducibility import (
    load_experiment_config,
    resolve_path,
    seed_everything,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _backend(cfg: Any) -> JEPAWMDroidBackend:
    return JEPAWMDroidBackend(
        third_party_root=cfg.model.third_party_root,
        jepa_checkpoint=cfg.model.jepa_checkpoint,
        dinov3_checkpoint=cfg.model.dinov3_checkpoint,
        official_planning_config=cfg.model.official_planning_config,
        device=cfg.device,
        planning_tag=cfg.model.get("planning_tag"),
        planning_subtask=cfg.model.get("planning_subtask"),
    )


def _load_method(cfg: Any, backend: JEPAWMDroidBackend) -> tuple[PEFTMethod, str | None]:
    method = build_method(str(cfg.method), backend, cfg.method_config).to(backend.device)
    if method.method_name == "base":
        return method, None
    checkpoint_path = resolve_path(cfg.paths.method_checkpoint)
    checkpoint = load_method_checkpoint(checkpoint_path)
    expected_appearance = ComposedPhotometricShift.metadata(
        float(cfg.appearance.get("training_severity", cfg.appearance.severity)),
        int(cfg.appearance.training_seed),
    )
    expected = {
        "method_name": method.method_name,
        "method_config": method.config_dict(),
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "upstream_commits": backend.upstream_commits,
        "appearance_metadata": expected_appearance,
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Offline method checkpoint does not match the configured run: {mismatches}"
        )
    method.load_method_checkpoint(checkpoint["peft_state_dict"])
    return method.eval(), sha256_file(checkpoint_path)


def _sample_mse(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise RuntimeError(
            f"Offline latent shape mismatch: predicted={tuple(predicted.shape)}, "
            f"target={tuple(target.shape)}"
        )
    return (predicted.float() - target.float()).square().mean(
        dim=tuple(range(1, predicted.ndim))
    )


@contextmanager
def _frozen_base_projection(
    backend: JEPAWMDroidBackend, method: PEFTMethod
) -> Any:
    if not isinstance(method, LastBlockAttentionLoRA):
        yield
        return
    if method.qkv_lora is None:
        raise RuntimeError("LoRA is not attached to the DINOv3 last-block attention")
    attention = backend.last_block.attn
    if attention.qkv is not method.qkv_lora:
        raise RuntimeError("Unexpected DINOv3 QKV module while encoding the frozen clean target")
    attention.qkv = method.qkv_lora.base_projection
    try:
        yield
    finally:
        attention.qkv = method.qkv_lora


def _rollout_errors(
    backend: JEPAWMDroidBackend,
    context: Tensor,
    target: Tensor,
    actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    action_suffix = rearrange(actions[:, 2:5], "b t a -> t b a")
    predicted = backend.official_model.unroll(context, act_suffix=action_suffix)
    predicted_future = predicted[-3:]
    target_future = rearrange(target, "b t ... -> t b ...")
    per_step = torch.stack(
        [_sample_mse(predicted_future[index], target_future[index]) for index in range(3)],
        dim=1,
    )
    return per_step[:, 0], per_step[:, :2].mean(dim=1), per_step.mean(dim=1)


def main() -> None:
    cfg = load_experiment_config()
    seed_everything(int(cfg.offline.get("seed", cfg.evaluation.eval_seed)))
    backend = _backend(cfg)
    method, method_checkpoint_sha256 = _load_method(cfg, backend)
    source = build_robocasa_dataset(
        jepa_wms_root=backend.jepa_repo,
        dataset_root=cfg.paths.dataset_root,
        hdf5_path=cfg.paths.robocasa_hdf5,
        task_name=str(cfg.data.task_name),
        camera_view=str(cfg.data.camera_view),
        output_environment_info=False,
        transform=None,
    )
    _, evaluation_episodes = split_episode_indices(
        len(source), float(cfg.data.train_fraction), int(cfg.data.split_seed)
    )
    num_frames = int(cfg.offline.num_frames)
    if num_frames != 6:
        raise ValueError(f"Offline one/two/three-step evaluation requires num_frames=6, found {num_frames}")
    candidates = RoboCasaWindowDataset.all_candidates(
        source, num_frames, int(cfg.data.frameskip)
    )
    selected = select_episode_balanced_windows(
        candidates,
        evaluation_episodes,
        int(cfg.offline.num_windows),
        int(cfg.offline.get("seed", cfg.evaluation.eval_seed)),
    )
    if len(selected) != int(cfg.offline.num_windows):
        raise RuntimeError(
            f"Offline evaluation requested {cfg.offline.num_windows} held-out windows, "
            f"but selected {len(selected)}"
        )
    windows = RoboCasaWindowDataset(
        source,
        selected,
        num_frames=num_frames,
        frameskip=int(cfg.data.frameskip),
        appearance_seed=int(cfg.appearance.seed),
        appearance_severity=float(cfg.appearance.severity),
    )
    loader = DataLoader(
        windows,
        batch_size=int(cfg.offline.batch_size),
        shuffle=False,
        num_workers=int(cfg.offline.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.offline.num_workers) > 0,
    )
    domains = [str(cfg.domain)] if str(cfg.domain) in {"clean", "ood"} else ["clean", "ood"]
    output = resolve_path(cfg.offline.output_directory)
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Offline results will not be overwritten: {output / 'metrics.json'}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, float]] = {domain: {} for domain in domains}
    counts = {domain: 0 for domain in domains}
    base_method = BaseMethod().to(backend.device)
    precision = str(cfg.planning.inference_precision)
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if backend.device.type == "cuda" and precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode():
        for batch in tqdm(loader, desc="offline held-out windows"):
            clean = batch["clean_images"]
            batch_size = clean.shape[0]
            actions = batch["actions"].to(backend.device, non_blocking=True).float()
            with _frozen_base_projection(backend, method), autocast():
                clean_latent = backend.encode_images(
                    clean, base_method, batch_size, num_frames
                )
                clean_target = backend.planning_latents(clean_latent[:, 3:6])
            for domain in domains:
                images = clean if domain == "clean" else batch["ood_images"]
                with autocast():
                    adapted_context = backend.encode_images(
                        images[:, :3], method, batch_size, 3
                    )
                    canonical = _sample_mse(adapted_context, clean_latent[:, :3])
                    context = backend.planning_latents(adapted_context)
                    one, two, three = _rollout_errors(backend, context, clean_target, actions)
                    shuffled_actions = actions.flip(dims=(1,))
                    _, _, shuffled = _rollout_errors(
                        backend, context, clean_target, shuffled_actions
                    )
                    _, _, zero = _rollout_errors(
                        backend, context, clean_target, torch.zeros_like(actions)
                    )
                tensors = {
                    "canonical_mse": canonical,
                    "one_step_mse": one,
                    "two_step_mse": two,
                    "three_step_mse": three,
                    "shuffled_action_mse": shuffled,
                    "zero_action_mse": zero,
                    "action_shuffle_gap": shuffled - three,
                    "zero_action_gap": zero - three,
                }
                for key, values in tensors.items():
                    totals[domain][key] = totals[domain].get(key, 0.0) + float(values.sum().cpu())
                for index in range(batch_size):
                    rows.append(
                        {
                            "domain": domain,
                            "episode_id": int(batch["episode_id"][index]),
                            "window_id": int(batch["window_id"][index]),
                            **{key: float(value[index].cpu()) for key, value in tensors.items()},
                        }
                    )
                counts[domain] += batch_size
            print(f"OFFLINE_PROGRESS completed={len(rows)} total={len(windows) * len(domains)}")
    domain_metrics = {
        domain: {key: value / counts[domain] for key, value in totals[domain].items()}
        for domain in domains
    }
    if "clean" in domain_metrics and "ood" in domain_metrics:
        clean_mse = domain_metrics["clean"]["three_step_mse"]
        domain_metrics["ood"]["ood_clean_degradation_ratio"] = (
            domain_metrics["ood"]["three_step_mse"] / clean_mse
            if clean_mse > 0.0
            else None
        )
    metrics = {
        "schema_version": "jepa_wm_offline_metrics_v1",
        "method": method.method_name,
        "task": str(cfg.planning.task_slug),
        "window_count": len(selected),
        "window_identities": [list(pair) for pair in selected],
        "episode_partition": "eval",
        "domains": domain_metrics,
        "method_parameter_count": method.parameter_count(),
        "method_checkpoint_sha256": method_checkpoint_sha256,
        "base_checkpoint_sha256": backend.base_checkpoint_sha256,
        "dinov3_checkpoint_sha256": backend.dinov3_checkpoint_sha256,
        "upstream_commits": backend.upstream_commits,
        "appearance": ComposedPhotometricShift.metadata(
            float(cfg.appearance.severity), int(cfg.appearance.seed)
        ),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary_rows = output / "per_window.jsonl.tmp"
    temporary_rows.write_text(lines, encoding="utf-8")
    temporary_rows.replace(output / "per_window.jsonl")
    _atomic_json(output / "metrics.json", metrics)
    print(f"Offline metrics written: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
