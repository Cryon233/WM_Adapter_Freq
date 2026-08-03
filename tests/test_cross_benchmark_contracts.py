from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from einops import rearrange
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm_adapter.benchmarks.base import (
    ActionTransform,
    ResolvedTask,
    canonical_sha256,
)
from wm_adapter.benchmarks.libero import (
    _camera_contract_from_shape,
    _validated_split_indices,
)
from wm_adapter.experiments.cross_benchmark import (
    JobSpec,
    block_job_for_failed_dependencies,
    load_suite_config,
    run_gpu_phase,
    training_contract_v2,
    validate_cache_v2,
    validate_checkpoint_v2,
    validate_task_manifest,
)
from wm_adapter.experiments.cross_jobs import build_job_graph
from wm_adapter.adapters.hfra import (
    HFRACoreOnlyAdapter,
    HybridFourierResidualAdapter,
)
from wm_adapter.data.feature_cache_v2 import (
    CACHE_SCHEMA_VERSION_V2,
    FeatureCacheV2Dataset,
    FeatureCacheV2Writer,
)
from wm_adapter.backends.jepa_wm_droid import JEPAWMDroidBackend
from wm_adapter.adapters.lora import LastBlockAttentionLoRA
from wm_adapter.backends.frozen_projection import frozen_base_projection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from monitor_all_paper_experiments import parse_training
from launch_cross_benchmark_suite import _args as launcher_args
from monitor_cross_benchmark_suite import _mark_stopped


def _training(*, steps: int = 2000, seed: int = 42) -> dict[str, object]:
    return {
        "max_optimizer_steps": steps,
        "microbatch_windows": 2,
        "views_per_window": 2,
        "gradient_accumulation": 8,
        "lr": 3.0e-4,
        "betas": [0.9, 0.95],
        "epsilon": 1.0e-8,
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "precision": "bf16",
        "seed": seed,
        "warmup_steps": 100,
        "minimum_lr": 3.0e-5,
        "scheduler": "cosine",
        "loss_name": "unified_trajectory_mse",
    }


def _cache_metadata() -> dict[str, object]:
    return {
        "benchmark": "libero",
        "task_key": "libero_goal_0",
        "task_manifest_sha256": "manifest",
        "dataset_sha256": "dataset",
        "camera_key": "agentview_rgb",
        "task_upstream_commits": {"libero": "commit"},
        "base_checkpoint_sha256": "base",
        "dinov3_checkpoint_sha256": "dino",
        "num_frames": 6,
        "context_frames": 3,
        "future_frames": 3,
        "num_encoder_blocks": 6,
        "middle_site_index": 3,
        "late_site_index": 5,
        "action_transform": _transform().as_dict(),
        "camera_height": 128,
        "camera_width": 160,
        "camera_channel_order": "RGB",
        "camera_vertical_flip": True,
    }


def _cache_batch(value: float) -> dict[str, object]:
    return {
        "clean_context_middle_tokens": torch.full((1, 3, 6, 8), value),
        "ood_context_middle_tokens": torch.full((1, 3, 6, 8), value + 1),
        "clean_target_latents": torch.full((1, 6, 4, 8), value + 2),
        "rollout_actions": torch.full((1, 3, 7), value + 3),
        "episode_id": torch.tensor([1]),
        "window_id": torch.tensor([2]),
        "source_trajectory_id": ["demo_1"],
        "appearance_seed": torch.tensor([2026]),
        "appearance_severity": torch.tensor([1.0]),
    }


class _OfficialRolloutFixture(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.ctxt_window = 2
        self.model = nn.Module()
        self.model.predictor = nn.Linear(dimension, dimension, bias=False)
        self.model.encoder = nn.Linear(dimension, dimension, bias=False)
        self.model.requires_grad_(False)

    def unroll(self, z_ctxt: torch.Tensor, act_suffix: torch.Tensor, debug: bool = False) -> torch.Tensor:
        del debug
        visual = z_ctxt
        for step in range(act_suffix.shape[0]):
            action = act_suffix[step].mean(dim=-1).reshape(-1, 1, 1, 1, 1)
            predicted = self.model.predictor(visual[:, -1]) + action
            visual = torch.cat((visual, predicted[:, None]), dim=1)
        return rearrange(visual, "b t ... -> t b ...")


def _transform(*, scale: float = 1.0, gripper: str = "identity") -> ActionTransform:
    arm = (scale,) * 6
    return ActionTransform(
        canonical_lower=tuple(-value for value in (*arm, 1.0)),
        canonical_upper=(*arm, 1.0),
        environment_lower=(-1.0,) * 7,
        environment_upper=(1.0,) * 7,
        controller_input_lower=(-1.0,) * 6,
        controller_input_upper=(1.0,) * 6,
        controller_output_lower=tuple(-value for value in arm),
        controller_output_upper=arm,
        translation_scale=arm[:3],
        rotation_scale=arm[3:],
        gripper_mapping=gripper,
        transform_name="contract_test",
        verified_identity=scale == 1.0 and gripper == "identity",
        verification_source="unit contract fixture",
        controller_type="OSC_POSE",
        control_frequency_hz=20.0,
        action_repeat=5,
    )


def _task(*, benchmark: str, transform: dict[str, object] | None) -> ResolvedTask:
    return ResolvedTask(
        task_key="robocasa_place" if benchmark == "robocasa" else "libero_goal_0",
        benchmark=benchmark,
        suite="single_stage" if benchmark == "robocasa" else "libero_goal",
        task_id=0,
        task_name="task",
        language_instruction=None,
        bddl_path=None,
        bddl_sha256=None,
        problem_folder=None,
        initial_states_sha256=None,
        initial_states_count=50,
        dataset_path="/dataset.hdf5",
        dataset_sha256="dataset",
        available_demonstrations=50,
        selected_train_demonstrations=tuple(f"demo_{index}" for index in range(30)),
        selected_test_demonstrations=tuple(f"demo_{index}" for index in range(30, 50)),
        camera_key="agentview_rgb",
        action_convention={"dimension": 7},
        environment_implementation="official",
        upstream_commits={},
        frameskip=5,
        max_episode_steps=300,
        episode_cap_basis="fixed",
        camera_height=128,
        camera_width=160,
        camera_channel_order="RGB",
        camera_vertical_flip=True,
        action_transform=transform,
    )


class CrossBenchmarkContractTest(unittest.TestCase):
    def test_hfra_identity_parameters_sites_and_gradients(self) -> None:
        adapter = HybridFourierResidualAdapter(
            embed_dim=1024,
            grid_height=16,
            grid_width=16,
            num_encoder_blocks=24,
            rank=4,
        )
        self.assertEqual(adapter.adapter_site_indices(24), (12, 23))
        self.assertEqual(adapter.parameter_count(), 17024)
        value = torch.randn(1, 2, 256, 1024)
        output = adapter.apply_at_site(12, value)
        self.assertTrue(torch.equal(output, value))
        output.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and torch.count_nonzero(parameter.grad) > 0
                for parameter in adapter.parameters()
            )
        )

    def test_hfra_bf16_identity_fft_shape_and_core_only_parameters(self) -> None:
        adapter = HybridFourierResidualAdapter(
            embed_dim=32,
            grid_height=4,
            grid_width=4,
            num_encoder_blocks=6,
            rank=4,
        ).to(torch.bfloat16)
        value = torch.randn(2, 3, 16, 32, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(adapter.apply_at_site(3, value), value))
        site = adapter.sites["3"]
        core = site.activation(site.down(site.norm(value)))
        self.assertEqual(site._spectral_residual(core).shape, core.shape)
        core_only = HFRACoreOnlyAdapter(
            embed_dim=32,
            grid_height=4,
            grid_width=4,
            num_encoder_blocks=6,
            rank=4,
        )
        self.assertFalse(
            any("frequency" in name or "channel_mixer" in name for name, _ in core_only.named_parameters())
        )
        diagnostics = adapter.sites["3"].latest_diagnostics()
        self.assertEqual(float(diagnostics["core_delta_ratio"]), 0.0)
        self.assertEqual(float(diagnostics["spectral_delta_ratio"]), 0.0)
        self.assertEqual(float(diagnostics["total_delta_ratio"]), 0.0)

    def test_official_rollout_parity_gradient_boundary_and_prefix_protection(self) -> None:
        backend = JEPAWMDroidBackend.__new__(JEPAWMDroidBackend)
        nn.Module.__init__(backend)
        backend.grid_height = 2
        backend.grid_width = 2
        backend.num_patch_tokens = 4
        backend.token_dim = 8
        backend.num_encoder_blocks = 4
        backend.official_model = _OfficialRolloutFixture(8)
        backend.encoder = backend.official_model.model.encoder
        adapter = HybridFourierResidualAdapter(
            embed_dim=8,
            grid_height=2,
            grid_width=2,
            num_encoder_blocks=4,
            rank=2,
        )
        context = torch.randn(2, 3, 4, 8)
        adapted = adapter.apply_at_site(2, context)
        actions = torch.randn(2, 3, 7)
        actual = backend.differentiable_unroll(adapted, actions)
        direct = backend.official_model.unroll(
            backend.planning_latents(adapted),
            rearrange(actions, "b t a -> t b a"),
        )[-3:]
        direct = rearrange(direct, "t b 1 h w d -> b t (h w) d")
        self.assertTrue(torch.allclose(actual, direct))
        actual.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and torch.count_nonzero(parameter.grad)
                for parameter in adapter.parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in backend.official_model.parameters())
        )

        flattened = torch.randn(6, 6, 8)
        prefix = flattened[:, :2].clone()
        output = backend._apply_method_site(flattened, adapter, 2, 2, 3, 2)
        self.assertTrue(torch.equal(output.reshape(6, 6, 8)[:, :2], prefix))

    def test_lora_frozen_base_projection_matches_unattached_goal_projection(self) -> None:
        backend = nn.Module()
        backend.last_block = nn.Module()
        backend.last_block.attn = nn.Module()
        original = nn.Linear(8, 24)
        backend.last_block.attn.qkv = original
        probe = torch.randn(2, 3, 8)
        expected = original(probe)
        method = LastBlockAttentionLoRA(embed_dim=8, rank=2, alpha=2.0)
        method.attach_backend(backend)
        with frozen_base_projection(backend):
            goal = backend.last_block.attn.qkv(probe)
        self.assertTrue(torch.equal(goal, expected))

    def test_v2_loader_rejects_v1_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["schema_version"] = "jepa_wm_robocasa_feature_cache_v2"
                handle.attrs["finalized"] = True
            with self.assertRaisesRegex(RuntimeError, "V2 loader rejects"):
                FeatureCacheV2Dataset(path)

    def test_v2_cache_content_fingerprint_roundtrip_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fingerprints = []
            for index, value in enumerate((1.0, 2.0)):
                path = Path(directory) / f"cache-{index}.h5"
                writer = FeatureCacheV2Writer(path, _cache_metadata())
                writer.append(_cache_batch(value))
                fingerprints.append(writer.finalize())
                dataset = FeatureCacheV2Dataset(path)
                self.assertEqual(len(dataset), 1)
                self.assertEqual(
                    float(dataset[0]["rollout_actions"].mean()), value + 3
                )
                validate_cache_v2(
                    path,
                    1,
                    benchmark="libero",
                    task="libero_goal_0",
                    expected_task_manifest_sha256="manifest",
                    expected_dataset_sha256="dataset",
                    expected_camera_key="agentview_rgb",
                    expected_task_upstream_commits={"libero": "commit"},
                    expected_action_transform=_transform().as_dict(),
                    expected_camera_contract={
                        "camera_height": 128,
                        "camera_width": 160,
                        "camera_channel_order": "RGB",
                        "camera_vertical_flip": True,
                    },
                    expected_base_checkpoint_sha256="base",
                    expected_dinov3_checkpoint_sha256="dino",
                )
            self.assertNotEqual(*fingerprints)

            missing_metadata = Path(directory) / "missing-metadata.h5"
            writer = FeatureCacheV2Writer(missing_metadata, _cache_metadata())
            writer.append(_cache_batch(3.0))
            writer.finalize()
            with h5py.File(missing_metadata, "r+") as handle:
                del handle.attrs["camera_key"]
            with self.assertRaisesRegex(RuntimeError, "missing metadata"):
                validate_cache_v2(
                    missing_metadata,
                    1,
                    benchmark="libero",
                    task="libero_goal_0",
                    expected_task_manifest_sha256="manifest",
                    expected_dataset_sha256="dataset",
                    expected_camera_key="agentview_rgb",
                    expected_task_upstream_commits={"libero": "commit"},
                    expected_action_transform=_transform().as_dict(),
                    expected_camera_contract={
                        "camera_height": 128,
                        "camera_width": 160,
                        "camera_channel_order": "RGB",
                        "camera_vertical_flip": True,
                    },
                    expected_base_checkpoint_sha256="base",
                    expected_dinov3_checkpoint_sha256="dino",
                )

            unfinalized = Path(directory) / "unfinalized.h5"
            writer = FeatureCacheV2Writer(unfinalized, _cache_metadata())
            temporary = writer.temporary_path
            writer.file.flush()
            writer.file.close()
            with self.assertRaisesRegex(RuntimeError, "not finalized"):
                FeatureCacheV2Dataset(temporary)
            temporary.unlink()

    def test_structured_step_progress_parser(self) -> None:
        text = (
            "TRAIN_PROGRESS step=120 total=2000 loss=0.0123 clean_mse=0.0098 "
            "ood_mse=0.0148 context_mse=0.0101 future_mse=0.0145 "
            "lr=0.000287 grad_norm=0.82 samples_per_sec=14.3 "
            "core_delta_ratio=0.031 spectral_delta_ratio=0.008"
        )
        progress = parse_training(text, "log updated 0s ago")
        self.assertIsNotNone(progress)
        assert progress is not None
        self.assertAlmostEqual(progress.percent, 6.0)
        self.assertIn("step 120/2000", progress.detail)

    def test_v2_checkpoint_contract_and_versioned_job_graphs(self) -> None:
        camera = {
            "camera_height": 128,
            "camera_width": 160,
            "camera_channel_order": "RGB",
            "camera_vertical_flip": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hfra.pt"
            training_contract = training_contract_v2(_training())
            payload = {
                    "schema_version": "wm_adapter_checkpoint_v2",
                    "method_name": "hfra",
                    "method_config": {"rank": 4},
                    "peft_state_dict": {},
                    "trainable_parameter_count": 17024,
                    "cache_fingerprint": "cache",
                    "base_checkpoint_sha256": "base",
                    "dinov3_checkpoint_sha256": "dino",
                    "upstream_commits": {},
                    **{
                        key: value
                        for key, value in training_contract.items()
                        if key != "effective_view_batch"
                    },
                    "data_metadata": {
                        "benchmark": "libero",
                        "task_key": "libero_goal_0",
                        "action_transform": _transform().as_dict(),
                        "task_manifest_sha256": "manifest",
                        "dataset_sha256": "dataset",
                        "camera_key": "agentview_rgb",
                        "task_upstream_commits": {"libero": "commit"},
                        **camera,
                    },
                }
            torch.save(payload, path)
            validate_checkpoint_v2(
                path,
                "hfra",
                "cache",
                benchmark="libero",
                task="libero_goal_0",
                expected_method_config={"name": "hfra", "rank": 4},
                expected_training_contract=training_contract,
                expected_action_transform=_transform().as_dict(),
                expected_camera_contract=camera,
                expected_data_contract={
                    "task_manifest_sha256": "manifest",
                    "dataset_sha256": "dataset",
                    "camera_key": "agentview_rgb",
                    "task_upstream_commits": {"libero": "commit"},
                },
            )
            payload["optimizer_config"] = dict(payload["optimizer_config"])
            payload["optimizer_config"]["lr"] = 1.0e-3
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "optimizer_config"):
                validate_checkpoint_v2(
                    path,
                    "hfra",
                    "cache",
                    benchmark="libero",
                    task="libero_goal_0",
                    expected_method_config={"name": "hfra", "rank": 4},
                    expected_training_contract=training_contract,
                    expected_action_transform=_transform().as_dict(),
                    expected_camera_contract=camera,
                    expected_data_contract={
                        "task_manifest_sha256": "manifest",
                        "dataset_sha256": "dataset",
                        "camera_key": "agentview_rgb",
                        "task_upstream_commits": {"libero": "commit"},
                    },
                )
        root = Path(__file__).resolve().parents[1]
        v1 = load_suite_config(root / "configs/experiment/cross_benchmark_v1.yaml")
        v2 = load_suite_config(root / "configs/experiment/cross_benchmark_v2.yaml")
        self.assertTrue(build_job_graph(v1))
        self.assertTrue(any(job.kind == "protocol" for job in build_job_graph(v2)))
        self_test_jobs = build_job_graph(v2, self_test=True)
        for job in self_test_jobs:
            if job.kind in {"checkpoint", "offline", "planning"}:
                self.assertIn("training.max_optimizer_steps=2", job.command)
                self.assertIn("suite_mode=self_test", job.command)

    def test_scheduler_continues_independent_method_and_blocks_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = JobSpec(
                job_id="train/task/method1", phase="train", benchmark="robocasa",
                task="task", method="method1", command=(sys.executable, "-c", "raise SystemExit(1)"),
                log_path=str(root / "failed.log"), artifact_path=str(root / "failed.pt"), kind="checkpoint",
            )
            independent = JobSpec(
                job_id="train/task/method2", phase="train", benchmark="robocasa",
                task="task", method="method2", command=(sys.executable, "-c", "print('ok')"),
                log_path=str(root / "ok.log"), artifact_path=str(root / "ok.pt"), kind="checkpoint",
            )
            state: dict[str, object] = {"jobs": {}}
            with patch(
                "wm_adapter.experiments.cross_benchmark._gpu_free_memory_mib",
                return_value={0: 40960},
            ), patch(
                "wm_adapter.experiments.cross_benchmark.benchmark_subprocess_environment",
                return_value=os.environ.copy(),
            ), patch.dict(os.environ, {"WM_ADAPTER_MIN_GPU_FREE_MIB": "0"}):
                failures = run_gpu_phase(
                    [failed, independent], [0], root / "state.json", state,
                    lambda job, path: {"path": path}, raise_on_failure=False,
                )
            self.assertEqual(failures, {failed.job_id})
            self.assertEqual(state["jobs"][independent.job_id]["status"], "completed")
            downstream = JobSpec(
                job_id="offline/task/method1", phase="offline", benchmark="robocasa",
                task="task", method="method1", command=(), log_path="log",
                artifact_path="artifact", kind="offline", dependencies=(failed.job_id,),
            )
            self.assertTrue(block_job_for_failed_dependencies(downstream, state))
            self.assertEqual(state["jobs"][downstream.job_id]["status"], "blocked")

    def test_self_test_lifecycle_arguments_and_dashboard_stop_state(self) -> None:
        for action in ("--status", "--attach", "--stop"):
            parsed = launcher_args(["--self-test", action])
            self.assertTrue(parsed.self_test)
            self.assertTrue(getattr(parsed, action.removeprefix("--").replace("-", "_")))
        parsed = launcher_args(["--self-test", "--dry-run"])
        self.assertTrue(parsed.self_test and parsed.dry_run)
        suite = Path(__file__).resolve().parents[1] / "configs/experiment/cross_benchmark_v2.yaml"
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text(
                json.dumps({"suite": "cross_benchmark_v2", "status": "running", "jobs": {"job": {"status": "running"}}}),
                encoding="utf-8",
            )
            _mark_stopped(state_path, suite)
            stopped = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(stopped["status"], "stopped")
            self.assertEqual(stopped["jobs"]["job"]["status"], "stopped")

    def test_split_capacity_uses_requested_count(self) -> None:
        train, heldout = _validated_split_indices(50, 0.6, 42, 20)
        self.assertEqual((len(train), len(heldout)), (30, 20))
        _validated_split_indices(50, 0.6, 42, 1)
        with self.assertRaisesRegex(RuntimeError, "held_out_count=16"):
            _validated_split_indices(40, 0.6, 42, 20)

    def test_camera_contract_comes_from_hdf5_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.hdf5"
            with h5py.File(path, "w") as handle:
                image = handle.create_dataset(
                    "agentview_rgb",
                    shape=(3, 96, 144, 3),
                    dtype=np.uint8,
                )
                contract = _camera_contract_from_shape(
                    image.shape,
                    vertical_flip=True,
                )
        self.assertEqual(contract["camera_height"], 96)
        self.assertEqual(contract["camera_width"], 144)

    def test_action_round_trips(self) -> None:
        identity = _transform()
        action = np.asarray([0.5, -0.25, 0.1, -0.2, 0.3, 0.0, 1.0])
        self.assertTrue(
            np.allclose(identity.canonical_to_environment_action(action), action)
        )
        scaled = _transform(scale=0.05)
        canonical = np.asarray([0.01, -0.02, 0.05, 0.0, 0.03, -0.01, -1.0])
        environment = scaled.canonical_to_environment_action(canonical)
        self.assertTrue(
            np.allclose(
                scaled.environment_to_canonical_action(environment),
                canonical,
            )
        )
        inverted = _transform(gripper="inverted")
        environment = inverted.canonical_to_environment_action(action)
        self.assertEqual(float(environment[-1]), -1.0)
        self.assertTrue(
            np.allclose(
                inverted.environment_to_canonical_action(environment),
                action,
            )
        )

    def test_manifest_hash_and_strict_libero_contract(self) -> None:
        transform = _transform(scale=0.05).as_dict()
        first = _task(benchmark="libero", transform=transform).as_dict()
        second = _task(benchmark="libero", transform=transform).as_dict()
        self.assertEqual(first["task_manifest_sha256"], second["task_manifest_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.json"
            valid.write_text(json.dumps(first), encoding="utf-8")
            validate_task_manifest(valid, "libero_goal_0")
            missing = _task(benchmark="libero", transform=None).as_dict()
            invalid = Path(directory) / "invalid.json"
            invalid.write_text(json.dumps(missing), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lacks an action transform"):
                validate_task_manifest(invalid, "libero_goal_0")

    def test_legacy_robocasa_manifest_remains_explicitly_compatible(self) -> None:
        payload = _task(benchmark="robocasa", transform=None).as_dict()
        for key in (
            "task_manifest_sha256",
            "camera_height",
            "camera_width",
            "camera_channel_order",
            "camera_vertical_flip",
            "action_transform",
        ):
            payload.pop(key, None)
        payload["task_manifest_sha256"] = canonical_sha256(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            validate_task_manifest(
                path,
                "robocasa_place",
                allow_legacy_place=True,
            )


if __name__ == "__main__":
    unittest.main()
