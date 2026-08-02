from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

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
from wm_adapter.experiments.cross_benchmark import validate_task_manifest


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
