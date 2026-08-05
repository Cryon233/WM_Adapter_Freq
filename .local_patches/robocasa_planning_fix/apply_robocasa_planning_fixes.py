#!/usr/bin/env python3
"""Apply verified RoboCasa planning fixes to WM_Adapter_Freq.

Target baseline: Cryon233/WM_Adapter_Freq commit
2ca0a5d5a6beb885354ccb01ca429e75b5e3811d.

The patcher is transactional at the source-file level:
- all target transformations are validated in memory first;
- originals and the pre-existing git diff are backed up;
- files are written atomically;
- py_compile and git diff --check are run after writing.

It intentionally preserves unrelated local edits when the expected source anchors
are still present. If an anchor changed, it aborts before writing anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

EXPECTED_BASE = "2ca0a5d5a6beb885354ccb01ca429e75b5e3811d"

PLAN_EVALUATOR = Path(
    "third_party/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py"
)
PLANNING_UTILS = Path(
    "third_party/jepa-wms/evals/simu_env_planning/planning/utils.py"
)
JEPA_PLANNER = Path("src/wm_adapter/planning/jepa_wm_planner.py")
ROBOCASA_BENCHMARK = Path("src/wm_adapter/benchmarks/robocasa.py")
SUITE_RUNNER = Path("scripts/run_cross_backend_adapter_suite.py")
MONITOR_CORE = Path("scripts/monitor_all_paper_experiments.py")
TARGETS = (
    PLAN_EVALUATOR,
    PLANNING_UTILS,
    JEPA_PLANNER,
    ROBOCASA_BENCHMARK,
    SUITE_RUNNER,
    MONITOR_CORE,
)


class PatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileChange:
    path: Path
    original: str
    modified: str


def run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def regex_replace_exact(
    text: str,
    pattern: str,
    replacement: str,
    *,
    expected: int,
    label: str,
) -> str:
    result, count = re.subn(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
    if count != expected:
        raise PatchError(f"{label}: expected {expected} matches, found {count}")
    return result


def patch_plan_evaluator(text: str) -> str:
    helper = '''log = get_logger(__name__)\n\n\nROBOCASA_PLANNING_INPUT_CONTRACT = "raw_rgb_uint8_or_unit_float_v1"\nROBOCASA_SUCCESS_AGGREGATION = "episode_any_low_level_step_v1"\n\n\ndef _raw_rgb_to_uint8(frames: torch.Tensor) -> torch.Tensor:\n    """Convert raw RGB to uint8 without applying an inverse normalization.\n\n    The WM Adapter benchmark constructs the RoboCasa source dataset with\n    ``transform=None``. Its RGB tensors are therefore either uint8 or floats in\n    [0, 1], not mean/std-normalized tensors. Calling the upstream\n    ``inverse_transform`` on these values corrupts the model input contract.\n    """\n    values = torch.as_tensor(frames)\n    if values.ndim not in {3, 4}:\n        raise ValueError(\n            "Raw planning RGB must be [C,H,W] or [T,C,H,W], "\n            f"received {tuple(values.shape)}"\n        )\n    if values.shape[-3] != 3:\n        raise ValueError(\n            f"Raw planning RGB must have three channels, received {tuple(values.shape)}"\n        )\n    if values.dtype == torch.uint8:\n        return values.detach().clone()\n    values = values.detach().float()\n    if not torch.isfinite(values).all():\n        raise ValueError("Raw planning RGB contains non-finite values")\n    minimum = float(values.amin())\n    maximum = float(values.amax())\n    tolerance = 1.0e-6\n    if minimum < -tolerance or maximum > 1.0 + tolerance:\n        raise ValueError(\n            "Raw planning float RGB must be in [0,1], "\n            f"found [{minimum}, {maximum}]"\n        )\n    return values.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)\n'''
    text = replace_once(
        text,
        'log = get_logger(__name__)\n',
        helper,
        "plan_evaluator: insert raw-RGB contract",
    )

    old_init = '''    def unroll_agent(self, env, obs, info, actor, preprocessor=None):\n        """\n        Returns:\n        List of Tensordicts with length the episode length, each td has 2 fields: visual and proprio\n        If "droid" in cfg.task_specification.task, the proprio and obs outputted by env.step_multiple() are dummy\n            so should not replan on them, hence done = True after first call to agent_actor.\n        """\n        done = False\n        ep_reward = 0\n        td = obs\n'''
    new_init = '''    def unroll_agent(self, env, obs, info, actor, preprocessor=None):\n        """\n        Returns:\n        List of Tensordicts with length the episode length, each td has 2 fields: visual and proprio\n        If "droid" in cfg.task_specification.task, the proprio and obs outputted by env.step_multiple() are dummy\n            so should not replan on them, hence done = True after first call to agent_actor.\n        """\n        done = False\n        ep_reward = 0\n        episode_success = False\n        state_dist = float("inf")\n        td = obs\n'''
    text = replace_once(
        text,
        old_init,
        new_init,
        "plan_evaluator: initialize episode success",
    )

    old_success = '''            # Evaluate success\n            if (\n                any(pref in self.cfg.task_specification.task for pref in ["mw", "robocasa"])\n                and self.cfg.task_specification.succ_def == "simu"\n            ):\n                success = infos[-1]["success"]\n                if str(self.cfg.task_specification).startswith("robocasa-"):\n                    state_dist = infos[-1]["hand_obj_dist"]\n                else:\n                    state_dist = np.linalg.norm(self.state_g - infos[-1]["state"])\n            else:\n                eval_results = env.eval_state(self.state_g, infos[-1]["state"])\n                success = eval_results["success"]\n                state_dist = eval_results["state_dist"]\n            if success and self.cfg.task_specification.done_at_succ:\n                done = True\n'''
    new_success = '''            # Evaluate success. RoboCasa success is an event that can occur at\n            # any low-level step inside an action-repeat block. Preserve it for\n            # the whole episode instead of overwriting it with the final step.\n            task_name = str(self.cfg.task_specification.task)\n            if (\n                any(pref in task_name for pref in ["mw", "robocasa"])\n                and self.cfg.task_specification.succ_def == "simu"\n            ):\n                step_success = any(\n                    bool(step_info.get("success", False)) for step_info in infos\n                )\n                if task_name.startswith("robocasa-"):\n                    subtask = str(\n                        self.cfg.task_specification.env.get("subtask", "")\n                    )\n                    distance_key = (\n                        "obj_goal_dist" if "place" in subtask else "hand_obj_dist"\n                    )\n                    current_state_dist = float(\n                        infos[-1].get(distance_key, float("inf"))\n                    )\n                else:\n                    current_state_dist = float(\n                        np.linalg.norm(self.state_g - infos[-1]["state"])\n                    )\n            else:\n                eval_results = env.eval_state(self.state_g, infos[-1]["state"])\n                step_success = bool(eval_results["success"])\n                current_state_dist = float(eval_results["state_dist"])\n            episode_success = episode_success or step_success\n            state_dist = min(state_dist, current_state_dist)\n            if episode_success and self.cfg.task_specification.done_at_succ:\n                done = True\n'''
    text = replace_once(
        text,
        old_success,
        new_success,
        "plan_evaluator: aggregate low-level success",
    )

    old_postfix = '''                    "obj_goal_dist": infos[-1].get("obj_goal_dist", -1.0),\n                    "success": success,\n                    "obj_lift": infos[-1].get("obj_lift", -1.0),\n'''
    new_postfix = '''                    "obj_goal_dist": infos[-1].get("obj_goal_dist", -1.0),\n                    "success": episode_success,\n                    "obj_lift": infos[-1].get("obj_lift", -1.0),\n'''
    text = replace_once(
        text,
        old_postfix,
        new_postfix,
        "plan_evaluator: report accumulated success",
    )

    text = replace_once(
        text,
        '''        return ep_obs_proprio_td_list, ep_reward, actions, infos_list, success, state_dist\n''',
        '''        return (\n            ep_obs_proprio_td_list,\n            ep_reward,\n            actions,\n            infos_list,\n            episode_success,\n            state_dist,\n        )\n''',
        "plan_evaluator: return accumulated success",
    )

    inverse_pattern = r'''\(\s*self\.agent\.preprocessor\.inverse_transform\(\s*observations\["visual"\]\[i : i \+ 1\]\s*\)\s*\*\s*255\s*\)\.to\(torch\.uint8\)'''
    text = regex_replace_exact(
        text,
        inverse_pattern,
        '_raw_rgb_to_uint8(observations["visual"][i : i + 1])',
        expected=2,
        label="plan_evaluator: remove invalid inverse transforms",
    )

    old_reset = '''                # Important: reprepare env back to same initial state\n                reset_vis, reset_info = env.prepare(ep_seed, init_state, env_info=env_info)\n                if "max_episode_steps" in cfg.task_specification:\n                    env.set_max_steps(cfg.task_specification.max_episode_steps)\n            init_obs = expert_obses[0]\n'''
    new_reset = '''                # Important: reprepare env back to the same initial state.\n                # Use the simulator render as the current observation so the\n                # first and subsequent history frames share one camera/size path.\n                reset_vis, reset_info = env.prepare(\n                    ep_seed, init_state, env_info=env_info\n                )\n                init_obs = make_td(reset_vis, reset_info)\n                if "max_episode_steps" in cfg.task_specification:\n                    env.set_max_steps(cfg.task_specification.max_episode_steps)\n            if "droid" in cfg.task_specification.task:\n                init_obs = expert_obses[0]\n'''
    text = replace_once(
        text,
        old_reset,
        new_reset,
        "plan_evaluator: use restored simulator observation",
    )

    old_media = '''        else:\n            if cfg.logging.optional_plots and cfg.task_specification.num_frames > agent.model.tubelet_size_enc:\n                # Keep the minimal last state\n                for x in episode_obses:\n                    x["visual"] = x["visual"][-agent.model.tubelet_size_enc :]\n            agent_goal_video_path = str(vis_work_dir / f"video_agent_goal_{'succ' if success else 'fail'}")\n            frames_list = [x["visual"] for x in episode_obses]\n            make_video(frames_list, 30, agent_goal_video_path, obs_concat_channels=env.obs_concat_channels)\n            make_video_pdf(\n                frames_list[:: self.cfg.frameskip],\n                agent_goal_video_path + ".pdf",\n                obs_concat_channels=env.obs_concat_channels,\n            )\n\n            coord_diffs, _repr_diffs = analyze_distances(\n                agent,\n                episode_obses,\n                goal_obs,\n                str(dist_work_dir / "agent"),\n                objective=agent.objective,\n            )\n            success_dist = float(coord_diffs[-1] < 0.05)\n            end_distance = coord_diffs[-1]\n            end_distance_xyz, end_distance_orientation, end_distance_closure = -1.0, -1.0, -1.0\n'''
    new_media = '''        else:\n            if cfg.logging.optional_plots:\n                if cfg.task_specification.num_frames > agent.model.tubelet_size_enc:\n                    # Keep the minimal last state\n                    for x in episode_obses:\n                        x["visual"] = x["visual"][-agent.model.tubelet_size_enc :]\n                agent_goal_video_path = str(\n                    vis_work_dir / f"video_agent_goal_{'succ' if success else 'fail'}"\n                )\n                frames_list = [x["visual"] for x in episode_obses]\n                make_video(\n                    frames_list,\n                    30,\n                    agent_goal_video_path,\n                    obs_concat_channels=env.obs_concat_channels,\n                )\n                make_video_pdf(\n                    frames_list[:: self.cfg.frameskip],\n                    agent_goal_video_path + ".pdf",\n                    obs_concat_channels=env.obs_concat_channels,\n                )\n                coord_diffs, _repr_diffs = analyze_distances(\n                    agent,\n                    episode_obses,\n                    goal_obs,\n                    str(dist_work_dir / "agent"),\n                    objective=agent.objective,\n                )\n                success_dist = float(coord_diffs[-1] < 0.05)\n                end_distance = float(coord_diffs[-1])\n            else:\n                # Formal sweeps disable optional plots. Avoid video encoding and\n                # an additional full latent-distance pass for every episode.\n                success_dist = float(success)\n                end_distance = float(state_dist)\n            end_distance_xyz, end_distance_orientation, end_distance_closure = (\n                -1.0,\n                -1.0,\n                -1.0,\n            )\n'''
    text = replace_once(
        text,
        old_media,
        new_media,
        "plan_evaluator: honor optional_plots",
    )
    return text


def patch_planning_utils(text: str) -> str:
    helper = '''FIGSIZE_BASE = (4.0, 3.0)\n\n\ndef _video_frame_to_uint8(image, *, obs_concat_channels=True):\n    """Convert one raw RGB observation to HWC uint8 without per-frame scaling."""\n    if isinstance(image, torch.Tensor):\n        values = image.detach().cpu().numpy()\n    else:\n        values = np.asarray(image)\n    if values.ndim == 4:\n        values = values[-3:] if obs_concat_channels else values[-1]\n    if values.ndim != 3:\n        raise ValueError(f"Video frame must be 3-D after selection, got {values.shape}")\n    if values.shape[0] == 3:\n        values = values.transpose(1, 2, 0)\n    elif values.shape[-1] != 3:\n        raise ValueError(f"Video frame must have three RGB channels, got {values.shape}")\n    if values.dtype == np.uint8:\n        return np.ascontiguousarray(values)\n    values = values.astype(np.float32, copy=False)\n    if not np.isfinite(values).all():\n        raise ValueError("Video frame contains non-finite values")\n    minimum = float(values.min())\n    maximum = float(values.max())\n    tolerance = 1.0e-6\n    if minimum >= -tolerance and maximum <= 1.0 + tolerance:\n        values = np.rint(np.clip(values, 0.0, 1.0) * 255.0)\n    elif minimum >= -tolerance and maximum <= 255.0 + tolerance:\n        values = np.rint(np.clip(values, 0.0, 255.0))\n    else:\n        raise ValueError(\n            "Video RGB must be uint8, [0,1] float, or [0,255] float; "\n            f"found [{minimum}, {maximum}]"\n        )\n    return np.ascontiguousarray(values.astype(np.uint8))\n'''
    text = replace_once(
        text,
        'FIGSIZE_BASE = (4.0, 3.0)\n',
        helper,
        "planning.utils: insert stable RGB conversion",
    )

    old_mp4 = '''    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")\n    for img in images:\n        img = (img[-3:] if obs_concat_channels else img[-1]).numpy() if isinstance(img, torch.Tensor) else img\n        img = img.transpose(1, 2, 0)\n        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)\n        # img: H W C\n        writer.append_data(img)\n'''
    new_mp4 = '''    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")\n    for img in images:\n        writer.append_data(\n            _video_frame_to_uint8(\n                img, obs_concat_channels=obs_concat_channels\n            )\n        )\n'''
    text = replace_once(
        text,
        old_mp4,
        new_mp4,
        "planning.utils: remove MP4 min-max scaling",
    )

    old_gif = '''    writer = imageio.get_writer(output_path, fps=fps, format="GIF", loop=10000)\n    for img in images:\n        img = (img[-3:] if obs_concat_channels else img[-1]).numpy() if isinstance(img, torch.Tensor) else img\n        img = img.transpose(1, 2, 0)\n        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)\n        writer.append_data(img)\n'''
    new_gif = '''    writer = imageio.get_writer(output_path, fps=fps, format="GIF", loop=10000)\n    for img in images:\n        writer.append_data(\n            _video_frame_to_uint8(\n                img, obs_concat_channels=obs_concat_channels\n            )\n        )\n'''
    text = replace_once(
        text,
        old_gif,
        new_gif,
        "planning.utils: remove GIF min-max scaling",
    )

    old_pdf = '''    # Process images to consistent format\n    processed_images = []\n    for img in images:\n        if isinstance(img, torch.Tensor):\n            img = (img[-3:] if obs_concat_channels else img[-1]).detach().cpu().numpy()\n\n        # Convert to HWC format\n        img = img.transpose(1, 2, 0)\n        processed_images.append(img)\n'''
    new_pdf = '''    # Process images with the same fixed RGB contract as MP4/GIF output.\n    processed_images = [\n        _video_frame_to_uint8(\n            image, obs_concat_channels=obs_concat_channels\n        )\n        for image in images\n    ]\n'''
    text = replace_once(
        text,
        old_pdf,
        new_pdf,
        "planning.utils: stabilize PDF frame conversion",
    )
    return text


def patch_jepa_planner(text: str) -> str:
    text = replace_once(
        text,
        'EVALUATION_PROTOCOL_VERSION = "2.0"\nEVALUATION_PROTOCOL_DIRECTORY = "protocol_v2"\n',
        'EVALUATION_PROTOCOL_VERSION = "2.1"\nEVALUATION_PROTOCOL_DIRECTORY = "protocol_v2_1"\n',
        "jepa_wm_planner: bump planning protocol",
    )

    helper_anchor = '''@torch.inference_mode()\ndef frozen_goal_latent_fingerprint(\n'''
    helper = '''def _raw_rgb_frame_to_uint8(frame: Tensor) -> Tensor:\n    """Match the exact uint8 goal representation used by PlanEvaluator."""\n    if frame.ndim != 3 or frame.shape[0] != 3:\n        raise ValueError(\n            f"Raw goal RGB must be [3,H,W], received {tuple(frame.shape)}"\n        )\n    if frame.dtype == torch.uint8:\n        return frame.detach().clone()\n    values = frame.detach().float()\n    if not torch.isfinite(values).all():\n        raise ValueError("Raw goal RGB contains non-finite values")\n    minimum = float(values.amin())\n    maximum = float(values.amax())\n    if minimum < -1.0e-6 or maximum > 1.0 + 1.0e-6:\n        raise ValueError(\n            f"Raw goal float RGB must be in [0,1], found [{minimum}, {maximum}]"\n        )\n    return values.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)\n\n\n@torch.inference_mode()\ndef frozen_goal_latent_fingerprint(\n'''
    text = replace_once(
        text,
        helper_anchor,
        helper,
        "jepa_wm_planner: add exact goal RGB conversion",
    )

    old_history = '''            current = obs[-1].detach().clone()\n            self._current_history.append(current)\n'''
    new_history = '''            current = obs[-1].detach().clone()\n            if self._current_history and tuple(current.shape) != tuple(\n                self._current_history[-1].shape\n            ):\n                raise RuntimeError(\n                    "Planning history changed image shape between replans: "\n                    f"previous={tuple(self._current_history[-1].shape)}, "\n                    f"current={tuple(current.shape)}. The initial observation must "\n                    "come from the restored simulator render, not the dataset RGB."\n                )\n            self._current_history.append(current)\n'''
    text = replace_once(
        text,
        old_history,
        new_history,
        "jepa_wm_planner: guard history image shape",
    )

    old_span = '''        # The pinned Place evaluator samples the complete contiguous\n        # ``subtask=place`` slice.  Compatibility manifests preserve that exact\n        # official sampling stream; fixed-span validation applies to every\n        # other v2 manifest.\n        variable_official_place_segment = bool(\n            manifest.get("legacy_place_reuse_compatible", False)\n        ) and str(experiment_config.benchmark.task_key) == "robocasa_place"\n        invalid_spans = (\n            [\n                {\n                    "instance_id": str(instance.get("instance_id", "")),\n                    "start": int(instance["segment_start"]),\n                    "end": int(instance["segment_end"]),\n                }\n                for instance in manifest_instances\n                if int(instance["segment_end"]) - int(instance["segment_start"])\n                != goal_span_steps\n            ]\n            if goal_span_steps is not None and not variable_official_place_segment\n            else []\n        )\n'''
    new_span = '''        # Protocol 2.1 requires the configured fixed goal span. Variable-length\n        # legacy Place segments are invalid and must be rebuilt.\n        invalid_spans = (\n            [\n                {\n                    "instance_id": str(instance.get("instance_id", "")),\n                    "start": int(instance["segment_start"]),\n                    "end": int(instance["segment_end"]),\n                }\n                for instance in manifest_instances\n                if int(instance["segment_end"]) - int(instance["segment_start"])\n                != goal_span_steps\n            ]\n            if goal_span_steps is not None\n            else []\n        )\n'''
    text = replace_once(
        text,
        old_span,
        new_span,
        "jepa_wm_planner: enforce fixed Place span",
    )

    old_fingerprint = '''                goal_base_latent_fingerprint = frozen_goal_latent_fingerprint(\n                    backend, selected_observation["visual"][-1]\n                )\n'''
    new_fingerprint = '''                goal_base_latent_fingerprint = frozen_goal_latent_fingerprint(\n                    backend,\n                    _raw_rgb_frame_to_uint8(\n                        selected_observation["visual"][-1]\n                    ),\n                )\n'''
    text = replace_once(
        text,
        old_fingerprint,
        new_fingerprint,
        "jepa_wm_planner: fingerprint actual goal representation",
    )
    return text


def patch_robocasa_benchmark(text: str) -> str:
    start_marker = '                frame_indices = np.flatnonzero(segments == segment_code)\n'
    end_marker = '            return self.finalize_evaluation_manifest(\n'
    start = text.find(start_marker)
    if start < 0:
        raise PatchError("robocasa.py: manifest segment start anchor not found")
    end = text.find(end_marker, start)
    if end < 0:
        raise PatchError("robocasa.py: manifest segment end anchor not found")

    replacement = '''                full_segment_indices = np.flatnonzero(segments == segment_code)\n                if full_segment_indices.size == 0 or not np.array_equal(\n                    full_segment_indices,\n                    np.arange(\n                        full_segment_indices[0],\n                        full_segment_indices[-1] + 1,\n                    ),\n                ):\n                    raise RuntimeError(\n                        f"RoboCasa {subtask_name} compatibility requires one contiguous "\n                        f"segment code {segment_code}: trajectory={trajectory}, "\n                        f"indices={full_segment_indices.tolist()}"\n                    )\n                if goal_span > 0:\n                    required_frames = goal_span + 1\n                    if int(full_segment_indices.size) < required_frames:\n                        raise RuntimeError(\n                            f"RoboCasa {subtask_name} segment is shorter than the "\n                            f"configured goal span: trajectory={trajectory}, "\n                            f"segment_frames={full_segment_indices.size}, "\n                            f"required_frames={required_frames}"\n                        )\n                    selected_frame_indices = full_segment_indices[-required_frames:]\n                else:\n                    selected_frame_indices = full_segment_indices\n                selected_start = int(selected_frame_indices[0])\n                selected_end = int(selected_frame_indices[-1])\n                (\n                    selected_observation,\n                    _,\n                    selected_states,\n                    _,\n                    _,\n                ) = source.get_frames(\n                    trajectory,\n                    range(selected_start, selected_end + 1),\n                    subtask=subtask_name,\n                )\n                expected_frames = selected_end - selected_start + 1\n                if (\n                    selected_states is None\n                    or int(selected_observation["visual"].shape[0])\n                    != expected_frames\n                    or int(selected_states.shape[0]) != expected_frames\n                ):\n                    raise RuntimeError(\n                        f"RoboCasa {subtask_name} selected span changed length: "\n                        f"trajectory={trajectory}, start={selected_start}, "\n                        f"end={selected_end}, expected={expected_frames}, "\n                        f"visual={selected_observation['visual'].shape}, "\n                        f"states={None if selected_states is None else selected_states.shape}"\n                    )\n                source_id = str(trajectory_info.get("demo_key", trajectory))\n                identity = {\n                    "task": task.task_key,\n                    "source": source_id,\n                    "start": selected_start,\n                    "end": selected_end,\n                    "evaluation_position": position,\n                }\n                instances.append(\n                    EvaluationInstance(\n                        instance_id=canonical_sha256(identity)[:24],\n                        source_trajectory_id=source_id,\n                        source_trajectory_index=trajectory,\n                        segment_start=selected_start,\n                        segment_end=selected_end,\n                        initialization_fingerprint=array_sha256(\n                            selected_states[0].numpy()\n                        ),\n                        goal_fingerprint=array_sha256(\n                            selected_observation["visual"][-1].numpy()\n                        ),\n                        environment_seed=int(\n                            (seed * seed + position * seed) % (2**32 - 2)\n                        ),\n                        cem_seed=seed,\n                        appearance_seed=appearance_seed,\n                    )\n                )\n'''
    text = text[:start] + replacement + text[end:]

    text = replace_once(
        text,
        '                    "legacy_place_reuse_compatible": subtask_name == "place",\n',
        '                    "legacy_place_reuse_compatible": False,\n'
        '                    "fixed_goal_span_steps": (\n'
        '                        goal_span if goal_span > 0 else None\n'
        '                    ),\n',
        "robocasa.py: mark fixed-span manifest",
    )
    return text


def patch_suite_runner(text: str) -> str:
    import_anchor = '''from wm_adapter.experiments.cross_benchmark import (
    JobSpec,
    archive_incomplete,
    block_job_for_failed_dependencies,
    load_suite_config,
    phase_summary,
    run_gpu_phase,
)
'''
    import_new = import_anchor + '''from wm_adapter.planning.jepa_wm_planner import (
    EVALUATION_PROTOCOL_VERSION,
)
'''
    text = replace_once(
        text,
        import_anchor,
        import_new,
        "suite runner: import planning protocol",
    )

    manifest_start = text.find("def _validate_evaluation_manifest(")
    manifest_end = text.find("\ndef _validate_cache(", manifest_start)
    if manifest_start < 0 or manifest_end < 0:
        raise PatchError("suite runner: evaluation-manifest validator section not found")
    manifest_section = text[manifest_start:manifest_end]
    manifest_anchor = '''    if not isinstance(instances, list) or len(instances) < int(
        job.required_count or 0
    ):
        raise RuntimeError(
            f"Evaluation manifest has too few instances: {job.artifact_path}"
        )
'''
    manifest_new = manifest_anchor + '''    fixed_goal_span_steps = payload.get("fixed_goal_span_steps")
    if job.task == "robocasa_place":
        spans = [
            int(instance["segment_end"]) - int(instance["segment_start"])
            for instance in instances[: int(job.required_count or 0)]
        ]
        if (
            set(spans) != {25}
            or fixed_goal_span_steps != 25
            or bool(payload.get("legacy_place_reuse_compatible", False))
        ):
            raise RuntimeError(
                "RoboCasa Place evaluation manifest is not protocol-2.1 fixed-span: "
                f"spans={sorted(set(spans))}, "
                f"fixed_goal_span_steps={fixed_goal_span_steps}, "
                f"legacy={payload.get('legacy_place_reuse_compatible')}, "
                f"path={job.artifact_path}"
            )
'''
    manifest_section = replace_once(
        manifest_section,
        manifest_anchor,
        manifest_new,
        "suite runner: validate fixed Place span",
    )
    text = text[:manifest_start] + manifest_section + text[manifest_end:]

    planning_start = text.find("def _validate_planning(")
    planning_end = text.find("\ndef _validate(\n", planning_start)
    if planning_start < 0 or planning_end < 0:
        raise PatchError("suite runner: planning validator section not found")
    planning_section = text[planning_start:planning_end]

    expected_anchor = '''        "cache_fingerprint": str(cache["cache_fingerprint"]),
        "cache_file_sha256": str(cache["cache_file_sha256"]),
    }
    actual = {
'''
    expected_new = '''        "cache_fingerprint": str(cache["cache_fingerprint"]),
        "cache_file_sha256": str(cache["cache_file_sha256"]),
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
    }
    actual = {
'''
    planning_section = replace_once(
        planning_section,
        expected_anchor,
        expected_new,
        "suite runner: require protocol version",
    )

    actual_anchor = '''        "cache_fingerprint": payload.get("cache_fingerprint"),
        "cache_file_sha256": payload.get("cache_file_sha256"),
    }
    if actual != expected:
'''
    actual_new = '''        "cache_fingerprint": payload.get("cache_fingerprint"),
        "cache_file_sha256": payload.get("cache_file_sha256"),
        "evaluation_protocol_version": payload.get(
            "evaluation_protocol_version"
        ),
    }
    if actual != expected:
'''
    planning_section = replace_once(
        planning_section,
        actual_anchor,
        actual_new,
        "suite runner: read protocol version",
    )

    consistency_anchor = '''    if actual != expected:
        raise RuntimeError(
            f"Cross-backend planning contract mismatch: expected={expected}, actual={actual}"
        )
    cem = payload.get("cem", {})
'''
    consistency_new = '''    if actual != expected:
        raise RuntimeError(
            f"Cross-backend planning contract mismatch: expected={expected}, actual={actual}"
        )
    computed_success_count = sum(bool(value) for value in success)
    reported_success_count = int(payload.get("success_count", -1))
    reported_total = int(payload.get("total_episodes", -1))
    reported_rate = float(payload.get("success_rate", -1.0))
    expected_rate = computed_success_count / len(success)
    if (
        reported_success_count != computed_success_count
        or reported_total != len(success)
        or abs(reported_rate - expected_rate) > 1.0e-12
    ):
        raise RuntimeError(
            "Planning success summary is inconsistent with per_episode_success: "
            f"reported_count={reported_success_count}, "
            f"computed_count={computed_success_count}, "
            f"reported_total={reported_total}, total={len(success)}, "
            f"reported_rate={reported_rate}, expected_rate={expected_rate}"
        )
    cem = payload.get("cem", {})
'''
    planning_section = replace_once(
        planning_section,
        consistency_anchor,
        consistency_new,
        "suite runner: validate success summary",
    )
    text = text[:planning_start] + planning_section + text[planning_end:]
    return text


def patch_monitor_core(text: str) -> str:
    text = replace_once(
        text,
        '    artifact = entry.get("artifact")\n',
        '    artifact = entry.get("artifact_validation", entry.get("artifact"))\n',
        "monitor: read cross-backend artifact_validation",
    )
    text = replace_once(
        text,
        '        used = int(artifact.get("used_episodes", artifact.get("available_episodes", 0)))\n',
        '        used = int(\n'
        '            artifact.get(\n'
        '                "episodes",\n'
        '                artifact.get(\n'
        '                    "used_episodes",\n'
        '                    artifact.get("available_episodes", 0),\n'
        '                ),\n'
        '            )\n'
        '        )\n',
        "monitor: use planning episodes denominator",
    )
    return text


PATCHERS: dict[Path, Callable[[str], str]] = {
    PLAN_EVALUATOR: patch_plan_evaluator,
    PLANNING_UTILS: patch_planning_utils,
    JEPA_PLANNER: patch_jepa_planner,
    ROBOCASA_BENCHMARK: patch_robocasa_benchmark,
    SUITE_RUNNER: patch_suite_runner,
    MONITOR_CORE: patch_monitor_core,
}

MARKERS = {
    PLAN_EVALUATOR: "ROBOCASA_PLANNING_INPUT_CONTRACT",
    PLANNING_UTILS: "def _video_frame_to_uint8",
    JEPA_PLANNER: 'EVALUATION_PROTOCOL_VERSION = "2.1"',
    ROBOCASA_BENCHMARK: '"fixed_goal_span_steps"',
    SUITE_RUNNER: "computed_success_count = sum(bool(value) for value in success)",
    MONITOR_CORE: 'entry.get("artifact_validation", entry.get("artifact"))',
}


def locate_repo(value: str | None) -> Path:
    candidate = Path(value or os.getcwd()).expanduser().resolve()
    try:
        root = run(["git", "rev-parse", "--show-toplevel"], cwd=candidate).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise PatchError(f"Not inside a Git repository: {candidate}") from error
    return Path(root).resolve()


def read_targets(repo: Path) -> dict[Path, str]:
    values: dict[Path, str] = {}
    for relative in TARGETS:
        path = repo / relative
        if not path.is_file():
            raise PatchError(f"Required file is missing: {path}")
        values[relative] = path.read_text(encoding="utf-8")
    return values


def build_changes(repo: Path) -> list[FileChange]:
    originals = read_targets(repo)
    applied = {
        path: marker in originals[path]
        for path, marker in MARKERS.items()
    }
    if all(applied.values()):
        return []
    if any(applied.values()):
        partial = [str(path) for path, value in applied.items() if value]
        raise PatchError(
            "A partial fix is already present; refusing an ambiguous mixed state. "
            f"Marked files: {partial}"
        )
    changes = []
    for relative, patcher in PATCHERS.items():
        original = originals[relative]
        modified = patcher(original)
        if modified == original:
            raise PatchError(f"Patcher produced no change for {relative}")
        changes.append(FileChange(relative, original, modified))
    return changes


def backup(repo: Path, changes: list[FileChange]) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = repo / ".planning_fix_backups" / stamp
    root.mkdir(parents=True, exist_ok=False)
    for change in changes:
        destination = root / "original" / change.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(change.original, encoding="utf-8")
    status = run(["git", "status", "--short"], cwd=repo).stdout
    diff = run(["git", "diff", "--binary"], cwd=repo).stdout
    cached = run(["git", "diff", "--cached", "--binary"], cwd=repo).stdout
    (root / "git-status-before.txt").write_text(status, encoding="utf-8")
    (root / "git-diff-before.patch").write_text(diff, encoding="utf-8")
    (root / "git-diff-cached-before.patch").write_text(cached, encoding="utf-8")
    restore_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'REPO="${1:-' + str(repo) + '}"',
    ]
    for change in changes:
        restore_lines.append(
            f'install -D -m 0644 "$(dirname "$0")/original/{change.path}" '
            f'"$REPO/{change.path}"'
        )
    restore_lines.append('echo "Restored planning source files."')
    restore = root / "restore.sh"
    restore.write_text("\n".join(restore_lines) + "\n", encoding="utf-8")
    restore.chmod(0o755)
    return root


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.chmod(path.stat().st_mode)
    temporary.replace(path)


def verify(repo: Path) -> None:
    for relative, marker in MARKERS.items():
        content = (repo / relative).read_text(encoding="utf-8")
        if marker not in content:
            raise PatchError(f"Verification marker missing in {relative}: {marker}")
    compile_paths = [str(path) for path in TARGETS]
    run([sys.executable, "-m", "py_compile", *compile_paths], cwd=repo)
    run(["git", "diff", "--check", "--", *compile_paths], cwd=repo)


def write_preview(repo: Path, changes: list[FileChange], destination: Path) -> None:
    blocks: list[str] = []
    for change in changes:
        blocks.extend(
            difflib.unified_diff(
                change.original.splitlines(keepends=True),
                change.modified.splitlines(keepends=True),
                fromfile=f"a/{change.path}",
                tofile=f"b/{change.path}",
            )
        )
    destination.write_text("".join(blocks), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="Repository root or a path inside it")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Apply fixes (default)")
    mode.add_argument("--check", action="store_true", help="Validate applicability only")
    mode.add_argument(
        "--preview",
        metavar="PATCH",
        help="Write a unified diff without modifying the repository",
    )
    args = parser.parse_args()

    try:
        repo = locate_repo(args.repo)
        head = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
        print(f"Repository: {repo}")
        print(f"HEAD:       {head}")
        if head != EXPECTED_BASE:
            print(
                f"WARNING: expected baseline {EXPECTED_BASE}; exact source anchors "
                "will still be validated before any write.",
                file=sys.stderr,
            )
        changes = build_changes(repo)
        if not changes:
            verify(repo)
            print("Fix set is already applied and verified.")
            return 0
        print("Validated changes:")
        for change in changes:
            print(f"  - {change.path}")
        if args.check:
            print("Applicability check passed; no files were modified.")
            return 0
        if args.preview:
            destination = Path(args.preview).expanduser().resolve()
            write_preview(repo, changes, destination)
            print(f"Wrote preview patch: {destination}")
            return 0

        backup_root = backup(repo, changes)
        try:
            for change in changes:
                atomic_write(repo / change.path, change.modified)
            verify(repo)
        except Exception:
            for change in changes:
                original = backup_root / "original" / change.path
                shutil.copy2(original, repo / change.path)
            raise
        write_preview(repo, changes, backup_root / "applied.patch")
        for change in changes:
            destination = backup_root / "patched" / change.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(change.modified, encoding="utf-8")
        print(f"Applied and verified. Backup: {backup_root}")
        print(f"Restore command: {backup_root / 'restore.sh'} {repo}")
        print("Next: archive old Planning results and rebuild RoboCasa Place manifests.")
        return 0
    except (PatchError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stdout:
            print(error.stdout, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
