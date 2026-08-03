# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# The below code is inspired from TD-MPC2 https://github.com/nicklashansen/tdmpc2 and https://github.com/ARISE-Initiative/robosuite
# both licensed under the MIT License

import logging
import os
import re
import sys
import time

import gym
import numpy as np
import robocasa
import robosuite
from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS
from robocasa.utils.env_utils import create_env
from scipy.spatial.transform import Rotation as R

from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit

ROBOCASA_ASSET_ROOT_PATH = os.path.join(
    os.path.dirname(robocasa.__file__), "models", "assets"
)
BASE_ASSET_ROOT_PATH = os.path.join(ROBOCASA_ASSET_ROOT_PATH, "objects")

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

RCASA_CONTROLLER_INPUT_LIMS = np.array([1.0, -1])
RCASA_CONTROLLER_OUTPUT_LIMS = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def _robot_diagnostics(env):
    robots = getattr(env, "robots", None)
    if not robots:
        return {
            "robot_class": "unavailable",
            "robot_model_class": "unavailable",
            "gripper_class": "unavailable",
        }

    robot = robots[0]
    robot_model = getattr(robot, "robot_model", None)
    gripper = getattr(robot, "gripper", None)
    if isinstance(gripper, dict):
        grippers = [value for value in gripper.values() if value is not None]
    elif isinstance(gripper, (list, tuple)):
        grippers = [value for value in gripper if value is not None]
    elif gripper is None:
        grippers = []
    else:
        grippers = [gripper]

    gripper_classes = sorted({type(value).__name__ for value in grippers})
    if gripper_classes:
        gripper_class = ",".join(gripper_classes)
    else:
        gripper_class = str(
            getattr(robot, "gripper_type", getattr(robot_model, "default_gripper", "unavailable"))
        )
    return {
        "robot_class": type(robot).__name__,
        "robot_model_class": type(robot_model).__name__ if robot_model is not None else "unavailable",
        "gripper_class": gripper_class,
    }


class RoboCasaWrapper(gym.Wrapper):
    """
    Wrapper for RoboCasa environments.
    """

    def __init__(self, env, cfg=None, env_name="PnPCounterToSink", camera_name="robot0_agentview_left"):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.rescale_act_droid_to_rcasa = cfg.task_specification.env.get("rescale_act_droid_to_rcasa", False)
        logger.info(f"RoboCasaWrapper: {self.rescale_act_droid_to_rcasa=}")
        self.custom_task = cfg.task_specification.env.get("custom_task", False)
        self.subtask = cfg.task_specification.env.get("subtask", None)
        self.goal_obj_pos = None
        self.env_name = env_name
        self.camera_name = camera_name  # default camera name working with the underlying robosuite env
        self.custom_camera_name = self.camera_name
        self.camera_width = self.env.camera_widths[0]
        self.camera_height = self.env.camera_heights[0]
        self.full_action_dim = self.env.action_dim  # 12: 7 for arm, 5 for base navigation
        self.manip_only = cfg.task_specification.env.get("manip_only", True)
        self.action_dim = 7 if self.manip_only else self.full_action_dim
        self.action_space = gym.spaces.Box(
            low=np.full(self.action_dim, -1.0), high=np.full(self.action_dim, 1.0), dtype=np.float32
        )
        self.reach_threshold = cfg.task_specification.env.get("reach_threshold", 0.2)
        self.place_threshold = cfg.task_specification.env.get("place_threshold", 0.15)
        logger.info(f"Set {self.reach_threshold=} and {self.place_threshold=}")
        if self.custom_task:
            self.custom_camera_name = "robot0_droid_agentview_left"  # "robot0_leftview"
            self.custom_camera_pos = [0.4, 0.4, 0.6]
            self.custom_camera_quat = [0.0, -0.0, 0.6, 1.0]
            self.custom_camera_fovy = 85

    @property
    def spec(self):
        """Return None to avoid bug when wrapping in TimeLimit."""
        return None

    def eef_quat_to_xyz(self, eef_quat):
        # shape (4,)
        # If your quaternion is [w, x, y, z], convert to [x, y, z, w] for scipy
        eef_quat_xyzw = np.array([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
        # Convert to Euler angles (xyz order, radians)
        eef_euler = R.from_quat(eef_quat_xyzw).as_euler("xyz", degrees=False)
        return eef_euler  # shape (3,)

    def gripper_2d_to_1d(self, gripper_qpos):
        """
        Convert 2D gripper position to 1D representation.

        Args:
            gripper_qpos: Array of shape (2,) for gripper position.

        Returns:
            Array of shape (1,) for gripper state.
        """
        return gripper_qpos[0:1] - gripper_qpos[1:2]

    def get_obs_proprio_succ_from_info(self, info):
        """
        Extract proprioceptive state and success info from environment info dict.

        The observation part is not used here; visual data reaches the PixelWrapper
        via the render() function.

        Args:
            info: Environment info dictionary containing robot state.

        Returns:
            Tuple of (obs, info) where info contains proprio and success keys.
        """
        obs = np.random.randn(1)  # Dummy observation, not used
        eef_angle = self.eef_quat_to_xyz(info["robot0_eef_quat"])
        gripper_closure = self.gripper_2d_to_1d(info["robot0_gripper_qpos"])
        info["proprio"] = np.concatenate(
            [
                info["robot0_eef_pos"],  # Cartesian position (3,)
                eef_angle,  # Euler angles (3,)
                gripper_closure,  # Gripper state (1,)
            ]
        )
        # Need to call _check_success() to define env.obj_up_once
        # and other variables used in subtask_success()
        info["success"] = self.env._check_success()
        if self.subtask is not None:
            info = self.subtask_success(info)
        return obs, info

    def subtask_success(self, info):
        """
        Evaluate success for specific subtasks (reach, pick, place, or combinations).

        Args:
            info: Environment info dictionary.

        Returns:
            Updated info dictionary with success and metric fields.
        """
        obj = self.env.objects["obj"]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        hand_pos = np.array(
            self.sim.data.body_xpos[self.sim.model.body_name2id(self.robots[0].gripper["right"].root_body)]
        )
        hand_obj_dist = np.linalg.norm(hand_pos - obj_pos)
        reach = hand_obj_dist < self.reach_threshold
        obj_goal_dist = np.linalg.norm(self.goal_obj_pos - obj_pos) if self.goal_obj_pos is not None else -1.0
        place = obj_goal_dist < self.place_threshold

        if self.subtask == "reach-pick-place":
            success = place
        elif self.subtask == "reach-pick":
            success = reach and self.env.obj_up_once
        elif self.subtask == "pick-place":
            success = self.env.obj_up_once and place
        elif self.subtask == "reach":
            success = reach
        elif self.subtask == "pick":
            success = self.env.obj_up_once
        elif self.subtask == "place":
            success = place
        else:
            raise ValueError(f"Unknown subtask: {self.subtask}")

        info["success"] = success
        info["obj_pos"] = obj_pos
        info["hand_pos"] = hand_pos
        info["obj_goal_dist"] = obj_goal_dist
        info["hand_obj_dist"] = hand_obj_dist
        info["obj_initial_height"] = self.env.obj_initial_height if hasattr(self.env, "obj_initial_height") else -1
        info["obj_lift"] = obj_pos[2] - info["obj_initial_height"]
        info["near_object"] = hand_obj_dist
        info["obj_up_once"] = self.env.obj_up_once if hasattr(self.env, "obj_up_once") else -1
        return info

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation.
        """
        info = self.env.reset()
        return self.get_obs_proprio_succ_from_info(info)

    def step(self, action):
        """
        Perform a step in the environment.
        action: np array of shape (action_dim,)
        """
        if self.manip_only:
            # If we're only controlling the arm, pad the action with zeros for the base nav
            full_action = np.zeros(self.full_action_dim)
            full_action[:7] = action
        else:
            full_action = action

        scaled_action = full_action.copy()
        if self.rescale_act_droid_to_rcasa:
            scaled_action[:7] = full_action[:7] * RCASA_CONTROLLER_INPUT_LIMS[0] / RCASA_CONTROLLER_OUTPUT_LIMS

        info, reward, done, _ = self.env.step(scaled_action)
        obs, info = self.get_obs_proprio_succ_from_info(info)
        if info["success"]:
            logger.info("RoboCasaWrapper: Task success detected in step()")
        return obs, reward, None, done, info

    def render(self, *args, **kwargs):
        """
        Render the environment using the specified camera.
        Returns: H W 3
        Making a deepcopy is essential to avoid race conditions or corrupted images
        when the underlying simulator updates the visual buffer asynchronously
        """
        if self.custom_camera_name in self.env.sim.model._camera_name2id.keys():
            camera_to_use = self.custom_camera_name
        else:
            camera_to_use = self.camera_name
        logger.info(f"Using camera: {camera_to_use}")
        result = self.env.sim.render(
            height=self.camera_height, width=self.camera_width, camera_name=camera_to_use
        ).copy()
        if camera_to_use != "robot0_rightview":
            result = result[::-1]  # flip vertically
        else:
            result = result[:, ::-1]  # flip horizontally
        return result

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def update_env(self, env_info):
        """Update environment configuration (placeholder for interface compatibility)."""
        pass

    def prepare(self, seed, init_state, env_info=None):
        """
        Reset environment with controlled initial state.

        Inspired by robocasa/robocasa/utils/robomimic/robomimic_env_wrapper.py.

        Args:
            seed: Random seed for reproducibility.
            init_state: Initial simulator state.
            env_info: Optional dict containing model_xml and ep_meta.

        Returns:
            Tuple of (obs, info).
        """
        prep_start_time = time.time()
        self.seed(seed)
        model_xml = env_info.get("model_xml", None)
        ep_meta = env_info.get("ep_meta", None)
        # # Uncomment to save out the model XML for debugging
        # xml_path = f"evals/simu_env_planning/envs/robocasa/{self.env_name}_model.xml"
        # with open(xml_path, "w") as f:
        #     f.write(model_xml)
        #     logger.info(f"Saved model XML to {xml_path}")
        if self.custom_task:
            if model_xml is None:
                raise RuntimeError("Custom RoboCasa camera injection requires a non-empty model_xml")
            # Modify the XML to add the custom camera
            import xml.etree.ElementTree as ET

            tree = ET.ElementTree(ET.fromstring(model_xml))
            camera_container = tree.find(".//body[@name='base0_support']")
            if camera_container is None:
                raise RuntimeError(
                    "Custom RoboCasa camera injection could not find body "
                    "named 'base0_support' in model_xml"
                )

            camera_elem = ET.SubElement(camera_container, "camera")
            camera_elem.set("name", self.custom_camera_name)
            camera_elem.set("pos", " ".join(map(str, self.custom_camera_pos)))
            camera_elem.set("quat", " ".join(map(str, self.custom_camera_quat)))
            camera_elem.set("fovy", str(self.custom_camera_fovy))
            camera_elem.set("mode", "fixed")

            model_xml = ET.tostring(tree.getroot(), encoding="unicode")

        if model_xml is not None:
            if ep_meta is not None:
                ep_meta["object_cfgs"] = update_mjcf_paths(ep_meta["object_cfgs"])
                if hasattr(self.env, "set_attrs_from_ep_meta"):
                    self.env.set_attrs_from_ep_meta(ep_meta)
                elif hasattr(self.env, "set_ep_meta"):
                    self.env.set_ep_meta(ep_meta)

            logger.info("Resetting from provided model XML")
            xml = _prepare_xml(self.env, model_xml)
            xml = path_change(xml)
            logger.info(
                "Loading external RoboCasa model XML with stale dummy task-model "
                "ID mappings disabled"
            )
            self.env._external_model_xml_replay = True
            self.env._skip_model_id_mappings_on_external_xml_reset = True
            try:
                self.env.reset_from_xml_string(xml)
            finally:
                self.env._skip_model_id_mappings_on_external_xml_reset = False
            self.env.sim.reset()
            logger.info("Finished resetting from provided model XML")
        else:
            self.env._external_model_xml_replay = False
            self.reset()

        flattened_init_state = np.asarray(init_state).reshape(-1)
        model = self.env.sim.model
        model_nq = int(model.nq)
        model_nv = int(model.nv)
        model_na = int(model.na)
        expected_state_length = 1 + model_nq + model_nv
        diagnostics = _robot_diagnostics(self.env)
        state_context = (
            f"received state length={flattened_init_state.size}, "
            f"expected state length={expected_state_length}, model.nq={model_nq}, "
            f"model.nv={model_nv}, model.na={model_na}, env_name={self.env_name}, "
            f"robot_class={diagnostics['robot_class']}, "
            f"gripper_class={diagnostics['gripper_class']}"
        )
        if model_na != 0:
            raise RuntimeError(
                "RoboCasa simulator state restore requires model.na=0 for MjSimState.from_flattened: "
                f"{state_context}"
            )
        if flattened_init_state.size != expected_state_length:
            raise RuntimeError(f"RoboCasa simulator state length mismatch: {state_context}")

        try:
            self.env.sim.set_state_from_flattened(flattened_init_state)
        except Exception as error:
            raise RuntimeError(f"Failed to restore RoboCasa simulator state: {state_context}") from error
        self.env.sim.forward()

        if hasattr(self.env, "update_sites"):
            self.env.update_sites()
        if hasattr(self.env, "update_state"):
            self.env.update_state()

        if hasattr(self.env, "_get_observation"):
            current_info = self.env._get_observation()
        elif hasattr(self.env, "_get_observations"):
            current_info = self.env._get_observations(force_update=True)
        else:
            raise RuntimeError(
                "RoboCasa environment cannot provide observations after state restore"
            )
        obs, info = self.get_obs_proprio_succ_from_info(current_info)

        logger.info(f"robocasa env.prepare() took {time.time() - prep_start_time:.2f} seconds")
        return obs, info

    @property
    def unwrapped(self):
        return self.env


def make_env(cfg):
    """
    Create a RoboCasa environment and wrap it with RoboCasaWrapper.

    Args:
        cfg: Configuration object containing task and environment settings.

    Returns:
        Wrapped RoboCasa environment with TimeLimit.
    """
    env_name = cfg.task_specification.task.split("-", 1)[-1]
    all_tasks = list(SINGLE_STAGE_TASK_DATASETS.keys()) + list(MULTI_STAGE_TASK_DATASETS.keys()) + ["PnPCounterTop"]

    if not cfg.task_specification.task.startswith("robocasa-") or env_name not in all_tasks:
        raise ValueError("Unknown task:", cfg.task_specification.task)
    robot = cfg.task_specification.env.get("robots", "PandaOmron")
    is_droid = cfg.task_specification.env.get("rescale_act_droid_to_rcasa", False)
    default_gripper = "Robotiq85Gripper" if is_droid else "default"
    gripper_types = cfg.task_specification.env.get("gripper_types", default_gripper)
    external_xml_dummy = cfg.task_specification.get("goal_source") == "dset"
    logger.info(
        "Creating RoboCasa environment with robot=%s, gripper_types=%s, "
        "is_droid=%s, external_xml_dummy=%s",
        robot,
        gripper_types,
        is_droid,
        external_xml_dummy,
    )
    # Dummy env that is later modified in RobocasaWrapper.prepare()
    # logger.info(f"Creating dummy RoboCasa PnPSinkToCounter..")
    env = create_env(
        env_name=env_name,  # e.g. "PnPSinkToCounter",
        robots=robot,
        gripper_types=gripper_types,
        camera_names=["robot0_leftview"],
        camera_widths=cfg.task_specification.img_size,
        camera_heights=cfg.task_specification.img_size,
        seed=cfg.meta.seed,
        render_onscreen=False,
        external_xml_dummy=external_xml_dummy,
    )
    env = RoboCasaWrapper(
        env, cfg, env_name, camera_name=cfg.task_specification.env.get("camera_name", "robot0_agentview_left")
    )
    model = getattr(getattr(env.env, "sim", None), "model", None)
    diagnostics = _robot_diagnostics(env.env)
    logger.info(
        "RoboCasa environment compatibility: env.action_dim=%s, manip_only=%s, "
        "wrapper.action_dim=%s, sim.model.nq=%s, sim.model.nv=%s, sim.model.na=%s, "
        "robot_model_class=%s, gripper_class=%s",
        getattr(env.env, "action_dim", "unavailable"),
        getattr(env, "manip_only", "unavailable"),
        getattr(env, "action_dim", "unavailable"),
        getattr(model, "nq", "unavailable"),
        getattr(model, "nv", "unavailable"),
        getattr(model, "na", "unavailable"),
        diagnostics["robot_model_class"],
        diagnostics["gripper_class"],
    )
    logger.info("Wrapped RoboCasa environment with RoboCasaWrapper")
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env


def update_mjcf_paths(object_cfgs):
    """
    Update mjcf_path in object_cfgs by replacing source paths with local paths.

    Args:
        object_cfgs: List of object configuration dicts containing 'info' with 'mjcf_path'.

    Returns:
        list: Updated object_cfgs with modified mjcf_path.
    """
    for i, object_cfg in enumerate(object_cfgs):
        path = object_cfg["info"]["mjcf_path"]
        models_index = path.find("objects")
        relative_path = path[models_index:]  # e.g. 'models/assets/objects/aigen_objs/apple/apple_5/model.xml'
        full_local_path = os.path.join(BASE_ASSET_ROOT_PATH, relative_path[len("objects/") :])
        object_cfgs[i]["info"]["mjcf_path"] = full_local_path
    return object_cfgs


def path_change(xml_string):
    """
    Fix absolute file paths in the MJCF XML by replacing them with local paths
    rooted at BASE_ASSET_ROOT_PATH.
    """

    def replace_path(match):
        original_path = match.group(1)
        asset_marker = "robocasa/models/assets/"
        normalized_path = original_path.replace("\\", "/")
        asset_index = normalized_path.find(asset_marker)
        if asset_index != -1:
            relative_path = normalized_path[asset_index + len(asset_marker) :]
            new_path = os.path.normpath(
                os.path.join(ROBOCASA_ASSET_ROOT_PATH, relative_path)
            )
            return f'file="{new_path}"'

        model_index = original_path.find("objects/")
        if model_index == -1:
            return f'file="{original_path}"'

        relative_path = original_path[model_index + len("objects/") :]
        new_path = os.path.join(BASE_ASSET_ROOT_PATH, relative_path)
        new_path = os.path.normpath(new_path)

        return f'file="{new_path}"'

    updated_xml = re.sub(r'file="([^"]+)"', replace_path, xml_string)
    return updated_xml


def _prepare_xml(env, model_xml):
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        return postprocess_model_xml(model_xml)
    else:
        return env.edit_model_xml(model_xml)
