"""Pick and Place environment.

Extends lift_cube with transport and placement phases.
Agent must: reach → grasp → lift → transport → lower → release.
"""
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.controllers.ik_controller import IKController
from src.envs.rewards import REWARD_FUNCTIONS


class PickPlaceEnv(gym.Env):
    """Pick and Place with Cartesian action space.

    Task: Pick up cube, transport to target location, and place it down.

    Action space (4 dims):
        - Delta X, Y, Z for end-effector position
        - Gripper open/close (-1 to 1)

    Observation space (24 dims):
        - Joint positions (6)
        - Joint velocities (6)
        - Gripper position (3)
        - Gripper orientation (3) - euler angles
        - Cube position (3)
        - Target position (3)

    Success: Cube within 2cm of target XY, on table (z < 2.5cm), gripper released.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 400,
        action_scale: float = 0.02,
        lift_height: float = 0.08,
        reward_type: str = "dense",
        reward_version: str = "v20",
        curriculum_stage: int = 3,
        place_target: tuple[float, float] = (0.35, 0.10),
        target_radius: float = 0.02,  # Success radius in XY
        randomize_cube: bool = True,
        randomize_target: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.lift_height = lift_height
        self.reward_type = reward_type
        self.reward_version = reward_version
        self.curriculum_stage = curriculum_stage
        self.place_target = place_target
        self.target_radius = target_radius
        self.randomize_cube = randomize_cube
        self.randomize_target = randomize_target

        self._step_count = 0
        self._was_grasping = False
        self._prev_action = np.zeros(4)
        self._place_target_pos = None

        # Load model
        scene_path = Path(__file__).parent.parent.parent / "models/so101/pick_place.xml"
        if not scene_path.exists():
            # Fallback to lift_cube.xml
            scene_path = Path(__file__).parent.parent.parent / "models/so101/lift_cube.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        # Geom IDs for contact detection
        self._static_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "static_finger_pad")
        self._moving_pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_finger_pad")
        self._cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        # IK controller
        self.ik = IKController(self.model, self.data, end_effector_site="gripperframe")

        # Joint info
        self.n_joints = 6
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Action space: delta XYZ + gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space: joints + gripper + cube + target
        obs_dim = 6 + 6 + 3 + 3 + 3 + 3  # 24 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Target EE position (updated each step)
        self._target_ee_pos = None

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._renderer = mujoco.Renderer(self.model)

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[:self.n_joints].copy()
        joint_vel = self.data.qvel[:self.n_joints].copy()
        gripper_pos = self.ik.get_ee_position()
        gripper_mat = self.ik.get_ee_orientation()
        gripper_euler = self._rotation_matrix_to_euler(gripper_mat)
        cube_pos = self.data.sensor("cube_pos").data.copy()

        return np.concatenate([
            joint_pos,
            joint_vel,
            gripper_pos,
            gripper_euler,
            cube_pos,
            self._place_target_pos,
        ]).astype(np.float32)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        return np.array([roll, pitch, yaw])

    def _get_gripper_state(self) -> float:
        gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        gripper_qpos_addr = self.model.jnt_qposadr[gripper_joint_id]
        return self.data.qpos[gripper_qpos_addr]

    def _check_cube_contacts(self) -> tuple[bool, bool]:
        """Check if cube contacts both finger pads."""
        has_static = False
        has_moving = False
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact[i].geom1, self.data.contact[i].geom2
            other = None
            if g1 == self._cube_geom_id:
                other = g2
            elif g2 == self._cube_geom_id:
                other = g1
            if other == self._static_pad_geom_id:
                has_static = True
            if other == self._moving_pad_geom_id:
                has_moving = True
        return has_static, has_moving

    def _is_grasping(self) -> bool:
        gripper_state = self._get_gripper_state()
        is_closed = gripper_state < 0.25
        has_static, has_moving = self._check_cube_contacts()
        return is_closed and has_static and has_moving

    def _get_moving_finger_pos(self) -> np.ndarray:
        return self.data.geom_xpos[self._moving_pad_geom_id].copy()

    def _get_info(self) -> dict[str, Any]:
        gripper_pos = self.ik.get_ee_position()
        cube_pos = self.data.sensor("cube_pos").data.copy()
        cube_z = cube_pos[2]
        is_grasping = self._is_grasping()

        # Distance metrics
        gripper_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        cube_to_target_xy = np.linalg.norm(cube_pos[:2] - self._place_target_pos[:2])

        # Success conditions
        cube_at_target = cube_to_target_xy < self.target_radius
        cube_on_table = cube_z < 0.025
        is_placed = cube_at_target and cube_on_table and not is_grasping

        has_static, has_moving = self._check_cube_contacts()

        return {
            "gripper_to_cube": gripper_to_cube,
            "cube_pos": cube_pos.copy(),
            "cube_z": cube_z,
            "gripper_pos": gripper_pos.copy(),
            "gripper_state": self._get_gripper_state(),
            "has_gripper_contact": has_static,
            "has_jaw_contact": has_moving,
            "is_grasping": is_grasping,
            "is_lifted": is_grasping and cube_z > self.lift_height,
            "cube_to_target": cube_to_target_xy,
            "place_target": self._place_target_pos.copy(),
            "is_placed": is_placed,
            "is_success": is_placed,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Get cube joint
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        # Randomize cube position
        if self.randomize_cube and self.np_random is not None:
            cube_x = 0.25 + self.np_random.uniform(-0.03, 0.03)
            cube_y = 0.0 + self.np_random.uniform(-0.03, 0.03)
        else:
            cube_x, cube_y = 0.25, 0.0
        self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = [cube_x, cube_y, 0.015]
        self.data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]

        # Randomize target position
        if self.randomize_target and self.np_random is not None:
            target_x = self.place_target[0] + self.np_random.uniform(-0.02, 0.02)
            target_y = self.place_target[1] + self.np_random.uniform(-0.02, 0.02)
        else:
            target_x, target_y = self.place_target
        self._place_target_pos = np.array([target_x, target_y, 0.015])

        # Setup arm based on curriculum stage
        if self.curriculum_stage >= 3:
            # Gripper near cube, open
            self._reset_gripper_near_cube(cube_qpos_addr)
        else:
            # Top-down config
            self.data.qpos[3] = np.pi / 2
            self.data.qpos[4] = np.pi / 2
            self.data.ctrl[3] = np.pi / 2
            self.data.ctrl[4] = np.pi / 2

        mujoco.mj_forward(self.model, self.data)

        # Initialize target EE position
        self._target_ee_pos = self.ik.get_ee_position().copy()

        self._step_count = 0
        self._was_grasping = False
        self._prev_action = np.zeros(4)

        return self._get_obs(), self._get_info()

    def _reset_gripper_near_cube(self, cube_qpos_addr: int):
        """Position gripper above cube, open, ready to grasp."""
        height_offset = 0.03
        gripper_open = 0.3
        grasp_z_offset = 0.005
        finger_width_offset = -0.015

        # Top-down config
        self.data.qpos[3] = np.pi / 2
        self.data.qpos[4] = np.pi / 2
        self.data.ctrl[3] = np.pi / 2
        self.data.ctrl[4] = np.pi / 2
        mujoco.mj_forward(self.model, self.data)

        # Let cube settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Get cube position
        cube_pos = self.data.qpos[cube_qpos_addr:cube_qpos_addr + 3].copy()

        # Target above cube
        above_pos = cube_pos.copy()
        above_pos[1] += finger_width_offset
        above_pos[2] = cube_pos[2] + grasp_z_offset + height_offset

        # Move to position
        for _ in range(100):
            ctrl = self.ik.step_toward_target(
                above_pos, gripper_action=gripper_open, gain=0.5, locked_joints=[3, 4]
            )
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)
        delta_xyz = action[:3] * self.action_scale
        gripper_action = action[3]

        # Update target position
        self._target_ee_pos += delta_xyz

        # Clamp to workspace
        self._target_ee_pos[0] = np.clip(self._target_ee_pos[0], 0.1, 0.5)
        self._target_ee_pos[1] = np.clip(self._target_ee_pos[1], -0.3, 0.3)
        self._target_ee_pos[2] = np.clip(self._target_ee_pos[2], 0.01, 0.4)

        # IK control - agent controls gripper for release
        ctrl = self.ik.step_toward_target(
            self._target_ee_pos,
            gripper_action=gripper_action,
            gain=0.5,
            locked_joints=[4],  # Lock wrist_roll
        )
        ctrl[4] = np.pi / 2

        # Step simulation
        self.data.ctrl[:] = ctrl
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        info = self._get_info()

        # Compute reward
        reward = self._compute_reward(info, was_grasping=self._was_grasping, action=action)

        # Update state
        self._was_grasping = info["is_grasping"]
        self._prev_action = action.copy()

        terminated = info["is_success"]
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, info: dict[str, Any], was_grasping: bool = False, action: np.ndarray | None = None) -> float:
        if self.reward_type == "sparse":
            return 10.0 if info["is_success"] else 0.0

        reward_fn = REWARD_FUNCTIONS.get(self.reward_version)
        if reward_fn is None:
            raise ValueError(f"Unknown reward version: {self.reward_version}")
        return reward_fn(self, info, was_grasping=was_grasping, action=action)

    def render(self, camera: str = "wide") -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.30, 0.05, 0.05]
            cam.distance = 0.8
            cam.azimuth = 135
            cam.elevation = -25
            self._renderer.update_scene(self.data, camera=cam)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
