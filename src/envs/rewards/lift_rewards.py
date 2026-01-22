"""Working reward functions for lift cube task.

These are the reward functions that achieve 100% success rate:
- v11: State-based (SAC) - 100% success at 1M steps
- v19: Image-based (DrQ-v2) - 100% success at 2M steps

For historical/experimental reward versions, see _legacy_rewards.py.
"""

import numpy as np


def reward_v11(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V11: Dense reward for state-based training.

    Structure:
    - Reach reward (tanh distance)
    - Push-down penalty
    - Drop penalty
    - Grasp bonus + continuous lift reward
    - Binary lift bonus
    - Target height bonus
    - Action rate penalty (only when lifted)
    - Success bonus

    Achieved 100% success at 1M steps with SAC.
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Target height bonus (aligned with success: z > lift_height)
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty for smoothness (only when lifted, to not hinder lifting)
    if action is not None and cube_z > 0.06:
        action_delta = action - env._prev_action
        action_penalty = 0.01 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v19(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V19: Dense reward for image-based training.

    Key innovations over v11:
    - Per-finger reach reward (moving finger gets own reach gradient)
    - Stronger grasp bonus (1.5 vs 0.25)
    - Doubled lift coefficient (4.0 vs 2.0)
    - Threshold ramp from 0.04m to 0.08m
    - Hold count bonus (escalating reward for sustained height)

    Achieved 100% success at 2M steps with DrQ-v2.
    """
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    gripper_state = info["gripper_state"]
    is_grasping = info["is_grasping"]
    hold_count = info["hold_count"]
    is_closed = gripper_state < 0.25

    # Standard gripper reach (static finger is part of gripper frame)
    gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_cube)

    # Moving finger reach - only applies when gripper is close to cube
    reach_threshold = 0.7  # ~3cm from cube
    if gripper_reach < reach_threshold:
        reach_reward = gripper_reach
    else:
        if is_closed:
            moving_reach = 1.0
        else:
            moving_finger_pos = env._get_moving_finger_pos()
            moving_to_cube = np.linalg.norm(moving_finger_pos - cube_pos)
            moving_reach = 1.0 - np.tanh(10.0 * moving_to_cube)

        reach_reward = (gripper_reach + moving_reach) * 0.5

    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 1.5

        # Continuous lift reward (4.0x coefficient)
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 4.0

        # Binary lift bonus at 0.02m
        if cube_z > 0.02:
            reward += 1.0

        # Linear threshold ramp from 0.04m to 0.08m
        if cube_z > 0.04:
            threshold_progress = min(1.0, (cube_z - 0.04) / (env.lift_height - 0.04))
            reward += threshold_progress * 2.0

    # Target height bonus
    if cube_z > env.lift_height:
        reward += 1.0

        # Hold count bonus - escalating reward for sustained height
        reward += 0.5 * hold_count

    # Action rate penalty during hold phase
    if action is not None and cube_z > env.lift_height and hold_count > 0:
        action_delta = action - env._prev_action
        action_penalty = 0.02 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v20(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V20: Pick-and-place reward function.

    Extends v11/v19 with transport and placement phases:
    - Phase 1: Reach to cube (same as v11)
    - Phase 2: Grasp and lift (same as v11)
    - Phase 3: Transport to target (cube XY → target XY while lifted)
    - Phase 4: Lower and release at target

    Task structure:
    1. Move gripper to cube
    2. Close gripper to grasp
    3. Lift cube to transport height
    4. Move cube over target location
    5. Lower cube to table
    6. Open gripper to release

    Success: cube within 2cm of target XY, on table, not grasped.
    """
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Get place target info (required for this reward)
    place_target = info.get("place_target")
    cube_to_target = info.get("cube_to_target", float('inf'))
    is_placed = info.get("is_placed", False)

    # ========== PHASE 1: REACH ==========
    # Reward for moving gripper close to cube
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty (don't knock cube off table)
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # ========== PHASE 2: GRASP AND LIFT ==========
    # Drop penalty - but only if far from target (intentional release near target is ok)
    if was_grasping and not is_grasping:
        if cube_to_target > 0.04:  # Dropped far from target
            reward -= 3.0
        elif cube_z > 0.03:  # Dropped at height (not placing)
            reward -= 1.5

    if is_grasping:
        # Grasp bonus
        reward += 0.5

        # Continuous lift reward
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

        # Binary lift bonus
        if cube_z > 0.03:
            reward += 1.0

        # Target height bonus (ready to transport)
        if cube_z > env.lift_height:
            reward += 1.0

    # ========== PHASE 3: TRANSPORT ==========
    # Reward for moving cube toward target while lifted
    if place_target is not None and cube_z > env.lift_height * 0.7:
        # Transport reward - cube XY approaching target XY
        transport_reward = 1.0 - np.tanh(5.0 * cube_to_target)
        reward += transport_reward * 1.5

        # Bonus for reaching target zone while grasping and lifted
        if cube_to_target < 0.04 and is_grasping:
            reward += 2.0  # At target, ready to place

    # ========== PHASE 4: LOWER AND RELEASE ==========
    if place_target is not None and cube_to_target < 0.05:
        # Near target - reward for lowering
        if is_grasping:
            # Reward for lowering cube while at target
            if cube_z < env.lift_height:
                lower_progress = (env.lift_height - cube_z) / (env.lift_height - 0.015)
                reward += lower_progress * 2.0

        # Release reward - cube on table at target, gripper open
        if not is_grasping and cube_z < 0.025:
            reward += 3.0  # Released at target

    # ========== SMOOTHNESS PENALTY ==========
    if action is not None and cube_z > 0.04:
        action_delta = action - env._prev_action
        action_penalty = 0.01 * np.sum(action_delta**2)
        reward -= action_penalty

    # ========== SUCCESS BONUS ==========
    if is_placed:
        reward += 15.0

    return reward
