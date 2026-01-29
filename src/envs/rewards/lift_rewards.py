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
    """V20: Pick-and-place reward function (legacy - has local optima issues)."""
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    place_target = info.get("place_target")
    cube_to_target = info.get("cube_to_target", float('inf'))
    is_placed = info.get("is_placed", False)

    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    if was_grasping and not is_grasping:
        if cube_to_target > 0.04:
            reward -= 3.0
        elif cube_z > 0.03:
            reward -= 1.5

    if is_grasping:
        reward += 0.5
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0
        if cube_z > 0.03:
            reward += 1.0
        if cube_z > env.lift_height:
            reward += 1.0

    if place_target is not None and cube_z > env.lift_height * 0.7:
        transport_reward = 1.0 - np.tanh(5.0 * cube_to_target)
        reward += transport_reward * 1.5
        if cube_to_target < 0.04 and is_grasping:
            reward += 2.0

    if place_target is not None and cube_to_target < 0.05:
        if is_grasping:
            if cube_z < env.lift_height:
                lower_progress = (env.lift_height - cube_z) / (env.lift_height - 0.015)
                reward += lower_progress * 2.0
        if not is_grasping and cube_z < 0.025:
            reward += 3.0

    if action is not None and cube_z > 0.04:
        action_delta = action - env._prev_action
        action_penalty = 0.01 * np.sum(action_delta**2)
        reward -= action_penalty

    if is_placed:
        reward += 15.0

    return reward


def reward_v21(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V21: Improved pick-and-place reward with better grasp incentives.
    
    Key improvements over v20:
    - Strong gripper closing incentive when near cube (prevents pushing)
    - Time penalty for not grasping (forces early grasp)
    - Gated rewards: transport/place rewards REQUIRE grasping
    - Clearer phase transitions with larger bonuses
    - Cube height tracking to prevent pushing off table
    
    Reward structure:
    - Phase 1 (Reach): Move gripper to cube, close gripper when close
    - Phase 2 (Grasp): Secure grasp, large bonus for contact
    - Phase 3 (Lift): Lift cube to transport height
    - Phase 4 (Transport): Move lifted cube to target
    - Phase 5 (Place): Lower and release at target
    """
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    gripper_state = info.get("gripper_state", 1.0)  # 1.0 = open, 0.0 = closed
    
    place_target = info.get("place_target")
    cube_to_target = info.get("cube_to_target", float('inf'))
    is_placed = info.get("is_placed", False)
    step_count = getattr(env, '_step_count', 0)
    
    # ========== PHASE 1: REACH ==========
    # Dense reach reward - but capped to prevent farming
    reach_reward = 1.0 - np.tanh(8.0 * gripper_to_cube)
    reward += reach_reward * 0.5  # Reduced weight
    
    # CRITICAL: Incentive to close gripper when near cube
    if gripper_to_cube < 0.05:  # Within 5cm
        # Reward gripper closing (gripper_state: 1=open, 0=closed)
        close_reward = (1.0 - gripper_state) * 1.5
        reward += close_reward
        
        # Extra bonus for very close + closing
        if gripper_to_cube < 0.03:
            reward += (1.0 - gripper_state) * 1.0
    
    # Push-down penalty - cube should stay on table
    if cube_z < 0.005:
        reward -= 2.0
    elif cube_z < 0.01:
        reward -= (0.01 - cube_z) * 100.0
    
    # ========== PHASE 2: GRASP ==========
    if is_grasping:
        # Large grasp bonus - this is critical!
        reward += 3.0
        
        # ========== PHASE 3: LIFT ==========
        # Continuous lift reward
        lift_progress = max(0, cube_z - 0.015) / max(0.001, env.lift_height - 0.015)
        lift_progress = min(1.0, lift_progress)  # Clamp to [0, 1]
        reward += lift_progress * 3.0
        
        # Milestone bonuses for lifting
        if cube_z > 0.02:
            reward += 1.0
        if cube_z > 0.04:
            reward += 1.0
        if cube_z > env.lift_height:
            reward += 2.0  # Transport ready bonus
        
        # ========== PHASE 4: TRANSPORT ==========
        if place_target is not None and cube_z > env.lift_height * 0.6:
            # Transport reward - only when lifted and grasping
            transport_reward = 1.0 - np.tanh(5.0 * cube_to_target)
            reward += transport_reward * 2.0
            
            # At target zone bonus
            if cube_to_target < 0.04:
                reward += 3.0
            if cube_to_target < 0.02:
                reward += 2.0
        
        # ========== PHASE 5: LOWER ==========
        if place_target is not None and cube_to_target < 0.05:
            # At target - reward for lowering
            if cube_z < env.lift_height and cube_z > 0.015:
                lower_progress = 1.0 - (cube_z - 0.015) / max(0.001, env.lift_height - 0.015)
                reward += lower_progress * 2.0
    
    else:
        # NOT GRASPING penalties/incentives
        
        # Time penalty for not grasping (encourages quick grasp)
        if step_count > 50:  # Give some time to reach
            time_penalty = min(0.5, (step_count - 50) * 0.005)
            reward -= time_penalty
        
        # If was grasping and dropped
        if was_grasping:
            if cube_to_target < 0.03 and cube_z < 0.025:
                # Good drop - at target, on table
                reward += 5.0
            elif cube_to_target > 0.05:
                # Bad drop - far from target
                reward -= 3.0
            elif cube_z > 0.03:
                # Dropped at height
                reward -= 2.0
    
    # ========== SUCCESS ==========
    if is_placed:
        reward += 20.0
    
    # ========== SMOOTHNESS ==========
    if action is not None:
        action_delta = action - env._prev_action
        action_penalty = 0.005 * np.sum(action_delta**2)
        reward -= action_penalty
    
    return reward


def reward_v22(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V22: Sparse milestone reward - clearer learning signal.
    
    Uses sparse bonuses at key milestones instead of dense shaping.
    Better for avoiding local optima but may need more samples.
    
    Milestones:
    1. Grasp acquired (+5)
    2. Cube lifted to 3cm (+3)
    3. Cube lifted to transport height (+3)
    4. Cube at target XY while lifted (+5)
    5. Cube placed at target (+15)
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    gripper_state = info.get("gripper_state", 1.0)
    
    place_target = info.get("place_target")
    cube_to_target = info.get("cube_to_target", float('inf'))
    is_placed = info.get("is_placed", False)
    
    # Minimal reach shaping (just to get started)
    if gripper_to_cube < 0.08:
        reach_reward = 0.3 * (1.0 - gripper_to_cube / 0.08)
        reward += reach_reward
    
    # Gripper closing when close
    if gripper_to_cube < 0.04:
        reward += (1.0 - gripper_state) * 0.5
    
    # Push penalty
    if cube_z < 0.008:
        reward -= 1.0
    
    # === MILESTONE BONUSES ===
    
    # M1: Grasp acquired
    if is_grasping:
        reward += 2.0  # Per-step grasp maintenance
        
        # M2: Lifted off table
        if cube_z > 0.025:
            reward += 1.0
        
        # M3: At transport height
        if cube_z > env.lift_height:
            reward += 1.5
            
            # M4: At target while lifted
            if cube_to_target < 0.03:
                reward += 2.0
    
    # Drop penalties
    if was_grasping and not is_grasping:
        if cube_to_target > 0.04 or cube_z > 0.03:
            reward -= 2.0
    
    # M5: Success
    if is_placed:
        reward += 15.0
    
    return reward


def reward_v23(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V23: Pick-and-place with action-aware shaping.
    
    Key improvements over v21:
    - Penalizes downward motion when gripper is closed (prevents pushing)
    - Only rewards gripper closing when cube contacts are detected
    - Lift reward scaled by upward action to encourage lifting
    - Stronger grasp + lift coupling
    
    This prevents the "press down and close" local optimum.
    """
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    gripper_state = info.get("gripper_state", 1.0)  # 1=open, 0=closed
    has_static = info.get("has_gripper_contact", False)
    has_moving = info.get("has_jaw_contact", False)
    has_any_contact = has_static or has_moving
    
    place_target = info.get("place_target")
    cube_to_target = info.get("cube_to_target", float('inf'))
    is_placed = info.get("is_placed", False)
    step_count = getattr(env, '_step_count', 0)
    
    # Get action Z component (for shaping lift behavior)
    action_z = action[2] if action is not None else 0.0
    is_gripper_closed = gripper_state < 0.3
    
    # ========== PHASE 1: REACH ==========
    reach_reward = 1.0 - np.tanh(8.0 * gripper_to_cube)
    reward += reach_reward * 0.3  # Low weight - just for guidance
    
    # ========== GRIPPER CLOSING - only when contacting cube ==========
    if gripper_to_cube < 0.05 and has_any_contact:
        # Reward closing when we have contact
        close_reward = (1.0 - gripper_state) * 1.0
        reward += close_reward
    
    # ========== ANTI-PUSHING PENALTY ==========
    # Penalize pressing down when gripper is closed but not grasping
    if is_gripper_closed and not is_grasping and action_z < -0.3:
        # Pushing down with closed gripper = bad (pushing cube)
        reward -= 1.5
    
    # Push-down penalty on cube
    if cube_z < 0.008:
        reward -= 2.0
    elif cube_z < 0.012:
        reward -= (0.012 - cube_z) * 50.0
    
    # ========== PHASE 2: GRASP ==========
    if is_grasping:
        reward += 2.0  # Grasp maintenance bonus
        
        # ========== PHASE 3: LIFT ==========
        # Reward lifting action when grasping
        if action_z > 0:
            reward += action_z * 1.0  # Encourage upward motion
        
        # Lift progress reward
        lift_progress = max(0, cube_z - 0.015) / max(0.001, env.lift_height - 0.015)
        lift_progress = min(1.0, lift_progress)
        reward += lift_progress * 4.0  # Strong lift reward
        
        # Milestone bonuses
        if cube_z > 0.025:
            reward += 1.5
        if cube_z > 0.04:
            reward += 1.5
        if cube_z > env.lift_height:
            reward += 3.0
        
        # ========== PHASE 4: TRANSPORT ==========
        if place_target is not None and cube_z > env.lift_height * 0.6:
            transport_reward = 1.0 - np.tanh(5.0 * cube_to_target)
            reward += transport_reward * 2.5
            
            if cube_to_target < 0.04:
                reward += 3.0
            if cube_to_target < 0.02:
                reward += 2.0
        
        # ========== PHASE 5: LOWER ==========
        if place_target is not None and cube_to_target < 0.05:
            if cube_z < env.lift_height and cube_z > 0.015:
                lower_progress = 1.0 - (cube_z - 0.015) / max(0.001, env.lift_height - 0.015)
                reward += lower_progress * 2.0
    
    else:
        # NOT GRASPING
        
        # Time penalty after initial reaching phase
        if step_count > 30:
            time_penalty = min(0.3, (step_count - 30) * 0.003)
            reward -= time_penalty
        
        # Encourage upward motion to position for grasp (not downward)
        if gripper_to_cube < 0.04 and action_z > 0.2:
            reward += 0.3
        
        # Drop handling
        if was_grasping:
            if cube_to_target < 0.03 and cube_z < 0.025:
                reward += 5.0  # Good placement
            elif cube_to_target > 0.05:
                reward -= 3.0
            elif cube_z > 0.03:
                reward -= 2.0
    
    # ========== SUCCESS ==========
    if is_placed:
        reward += 20.0
    
    # ========== SMOOTHNESS ==========
    if action is not None:
        action_delta = action - env._prev_action
        reward -= 0.003 * np.sum(action_delta**2)
    
    return reward


def reward_v24(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V24: Simple positive-focused reward for pick-and-place.
    
    Key principles:
    - Mostly positive rewards to encourage exploration
    - Big bonuses for key milestones (grasp, lift, transport, place)
    - Minimal penalties (only for catastrophic failures)
    - Simple structure that's easy to optimize
    
    This is more forgiving than v23 which had too many penalties.
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    gripper_state = info.get("gripper_state", 1.0)
    
    place_target = info.get("place_target")
    cube_to_target = info.get("cube_to_target", float('inf'))
    is_placed = info.get("is_placed", False)
    
    # ========== ALWAYS-ON REWARDS (exploration friendly) ==========
    
    # 1. Reach reward - always positive, guides toward cube
    reach_reward = 1.0 - np.tanh(5.0 * gripper_to_cube)
    reward += reach_reward  # 0 to 1.0
    
    # 2. Gripper closing when close to cube
    if gripper_to_cube < 0.06:
        close_bonus = (1.0 - gripper_state) * 0.5
        reward += close_bonus  # 0 to 0.5
    
    # ========== GRASP MILESTONE ==========
    if is_grasping:
        reward += 3.0  # Big bonus for grasping!
        
        # ========== LIFT REWARDS ==========
        # Scale with height - very generous
        lift_reward = min(1.0, cube_z / env.lift_height) * 5.0
        reward += lift_reward  # 0 to 5.0
        
        # Milestone bonuses
        if cube_z > 0.03:
            reward += 2.0
        if cube_z > 0.05:
            reward += 2.0
        if cube_z > env.lift_height:
            reward += 3.0
        
        # ========== TRANSPORT (only when lifted) ==========
        if cube_z > env.lift_height * 0.5:
            transport_reward = (1.0 - np.tanh(3.0 * cube_to_target)) * 3.0
            reward += transport_reward
            
            # At target bonus
            if cube_to_target < 0.05:
                reward += 2.0
            if cube_to_target < 0.03:
                reward += 2.0
        
        # ========== LOWERING AT TARGET ==========
        if cube_to_target < 0.05 and cube_z < env.lift_height:
            # Reward for being low at target (ready to release)
            lower_reward = (1.0 - cube_z / env.lift_height) * 2.0
            reward += lower_reward
    
    # ========== PENALTIES (minimal, only for bad outcomes) ==========
    
    # Only penalize dropping far from target
    if was_grasping and not is_grasping:
        if cube_to_target > 0.05:
            reward -= 2.0  # Dropped away from target
    
    # Penalize cube falling off table
    if cube_z < 0.005:
        reward -= 1.0
    
    # ========== SUCCESS BONUS ==========
    if is_placed:
        reward += 25.0  # Big success bonus!
    
    return reward

