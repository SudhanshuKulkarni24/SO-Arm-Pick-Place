"""Test pick-and-place environment with scripted policy.

Demonstrates the full pick-and-place sequence:
1. Move above cube
2. Lower to grasp
3. Close gripper
4. Lift cube
5. Transport to target
6. Lower cube
7. Release gripper

Usage:
    PYTHONPATH=. uv run python tests/test_pick_place.py
    PYTHONPATH=. uv run python tests/test_pick_place.py --video  # Save video
"""
import argparse
import numpy as np
import mujoco
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.pick_place import PickPlaceEnv


def scripted_pick_place(env, verbose=True):
    """Execute scripted pick-and-place sequence."""
    obs, info = env.reset()

    if verbose:
        print(f"Cube at: {info['cube_pos']}")
        print(f"Target at: {info['place_target']}")
        print(f"Distance to target: {info['cube_to_target']:.3f}m")
        print()

    frames = []
    total_reward = 0.0

    # Phase 1: Move above cube
    if verbose:
        print("1. Moving above cube...")
    cube_pos = info['cube_pos']
    gripper_pos = info['gripper_pos']

    for step in range(50):
        # Move toward cube XY, stay at current Z
        delta = cube_pos[:2] - gripper_pos[:2]
        delta = np.clip(delta / env.action_scale, -1, 1)
        action = np.array([delta[0], delta[1], 0.0, 0.5])  # gripper open

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        gripper_pos = info['gripper_pos']

        if env.render_mode == "rgb_array":
            frames.append(env.render())

    # Phase 2: Lower to cube
    if verbose:
        print("2. Lowering to cube...")
    target_z = cube_pos[2] + 0.01  # Just above cube

    for step in range(40):
        delta_z = (target_z - gripper_pos[2]) / env.action_scale
        delta_z = np.clip(delta_z, -1, 1)
        action = np.array([0.0, 0.0, delta_z, 0.5])  # gripper open

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        gripper_pos = info['gripper_pos']

        if env.render_mode == "rgb_array":
            frames.append(env.render())

    # Phase 3: Close gripper
    if verbose:
        print("3. Closing gripper...")
    for step in range(30):
        action = np.array([0.0, 0.0, 0.0, -1.0])  # close gripper

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if env.render_mode == "rgb_array":
            frames.append(env.render())

        if info['is_grasping']:
            if verbose:
                print(f"   Grasped at step {step}!")
            break

    # Phase 4: Lift cube
    if verbose:
        print("4. Lifting cube...")
    lift_target = env.lift_height + 0.02

    for step in range(50):
        gripper_pos = info['gripper_pos']
        delta_z = (lift_target - gripper_pos[2]) / env.action_scale
        delta_z = np.clip(delta_z, -1, 1)
        action = np.array([0.0, 0.0, delta_z, -1.0])  # keep gripper closed

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if env.render_mode == "rgb_array":
            frames.append(env.render())

        if info['cube_z'] > env.lift_height:
            if verbose:
                print(f"   Lifted to {info['cube_z']:.3f}m")
            break

    # Phase 5: Transport to target
    if verbose:
        print("5. Transporting to target...")
    target_pos = info['place_target']

    for step in range(120):  # More steps for transport
        gripper_pos = info['gripper_pos']
        cube_pos = info['cube_pos']  # Track cube position, not gripper
        
        # Move toward target based on cube position
        delta = target_pos[:2] - cube_pos[:2]
        delta = np.clip(delta / env.action_scale, -1, 1)

        # Maintain height while transporting
        delta_z = (lift_target - gripper_pos[2]) / env.action_scale
        delta_z = np.clip(delta_z, -1, 1)

        action = np.array([delta[0], delta[1], delta_z, -1.0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if env.render_mode == "rgb_array":
            frames.append(env.render())

        if info['cube_to_target'] < 0.015:  # Stricter tolerance
            if verbose:
                print(f"   Reached target zone! Distance: {info['cube_to_target']:.3f}m")
            break
    
    if verbose and info['cube_to_target'] >= 0.015:
        print(f"   Transport done. Distance: {info['cube_to_target']:.3f}m")

    # Phase 6: Lower cube
    if verbose:
        print("6. Lowering cube...")
    place_z = 0.02

    for step in range(50):
        gripper_pos = info['gripper_pos']
        delta_z = (place_z - gripper_pos[2]) / env.action_scale
        delta_z = np.clip(delta_z, -1, 1)
        action = np.array([0.0, 0.0, delta_z, -1.0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if env.render_mode == "rgb_array":
            frames.append(env.render())

        if info['cube_z'] < 0.025:
            if verbose:
                print(f"   Cube on table at z={info['cube_z']:.3f}m")
            break

    # Phase 7: Release gripper
    if verbose:
        print("7. Releasing gripper...")
    
    # Move up slightly before releasing to avoid pushing cube
    for step in range(15):
        action = np.array([0.0, 0.0, 0.3, -0.5])  # move up, start opening
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if env.render_mode == "rgb_array":
            frames.append(env.render())

    # Now fully open
    for step in range(20):
        action = np.array([0.0, 0.0, 0.0, 1.0])  # open gripper

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if env.render_mode == "rgb_array":
            frames.append(env.render())

        if not info['is_grasping']:
            break

    # Hold position briefly
    for step in range(20):
        action = np.array([0.0, 0.0, 0.1, 1.0])  # move up slightly, stay open
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if env.render_mode == "rgb_array":
            frames.append(env.render())

    # Final status
    if verbose:
        print()
        print(f"Final cube position: {info['cube_pos']}")
        print(f"Final cube Z: {info['cube_z']:.4f}m")
        print(f"Distance to target: {info['cube_to_target']:.3f}m")
        print(f"Is placed: {info['is_placed']}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Result: {'SUCCESS' if info['is_success'] else 'FAILED'}")

    return frames, info['is_success'], total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", help="Save video")
    parser.add_argument("--target", type=float, nargs=2, default=[0.35, 0.10],
                        help="Target XY position")
    args = parser.parse_args()

    print("=== Pick and Place Test ===\n")

    render_mode = "rgb_array" if args.video else None

    env = PickPlaceEnv(
        render_mode=render_mode,
        max_episode_steps=400,
        place_target=tuple(args.target),
        randomize_cube=False,
        randomize_target=False,
    )

    frames, success, total_reward = scripted_pick_place(env, verbose=True)

    if args.video and frames:
        import imageio
        output_dir = Path("devlogs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = output_dir / f"pick_place_{timestamp}.mp4"
        imageio.mimsave(str(video_path), frames, fps=30)
        print(f"\nSaved {len(frames)} frames to {video_path}")

    env.close()


if __name__ == "__main__":
    main()
