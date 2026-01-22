"""Train a pick-and-place policy using SAC.

Task: Grasp cube → lift → transport to target → lower → release.

Usage:
    PYTHONPATH=. uv run python train_pick_place.py --config configs/pick_place.yaml
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.callbacks.plot_callback import PlotLearningCurveCallback
from src.envs.pick_place import PickPlaceEnv


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict):
    place_target = env_cfg.get("place_target", [0.35, 0.10])
    if place_target is not None:
        place_target = tuple(place_target)

    return PickPlaceEnv(
        render_mode=None,
        max_episode_steps=env_cfg.get("max_episode_steps", 400),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        reward_type=env_cfg.get("reward_type", "dense"),
        reward_version=env_cfg.get("reward_version", "v20"),
        curriculum_stage=env_cfg.get("curriculum_stage", 3),
        place_target=place_target,
        target_radius=env_cfg.get("target_radius", 0.02),
        randomize_cube=env_cfg.get("randomize_cube", True),
        randomize_target=env_cfg.get("randomize_target", True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pick_place.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained lift model (transfer learning)")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    env_cfg = config["env"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        resume_dir = Path(args.resume)
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / f"{timestamp}_resumed"
    else:
        resume_dir = None
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, output_dir / "config.yaml")

    if args.resume:
        with open(output_dir / "RESUME_INFO.txt", "w") as f:
            f.write(f"Resumed from: {resume_dir}\n")

    # Determine VecNormalize path
    pretrained = args.pretrained or train_cfg.get("pretrained")
    vec_normalize_path = None
    if args.resume:
        vec_normalize_path = resume_dir / "vec_normalize.pkl"

    # Create environments
    env = DummyVecEnv([lambda: make_env(env_cfg)])

    if vec_normalize_path and vec_normalize_path.exists():
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = True
        print(f"Loaded normalization stats from {vec_normalize_path}")
    elif vec_normalize_path:
        raise ValueError(f"vec_normalize.pkl not found: {vec_normalize_path}")
    else:
        env = VecNormalize(
            env,
            norm_obs=env_cfg.get("normalize_obs", True),
            norm_reward=env_cfg.get("normalize_reward", True),
        )
        if pretrained:
            print("Using fresh VecNormalize (not loading old stats for transfer)")

    eval_env = DummyVecEnv([lambda: make_env(env_cfg)])
    if vec_normalize_path and vec_normalize_path.exists():
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=env_cfg.get("normalize_obs", True),
            norm_reward=False,
            training=False,
        )

    # Create or load model
    resume_step = 0
    if args.resume:
        checkpoints = list((resume_dir / "checkpoints").glob("*.zip"))
        def get_step_number(path):
            for part in path.stem.split("_"):
                if part.isdigit():
                    return int(part)
            return 0
        checkpoints = sorted(checkpoints, key=get_step_number)
        if checkpoints:
            latest = checkpoints[-1]
            resume_step = get_step_number(latest)
            model = SAC.load(latest, env=env, device=device)
            model.tensorboard_log = str(output_dir / "tensorboard")
            print(f"Resumed from {latest} (step {resume_step})")
        else:
            raise ValueError(f"No checkpoints found in {resume_dir / 'checkpoints'}")
    elif pretrained:
        model = SAC.load(pretrained, env=env, device=device)
        model.tensorboard_log = str(output_dir / "tensorboard")
        model.num_timesteps = 0
        model._episode_num = 0
        model.replay_buffer.reset()
        print(f"Loaded pretrained weights from {pretrained}")
        with open(output_dir / "PRETRAINED_INFO.txt", "w") as f:
            f.write(f"Pretrained from: {pretrained}\n")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_cfg["learning_rate"],
            buffer_size=sac_cfg["buffer_size"],
            learning_starts=sac_cfg["learning_starts"],
            batch_size=sac_cfg["batch_size"],
            tau=sac_cfg["tau"],
            gamma=sac_cfg["gamma"],
            train_freq=sac_cfg["train_freq"],
            gradient_steps=sac_cfg["gradient_steps"],
            verbose=1,
            seed=train_cfg["seed"],
            device=device,
            tensorboard_log=str(output_dir / "tensorboard"),
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_pick_place",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_cfg["eval_freq"],
        deterministic=True,
        render=False,
    )

    plot_callback = PlotLearningCurveCallback(
        run_dir=output_dir,
        save_freq=train_cfg["save_freq"],
        verbose=1,
        resume_step=resume_step,
    )

    timesteps = args.timesteps if args.timesteps is not None else train_cfg["timesteps"]

    if args.resume:
        reset_num_timesteps = False
        learn_timesteps = timesteps
        print(f"\nResuming Pick-Place training from step {model.num_timesteps}...")
        print(f"Training for {timesteps} additional timesteps")
    else:
        reset_num_timesteps = True
        learn_timesteps = timesteps
        print(f"\nStarting Pick-Place training for {timesteps} timesteps...")

    print(f"Action space: delta XYZ + gripper (4 dims)")
    print(f"Place target: {env_cfg.get('place_target', [0.35, 0.10])}")
    print(f"Output directory: {output_dir}")

    model.learn(
        total_timesteps=learn_timesteps,
        callback=[checkpoint_callback, eval_callback, plot_callback],
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
