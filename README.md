# Pick and Place Task

Train an RL agent to pick up a cube, transport it to a target location, and place it down.

## Task Overview

The agent controls a SO-101 robot arm to:
1. **Reach** - Move gripper to the cube
2. **Grasp** - Close gripper to pick up the cube
3. **Lift** - Raise cube to transport height (8cm)
4. **Transport** - Move cube horizontally to target location
5. **Lower** - Bring cube down to table
6. **Release** - Open gripper to place cube

**Success Condition:** Cube within 2cm of target XY position, resting on table (z < 2.5cm), and gripper released.

## Quick Start

### Test the Environment (Scripted Policy)

```bash
cd pick-101

# Run scripted pick-and-place demo
PYTHONPATH=. uv run python tests/test_pick_place.py

# Generate video
PYTHONPATH=. uv run python tests/test_pick_place.py --video

# Custom target location
PYTHONPATH=. uv run python tests/test_pick_place.py --target 0.30 0.15
```

### Train the Agent

```bash
# Full training (2M steps, ~2-4 hours on GPU)
PYTHONPATH=. uv run python train_pick_place.py --config configs/pick_place.yaml

# Shorter training for testing
PYTHONPATH=. uv run python train_pick_place.py --config configs/pick_place.yaml --timesteps 500000

# Resume from checkpoint
PYTHONPATH=. uv run python train_pick_place.py --config configs/pick_place.yaml --resume runs/pick_place/<timestamp>

# Transfer from trained lift policy
PYTHONPATH=. uv run python train_pick_place.py --config configs/pick_place.yaml --pretrained runs/lift_curriculum_s3/<timestamp>/best_model/best_model.zip
```

### Evaluate Trained Model

```bash
# Evaluate with video recording
PYTHONPATH=. uv run python eval.py --exp-dir runs/pick_place/<timestamp> --episodes 10 --record
```

## Environment Details

### Observation Space (24 dimensions)

| Component | Dims | Description |
|-----------|------|-------------|
| Joint positions | 6 | Robot arm joint angles |
| Joint velocities | 6 | Robot arm joint velocities |
| Gripper position | 3 | End-effector XYZ position |
| Gripper orientation | 3 | End-effector euler angles |
| Cube position | 3 | Cube XYZ position |
| Target position | 3 | Target XYZ position |

### Action Space (4 dimensions)

| Component | Range | Description |
|-----------|-------|-------------|
| Delta X | [-1, 1] | End-effector X movement (scaled by 2cm) |
| Delta Y | [-1, 1] | End-effector Y movement (scaled by 2cm) |
| Delta Z | [-1, 1] | End-effector Z movement (scaled by 2cm) |
| Gripper | [-1, 1] | Gripper open (+1) / close (-1) |

### Episode Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max steps | 400 | Episode timeout |
| Action scale | 0.02m | Max movement per step |
| Lift height | 0.08m | Transport height |
| Target radius | 0.02m | Success threshold |

## Reward Function (v24)

The reward function uses a **simple positive-focused design** that encourages exploration while providing clear milestones for the pick-and-place task.

### Key Principles in v24

- **Mostly positive rewards**: Encourages exploration (random actions give ~0.5 reward)
- **Big milestone bonuses**: Clear signal for grasp (+3), lift (+5), transport (+3), success (+25)
- **Minimal penalties**: Only for catastrophic failures (dropping far from target)
- **Simple structure**: Easy to optimize, no complex action-dependent shaping

### Phase 1: Reach (Always Active)

```
reach_reward = 1.0 - tanh(5.0 × gripper_to_cube_distance)  # 0 to 1.0
```

**Gripper closing bonus** (when close to cube):
```
if gripper_to_cube < 0.06:
    close_bonus = (1.0 - gripper_state) × 0.5  # 0 to 0.5
```

### Phase 2: Grasp

When grasping (both finger pads contact cube + gripper closed):
```
grasp_bonus = +3.0  (big bonus for achieving grasp!)
```

### Phase 3: Lift (while grasping)

```
# Generous lift reward scaled by height
lift_reward = min(1.0, cube_z / lift_height) × 5.0  # 0 to 5.0

# Milestone bonuses
milestone_3cm = +2.0  (if cube_z > 0.03m)
milestone_5cm = +2.0  (if cube_z > 0.05m)
transport_ready = +3.0  (if cube_z > lift_height)
```

### Phase 4: Transport (→ Target)

When cube is lifted (z > 50% of lift_height) AND grasping:
```
transport_reward = (1.0 - tanh(3.0 × cube_to_target_xy)) × 3.0
at_target_5cm = +2.0  (if within 5cm)
at_target_3cm = +2.0  (if within 3cm)
```

### Phase 5: Lower at Target

When at target (< 5cm) AND grasping AND below lift height:
```
lower_reward = (1.0 - cube_z / lift_height) × 2.0
```

### Penalties (Minimal)

**Drop penalty** (only for bad drops):
```
if was_grasping and not grasping and cube_to_target > 0.05:
    penalty = -2.0  # Only penalize dropping far from target
```

**Cube falling off table:**
```
if cube_z < 0.005:
    penalty = -1.0
```

### Success Bonus

```
success_bonus = +25.0  (cube at target, on table, released)
```

### Reward Summary Table

| Component | Condition | Reward |
|-----------|-----------|--------|
| Reach | Always | 0 to 1.0 |
| Gripper close | < 6cm from cube | 0 to 0.5 |
| **Grasp bonus** | Grasping | **+3.0** |
| Lift progress | Grasping | 0 to 5.0 |
| Milestone 3cm | cube_z > 0.03 | +2.0 |
| Milestone 5cm | cube_z > 0.05 | +2.0 |
| Transport ready | cube_z > 0.08 | +3.0 |
| Transport | Lifted + grasping | 0 to 3.0 |
| At target 5cm | < 5cm, grasping | +2.0 |
| At target 3cm | < 3cm, grasping | +2.0 |
| Lowering | At target, grasping | 0 to 2.0 |
| **Success** | Task complete | **+25.0** |
| Drop (bad) | Far from target | -2.0 |
| Cube fell | cube_z < 0.005 | -1.0 |

### Typical Episode Rewards

| Outcome | Approximate Total Reward |
|---------|-------------------------|
| Random exploration | 40-80 |
| Reaching only | 80-150 |
| Grasped briefly | 200-350 |
| Grasped + lifted | 400-600 |
| Lifted + transported | 600-800 |
| **Successful placement** | **900-1200+** |

## Configuration

Edit `configs/pick_place.yaml` to customize training:

```yaml
experiment:
  name: "pick_place"
  base_dir: "runs"

training:
  timesteps: 2000000      # Total training steps
  eval_freq: 10000        # Evaluate every N steps
  save_freq: 100000       # Checkpoint every N steps
  seed: 42

sac:
  learning_rate: 3.0e-4
  buffer_size: 100000
  batch_size: 256
  gamma: 0.99

env:
  max_episode_steps: 400
  action_scale: 0.02
  lift_height: 0.08
  reward_version: "v24"   # Simple positive-focused reward - RECOMMENDED
  curriculum_stage: 3     # Start with gripper near cube
  place_target: [0.35, 0.10]  # Target XY position
```

## Reward Function History

| Version | Description | Status |
|---------|-------------|--------|
| v20 | Basic pick-and-place | Local optima (pushing) |
| v21 | Grasp incentives | Gets stuck pressing down |
| v23 | Action-aware shaping | Too many penalties (negative rewards) |
| **v24** | Simple positive-focused | **RECOMMENDED** |

## File Structure

```
pick-101/
├── configs/
│   └── pick_place.yaml         # Training configuration
├── models/so101/
│   └── pick_place.xml          # MuJoCo scene with target marker
├── src/envs/
│   ├── pick_place.py           # Pick-and-place environment
│   └── rewards/
│       └── lift_rewards.py     # Reward functions (v20)
├── tests/
│   └── test_pick_place.py      # Scripted policy test
├── train_pick_place.py         # Training script
└── docs/
    └── PICK_AND_PLACE.md       # This file
```

## Training Tips

1. **Start with curriculum stage 3** - Gripper starts near cube, making grasp learning easier.

2. **Transfer from lift policy** - Pre-train on the simpler lift task first:
   ```bash
   # Train lift first
   PYTHONPATH=. uv run python train_lift.py --config configs/curriculum_stage3.yaml
   
   # Then transfer to pick-place
   PYTHONPATH=. uv run python train_pick_place.py --pretrained runs/lift_curriculum_s3/<timestamp>/best_model/best_model.zip
   ```

3. **Monitor with TensorBoard:**
   ```bash
   tensorboard --logdir runs/pick_place
   ```

4. **Expected training time:**
   - GPU (RTX 3080+): ~2-4 hours for 2M steps
   - CPU: ~8-12 hours for 2M steps

5. **Success rate milestones:**
   - 500K steps: ~20-40% success (learning to grasp and lift)
   - 1M steps: ~50-70% success (learning transport)
   - 2M steps: ~80-95% success (refined placement)

## Troubleshooting

**Agent drops cube during transport:**
- The drop penalty may need tuning. Try increasing it in `lift_rewards.py`:
  ```python
  if cube_to_target > 0.04:
      reward -= 5.0  # Increase from 3.0
  ```

**Agent doesn't release at target:**
- Increase the release bonus or add a "gripper open near target" reward.

**Cube pushed away on release:**
- This is a physics challenge. The agent should learn to open gripper gradually or lift slightly before release.

**Training unstable:**
- Reduce learning rate to `1.0e-4`
- Increase `learning_starts` to `5000`
- Try reward normalization: `normalize_reward: true`

## Citation

If you use this environment, please cite:
```
@misc{pick101,
  title={Pick-101: RL Training for SO-101 Robot Manipulation},
  year={2026},
  url={https://github.com/ggand0/pick-101}
}
```
