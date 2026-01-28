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
PYTHONPATH=. uv run python eval.py --exp-dir runs/pick_place/20260128_211005 --episodes 10 --record
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

## Reward Function (v21)

The reward function is designed with 5 phases to guide the agent through the pick-and-place sequence. v21 includes improved grasp incentives to prevent the "pushing" local optimum.

### Key Improvements in v21 (over v20)

- **Gripper closing incentive** when near cube (prevents pushing)
- **Time penalty** for not grasping (forces early grasp)
- **Larger grasp bonus** (+3.0 vs +0.5)
- **Gated rewards**: transport/place rewards require grasping

### Phase 1: Reach (→ Cube)

```
reach_reward = (1.0 - tanh(8.0 × gripper_to_cube_distance)) × 0.5
```

Reduced weight (0.5) to prevent farming reach reward without grasping.

**Gripper closing incentive** (when within 5cm of cube):
```
close_reward = (1.0 - gripper_state) × 1.5   # gripper_state: 1=open, 0=closed
extra_close = (1.0 - gripper_state) × 1.0    # if within 3cm
```

**Push-down penalty:**
```
if cube_z < 0.005: penalty = -2.0
if cube_z < 0.01:  penalty = -(0.01 - cube_z) × 100.0
```

### Phase 2: Grasp

When grasping (both finger pads contact cube + gripper closed):
```
grasp_bonus = +3.0  (large bonus to incentivize grasping!)
```

### Phase 3: Lift (while grasping)

```
lift_reward = min(1.0, (cube_z - 0.015) / (lift_height - 0.015)) × 3.0
milestone_2cm = +1.0  (if cube_z > 0.02m)
milestone_4cm = +1.0  (if cube_z > 0.04m)
transport_ready = +2.0  (if cube_z > lift_height)
```

### Phase 4: Transport (→ Target)

When cube is lifted (z > 60% of lift_height) AND grasping:
```
transport_reward = (1.0 - tanh(5.0 × cube_to_target_xy)) × 2.0
at_target_4cm = +3.0  (if within 4cm)
at_target_2cm = +2.0  (if within 2cm)
```

### Phase 5: Lower & Release

When near target (< 5cm) AND grasping:
```
lowering_reward = (1.0 - (cube_z - 0.015) / (lift_height - 0.015)) × 2.0
```

When released at target:
```
good_release = +5.0  (if at target, cube on table)
```

### Penalties

**Time penalty** (encourages fast grasping):
```
if not grasping and step > 50:
    penalty = -min(0.5, (step - 50) × 0.005)
```

**Drop penalty:**
- Far from target (> 5cm) or at height (> 3cm): **-2.0 to -3.0**

### Success Bonus

```
success_bonus = +20.0  (cube at target, on table, released)
```

### Reward Summary Table

| Component | Condition | Reward |
|-----------|-----------|--------|
| Reach | Always | 0 to 0.5 |
| Gripper close | < 5cm from cube | 0 to 2.5 |
| Push penalty | cube_z < 0.01 | -2.0 to -1.0 |
| **Grasp bonus** | Grasping | **+3.0** |
| Lift progress | Grasping | 0 to 3.0 |
| Milestone 2cm | cube_z > 0.02 | +1.0 |
| Milestone 4cm | cube_z > 0.04 | +1.0 |
| Transport ready | cube_z > 0.08 | +2.0 |
| Transport | Lifted + grasping | 0 to 2.0 |
| At target 4cm | < 4cm, grasping | +3.0 |
| At target 2cm | < 2cm, grasping | +2.0 |
| Lowering | Near target, grasping | 0 to 2.0 |
| Good release | At target, on table | +5.0 |
| **Success** | Task complete | **+20.0** |
| Time penalty | Not grasping, step > 50 | -0.0 to -0.5 |
| Drop (bad) | Far from target | -2.0 to -3.0 |
| Action smoothness | Always | -0.005 × Δaction² |

### Typical Episode Rewards

| Outcome | Approximate Total Reward |
|---------|-------------------------|
| Failed to grasp (just pushing) | 50-100 |
| Grasped but dropped early | 150-250 |
| Lifted but missed target | 300-400 |
| Near target but failed place | 400-500 |
| **Successful placement** | **500-600+** |

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
  reward_version: "v21"   # Pick-and-place reward with improved grasp incentives
  curriculum_stage: 3     # Start with gripper near cube
  place_target: [0.35, 0.10]  # Target XY position
```

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
