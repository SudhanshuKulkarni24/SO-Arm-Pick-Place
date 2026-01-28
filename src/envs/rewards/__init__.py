"""Reward functions for lift cube environment.

Working rewards (use these for new training):
- v11: State-based (SAC) - 100% success at 1M steps
- v19: Image-based (DrQ-v2) - 100% success at 2M steps
- v21: Pick-and-place (SAC) - improved grasp incentives (RECOMMENDED)
- v22: Pick-and-place sparse milestones

Legacy rewards (for checkpoint compatibility only):
- v1-v10, v12-v18: Historical experiments
- v20: Pick-and-place (has local optima issues)
"""

from .lift_rewards import reward_v11, reward_v19, reward_v20, reward_v21, reward_v22
from ._legacy_rewards import (
    reward_v1,
    reward_v2,
    reward_v3,
    reward_v4,
    reward_v5,
    reward_v6,
    reward_v7,
    reward_v8,
    reward_v9,
    reward_v10,
    reward_v12,
    reward_v13,
    reward_v14,
    reward_v15,
    reward_v16,
    reward_v17,
    reward_v18,
)

# Registry of all reward functions
REWARD_FUNCTIONS = {
    # Legacy (don't use for new training)
    "v1": reward_v1,
    "v2": reward_v2,
    "v3": reward_v3,
    "v4": reward_v4,
    "v5": reward_v5,
    "v6": reward_v6,
    "v7": reward_v7,
    "v8": reward_v8,
    "v9": reward_v9,
    "v10": reward_v10,
    # Working (state-based lift)
    "v11": reward_v11,
    # Legacy (image-based experiments)
    "v12": reward_v12,
    "v13": reward_v13,
    "v14": reward_v14,
    "v15": reward_v15,
    "v16": reward_v16,
    "v17": reward_v17,
    "v18": reward_v18,
    # Working (image-based lift)
    "v19": reward_v19,
    # Legacy pick-and-place (has local optima issues)
    "v20": reward_v20,
    # Working (pick-and-place with improved grasp incentives) - RECOMMENDED
    "v21": reward_v21,
    # Sparse milestone reward (alternative)
    "v22": reward_v22,
}

__all__ = ["REWARD_FUNCTIONS", "reward_v11", "reward_v19", "reward_v20", "reward_v21", "reward_v22"]
