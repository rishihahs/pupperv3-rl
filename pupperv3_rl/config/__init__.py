"""Configuration module for Pupper V3 RL."""

from pupperv3_rl.config.defaults import (
    get_default_config,
    get_simulation_config,
    get_training_config,
    get_policy_config,
    get_reward_config
)
from pupperv3_rl.config.loader import load_config, merge_configs
