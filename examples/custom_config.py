"""Custom configuration example for Pupper V3 RL."""

from ml_collections import config_dict
from pupperv3_rl.config import get_default_config
import jax.numpy as jp

def get_config():
    """Gets custom configuration.
    
    Returns:
        Custom configuration.
    """
    # Start with default configuration
    config = get_default_config()
    
    # Override training parameters
    config.training.ppo.num_timesteps = 200_000_000  # Shorter training
    config.training.ppo.episode_length = 200        # Shorter episodes
    config.training.ppo.learning_rate = 5.0e-5      # Different learning rate
    
    # Override policy architecture
    config.policy.hidden_layer_sizes = (512, 256, 256, 128)  # Larger network
    config.policy.observation_history = 2                   # Less history
    
    # Override command ranges for easier tasks
    config.training.lin_vel_x_range = [0.0, 0.5]      # Only forward motion
    config.training.lin_vel_y_range = [-0.2, 0.2]     # Limited sideways motion
    config.training.ang_vel_yaw_range = [-1.0, 1.0]   # Reduced turning rate
    
    # Override reward weights
    config.reward.rewards.scales.tracking_lin_vel = 2.0   # Emphasize tracking
    config.reward.rewards.scales.feet_air_time = 0.05     # Encourage stepping
    
    # Enable heightfield for uneven terrain training
    config.training.height_field_random = True
    config.training.height_field_elevation_z = 0.015  # Smaller bumps
    
    # Override default pose for a different starting posture
    config.training.default_pose = jp.array(
        [0.2, 0.0, -0.5, -0.2, 0.0, 0.5, 0.2, 0.0, -0.5, -0.2, 0.0, 0.5]
    )
    
    return config
