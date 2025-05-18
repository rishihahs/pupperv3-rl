"""Plotting utilities for Pupper V3 RL."""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jp
from typing import List, Union, Optional, Tuple

def plot_multi_series(
    data: Union[np.ndarray, jp.ndarray],
    dt: float = 0.02,
    display_axes: List[int] = None,
    title: str = "Data Series",
    labels: List[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plots multiple time series.
    
    Args:
        data: Data array with shape (time_steps, num_series).
        dt: Time step.
        display_axes: List of indices to display.
        title: Plot title.
        labels: Series labels.
        figsize: Figure size.
    """
    # Convert to numpy if needed
    if isinstance(data, jp.ndarray):
        data = np.array(data)
    
    # Create time axis
    time = np.arange(data.shape[0]) * dt
    
    # Select axes to display
    if display_axes is None:
        display_axes = list(range(data.shape[1]))
    
    # Create labels if not provided
    if labels is None:
        labels = [f"Series {i}" for i in display_axes]
    elif len(labels) < len(display_axes):
        labels = labels + [f"Series {i}" for i in display_axes[len(labels):]]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    for i, axis in enumerate(display_axes):
        if axis < data.shape[1]:
            plt.plot(time, data[:, axis], label=labels[i])
    
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reward_components(
    metrics: dict,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Plots reward components.
    
    Args:
        metrics: Dictionary of metrics.
        figsize: Figure size.
    """
    # Extract reward components
    reward_keys = [key for key in metrics.keys() if key.startswith('reward_')]
    reward_values = [metrics[key] for key in reward_keys]
    
    # Format keys for display
    display_keys = [key.replace('reward_', '') for key in reward_keys]
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.barh(display_keys, reward_values)
    plt.title("Reward Components")
    plt.xlabel("Value")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()


def plot_observation_components(
    obs: Union[np.ndarray, jp.ndarray],
    observation_history: int = 4,
    component_sizes: dict = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Plots observation components.
    
    Args:
        obs: Observation array.
        observation_history: Number of observation history steps.
        component_sizes: Dictionary mapping component names to sizes.
        figsize: Figure size.
    """
    # Convert to numpy if needed
    if isinstance(obs, jp.ndarray):
        obs = np.array(obs)
    
    # Default component sizes
    if component_sizes is None:
        component_sizes = {
            'joint_pos': 12,
            'joint_vel': 12,
            'prev_action': 12,
            'imu': 15,
            'command': 3,
            'desired_world_z': 3,
            'gravity_dir': 3,
            'progress': 1,
            'action_latency': 2,
            'imu_latency': 2
        }
    
    # Calculate single observation size
    single_obs_size = sum(component_sizes.values())
    
    # Reshape observation to (history, components)
    obs_reshaped = obs.reshape(observation_history, single_obs_size)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    plt.imshow(obs_reshaped, aspect='auto', cmap='viridis')
    plt.colorbar(label='Value')
    
    # Add component labels
    component_boundaries = []
    start_idx = 0
    for component, size in component_sizes.items():
        component_boundaries.append((start_idx, component))
        start_idx += size
    
    plt.yticks(range(observation_history), [f"t-{i}" for i in range(observation_history)])
    plt.xticks([b[0] for b in component_boundaries], [b[1] for b in component_boundaries], rotation=45)
    
    plt.title("Observation Components")
    plt.xlabel("Components")
    plt.ylabel("History Steps")
    plt.tight_layout()
    plt.show()


def plot_joint_positions(
    state_rollout: List,
    dt: float = 0.02,
    joint_names: List[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Plots joint positions over time.
    
    Args:
        state_rollout: List of states.
        dt: Time step.
        joint_names: List of joint names.
        figsize: Figure size.
    """
    # Extract joint positions
    joint_pos = np.array([np.array(state.qpos[7:]) for state in state_rollout])
    
    # Default joint names
    if joint_names is None:
        joint_names = [
            "FR_hip", "FR_upper", "FR_lower",
            "FL_hip", "FL_upper", "FL_lower",
            "BR_hip", "BR_upper", "BR_lower",
            "BL_hip", "BL_upper", "BL_lower"
        ]
    
    # Create time axis
    time = np.arange(len(state_rollout)) * dt
    
    # Create plot
    plt.figure(figsize=figsize)
    
    for i in range(joint_pos.shape[1]):
        plt.plot(time, joint_pos[:, i], label=joint_names[i] if i < len(joint_names) else f"Joint {i}")
    
    plt.title("Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trajectory(
    state_rollout: List,
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """Plots robot trajectory in XY plane.
    
    Args:
        state_rollout: List of states.
        figsize: Figure size.
    """
    # Extract positions
    positions = np.array([np.array(state.qpos[:2]) for state in state_rollout])
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot trajectory
    plt.plot(positions[:, 0], positions[:, 1], '-o', label='Robot path')
    
    # Mark start and end
    plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    
    plt.title("Robot Trajectory")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()
