#!/usr/bin/env python
"""Visualization script for Pupper V3 RL."""

import os
import argparse
import jax
import jax.numpy as jp
import numpy as np
import mediapy as media
from ml_collections import config_dict
from brax import envs
from pathlib import Path

from pupperv3_rl.config.loader import load_config
from pupperv3_rl.visualization.plotting import (
    plot_multi_series, 
    plot_reward_components, 
    plot_joint_positions, 
    plot_trajectory
)

# Import these to register the environment
from pupperv3_rl.environment.pupperv3_env import PupperV3Env

# For policy loading
from brax.training.agents.ppo import networks as ppo_networks


def load_policy(
    checkpoint_path: str,
    config: config_dict.ConfigDict
):
    """Loads policy from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration.
        
    Returns:
        Inference function.
    """
    # Load params
    with np.load(checkpoint_path) as data:
        params = {key: data[key] for key in data.keys()}
    
    # Create network factory
    from pupperv3_rl.training.utils import activation_fn_map
    
    make_networks = ppo_networks.make_ppo_networks(
        policy_hidden_layer_sizes=config.policy.hidden_layer_sizes,
        activation=activation_fn_map(config.policy.activation)
    )
    
    # Create inference function
    inference_fn = make_networks(config.policy.action_dim, config.policy.obs_dim)[0].apply
    
    return inference_fn, params


def visualize_policy_rollout(
    config: config_dict.ConfigDict,
    params: dict,
    inference_fn,
    commands: list,
    output_dir: str,
    n_steps: int = 500,
    render_every: int = 2,
    seed: int = 0
):
    """Visualizes policy rollout.
    
    Args:
        config: Configuration.
        params: Policy parameters.
        inference_fn: Inference function.
        commands: List of commands to test.
        output_dir: Output directory.
        n_steps: Number of steps to simulate.
        render_every: Render every N steps.
        seed: Random seed.
    """
    # Setup environment
    from pupperv3_rl.scripts.train import setup_environment
    _, eval_env = setup_environment(config)
    
    # JIT functions
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(lambda obs, rng: inference_fn(params, obs, rng))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize each command
    for cmd_idx, command in enumerate(commands):
        print(f"Visualizing command: {command}")
        
        # Initialize state
        rng = jax.random.PRNGKey(seed + cmd_idx)
        state = jit_reset(rng)
        state.info['command'] = jp.array(command)
        rollout = [state.pipeline_state]
        states = [state]
        actions = []
        
        # Collect trajectory
        for i in range(n_steps):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            actions.append(ctrl)
            state = jit_step(state, ctrl)
            rollout.append(state.pipeline_state)
            states.append(state)
        
        # Render video
        video = eval_env.render(rollout[::render_every], camera='tracking_cam')
        
        # Save video
        video_path = os.path.join(output_dir, f'policy_cmd_{cmd_idx}.mp4')
        media.write_video(video_path, video, fps=1.0 / eval_env.dt / render_every)
        
        print(f"Saved video to {video_path}")
        
        # Plot actions
        action_array = np.array(actions)
        plot_multi_series(
            data=action_array,
            dt=config.training.environment_dt,
            display_axes=list(range(min(12, action_array.shape[1]))),
            title=f"Policy Actions for Command {command}"
        )
        
        # Plot joint positions
        plot_joint_positions(
            state_rollout=rollout,
            dt=config.training.environment_dt
        )
        
        # Plot trajectory
        plot_trajectory(state_rollout=rollout)
        
        # Plot final reward components
        plot_reward_components(states[-1].metrics)
        
        # Calculate velocity tracking errors
        command_array = np.array([state.info['command'] for state in states])
        velocity_array = np.array([state.pipeline_state.qvel[:3] for state in states])
        angular_velocity_array = np.array([state.pipeline_state.qvel[3:6] for state in states])
        
        # Plot velocity tracking
        plot_multi_series(
            data=np.column_stack([
                command_array[:, 0],  # Command X
                velocity_array[:, 0],  # Velocity X
                command_array[:, 1],  # Command Y
                velocity_array[:, 1],  # Velocity Y
            ]),
            dt=config.training.environment_dt,
            display_axes=[0, 1, 2, 3],
            labels=["Command X", "Velocity X", "Command Y", "Velocity Y"],
            title=f"Linear Velocity Tracking for Command {command}"
        )
        
        # Plot angular velocity tracking
        plot_multi_series(
            data=np.column_stack([
                command_array[:, 2],  # Command Z rotation
                angular_velocity_array[:, 2],  # Angular velocity Z
            ]),
            dt=config.training.environment_dt,
            display_axes=[0, 1],
            labels=["Command Z rot", "Angular Velocity Z"],
            title=f"Angular Velocity Tracking for Command {command}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize Pupper V3 RL policy")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to simulate")
    parser.add_argument("--render-every", type=int, default=2, help="Render every N steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config if args.config else None)
    
    # Load policy
    inference_fn, params = load_policy(args.checkpoint, config)
    
    # Define test commands
    commands = [
        (0.5, 0.0, 0.0),   # Forward
        (0.0, 0.5, 0.0),   # Sideways
        (0.0, 0.0, 1.0),   # Turn
        (0.5, 0.5, 0.0),   # Forward+Sideways
        (0.5, 0.0, 1.0),   # Forward+Turn
    ]
    
    # Visualize policy
    visualize_policy_rollout(
        config=config,
        params=params,
        inference_fn=inference_fn,
        commands=commands,
        output_dir=args.output_dir,
        n_steps=args.steps,
        render_every=args.render_every,
        seed=args.seed
    )
    
    print(f"Visualization completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
