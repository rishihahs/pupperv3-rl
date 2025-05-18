"""Training utilities for Pupper V3 RL."""

import os
import time
import dataclasses
import functools
import json
from typing import Dict, Tuple, List, Callable, Any, Optional
from datetime import datetime

import jax
import jax.numpy as jp
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import wandb
from flax import traverse_util


def activation_fn_map(activation_name: str) -> Callable:
    """Maps activation name to function.
    
    Args:
        activation_name: Name of activation function.
        
    Returns:
        Activation function.
    """
    activation_map = {
        'relu': jax.nn.relu,
        'elu': jax.nn.elu,
        'tanh': jax.nn.tanh,
        'sigmoid': jax.nn.sigmoid,
        'swish': jax.nn.swish,
    }
    
    if activation_name not in activation_map:
        raise ValueError(f"Activation function {activation_name} not supported. "
                         f"Supported activations: {list(activation_map.keys())}")
    
    return activation_map[activation_name]


def progress(
    num_steps: int, 
    metrics: Dict[str, jp.ndarray], 
    x_data: List, 
    y_data: List, 
    ydataerr: List, 
    times: List[datetime],
    num_timesteps: int,
    min_y: float = 0,
    max_y: float = 40,
) -> None:
    """Updates and logs training progress.
    
    Args:
        num_steps: Current training step.
        metrics: Dictionary of metrics.
        x_data: List of x data points for plotting.
        y_data: List of y data points for plotting.
        ydataerr: List of y data error bars for plotting.
        times: List of timestamps.
        num_timesteps: Total number of timesteps.
        min_y: Minimum y value for plotting.
        max_y: Maximum y value for plotting.
    """
    # Update data
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])
    
    # Update times list
    times.append(datetime.now())
    
    # Log to W&B if available
    try:
        wandb.log(metrics, step=num_steps)
    except Exception as e:
        print(f"Error logging to W&B: {e}")
    
    # Print progress
    print(f"Step: {num_steps}/{num_timesteps}, Reward: {metrics['eval/episode_reward']:.4f} ± {metrics['eval/episode_reward_std']:.4f}")
    
    # Plot progress if at least 2 data points
    if len(x_data) >= 2:
        plt.figure(figsize=(10, 5))
        plt.errorbar(x_data, y_data, yerr=ydataerr, fmt='-o')
        plt.xlabel('Training steps')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.grid(True)
        plt.ylim(min_y, max_y)
        plt.tight_layout()
        plt.show()


def save_checkpoint(
    current_step: int,
    make_policy: Callable,
    params: Dict[str, Any],
    checkpoint_path: str
) -> None:
    """Saves policy checkpoint.
    
    Args:
        current_step: Current training step.
        make_policy: Function to create policy.
        params: Policy parameters.
        checkpoint_path: Path to save checkpoint.
    """
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save parameters
    params_file = os.path.join(checkpoint_path, f'params_{current_step}.npz')
    np.savez(params_file, **params)
    
    # Log checkpoint to W&B
    try:
        artifact = wandb.Artifact(f'checkpoint_{current_step}', type='model')
        artifact.add_file(params_file)
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"Error logging artifact to W&B: {e}")


def visualize_policy(
    current_step: int,
    make_policy: Callable,
    params: Dict[str, Any],
    eval_env: Any,
    jit_step: Callable,
    jit_reset: Callable,
    output_folder: str,
    commands: List[Tuple[float, float, float]] = None,
    n_steps: int = 200,
    render_every: int = 2,
    seed: int = 0
) -> None:
    """Visualizes policy.
    
    Args:
        current_step: Current training step.
        make_policy: Function to create policy.
        params: Policy parameters.
        eval_env: Evaluation environment.
        jit_step: JIT-compiled step function.
        jit_reset: JIT-compiled reset function.
        output_folder: Output folder for videos.
        commands: List of commands to test.
        n_steps: Number of steps to simulate.
        render_every: Render every N steps.
        seed: Random seed.
    """
    # Create inference function
    inference_fn = make_policy(params)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Default commands if none provided
    if commands is None:
        commands = [
            (0.5, 0.0, 0.0),   # Forward
            (0.0, 0.5, 0.0),   # Sideways
            (0.0, 0.0, 1.0),   # Turn
        ]
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Visualize each command
    for cmd_idx, command in enumerate(commands):
        # Initialize state
        rng = jax.random.PRNGKey(seed + cmd_idx)
        state = jit_reset(rng)
        state.info['command'] = jp.array(command)
        rollout = [state.pipeline_state]
        
        # Collect trajectory
        for i in range(n_steps):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state.pipeline_state)
        
        # Render video
        video = eval_env.render(rollout[::render_every], camera='tracking_cam')
        
        # Save video
        video_path = os.path.join(output_folder, f'policy_{current_step}_cmd_{cmd_idx}.mp4')
        media.write_video(video_path, video, fps=1.0 / eval_env.dt / render_every)
        
        # Log video to W&B
        try:
            wandb.log({f"video_cmd_{cmd_idx}": wandb.Video(video_path, fps=1.0 / eval_env.dt / render_every, format="mp4")}, step=current_step)
        except Exception as e:
            print(f"Error logging video to W&B: {e}")
        
        print(f"Saved video for command {command} at {video_path}")


def download_checkpoint(
    entity_name: str,
    project_name: str,
    run_number: int,
    save_path: str
) -> None:
    """Downloads checkpoint from W&B.
    
    Args:
        entity_name: W&B entity name.
        project_name: W&B project name.
        run_number: Run number.
        save_path: Path to save checkpoint.
    """
    try:
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize W&B
        api = wandb.Api()
        
        # Find run
        runs = api.runs(f"{entity_name}/{project_name}")
        run = None
        
        for r in runs:
            if str(run_number) in r.name:
                run = r
                break
        
        if run is None:
            print(f"Run {run_number} not found in project {entity_name}/{project_name}")
            return
        
        # Find checkpoint artifact
        artifacts = run.logged_artifacts()
        checkpoint_artifact = None
        
        for artifact in artifacts:
            if artifact.type == 'model' and 'checkpoint' in artifact.name:
                checkpoint_artifact = artifact
                break
        
        if checkpoint_artifact is None:
            print(f"No checkpoint artifact found for run {run_number}")
            return
        
        # Download checkpoint
        artifact_dir = checkpoint_artifact.download(root=save_path)
        print(f"Downloaded checkpoint to {artifact_dir}")
        
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
