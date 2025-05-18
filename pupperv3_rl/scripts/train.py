#!/usr/bin/env python
"""Training script for Pupper V3 RL."""

import os
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += (
    ' --xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
os.environ['XLA_FLAGS'] = xla_flags
os.environ['MUJOCO_GL'] = 'egl'

import argparse
import time
from datetime import datetime
import functools
from pathlib import Path
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jp
import numpy as np
import wandb
from ml_collections import config_dict
from brax import envs
from brax.io import mjcf, model
from etils import epath

from pupperv3_rl.config.loader import load_config, prepare_model, setup_environment
from pupperv3_rl.scene import heightfield
from pupperv3_mjx import environment, domain_randomization, obstacles, utils, export

# For PPO training
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


def train(
    config: config_dict.ConfigDict,
    output_dir: str,
    wandb_config: Dict[str, Any] = None
) -> Tuple[Any, Dict[str, Any], Any]:
    """Trains policy.
    
    Args:
        config: Configuration.
        output_dir: Output directory.
        wandb_config: Weights & Biases configuration.
        
    Returns:
        Inference function factory, parameters, and statistics.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize W&B
    if wandb_config:
        wandb.init(**wandb_config)
        
    # Prepare model
    model_path = prepare_model(config)
    config.simulation.model_path = model_path
    
    # Setup environment
    env = setup_environment(config)
    viz_env = setup_environment(config)
    eval_env = setup_environment(config)
    
    # Create JIT'd functions for visualization
    jit_viz_reset = jax.jit(viz_env.reset)
    jit_viz_step = jax.jit(viz_env.step)
    
    # Create training function
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.policy.hidden_layer_sizes,
        activation=utils.activation_fn_map(config.policy.activation)
    )
    
    train_fn = functools.partial(
        ppo.train,
        **(config.training.ppo.to_dict()),
        network_factory=make_networks_factory,
        randomization_fn=functools.partial(
            domain_randomization.domain_randomize,
            friction_range=config.training.friction_range,
            kp_multiplier_range=config.training.position_control_kp_multiplier_range,
            kd_multiplier_range=config.training.position_control_kd_multiplier_range,
            body_com_x_shift_range=config.training.body_com_x_shift_range,
            body_com_y_shift_range=config.training.body_com_y_shift_range,
            body_com_z_shift_range=config.training.body_com_z_shift_range,
            body_mass_scale_range=config.training.body_mass_scale_range,
            body_inertia_scale_range=config.training.body_inertia_scale_range,
        ),
        seed=42,
    )
    
    # Initialize tracking variables
    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    
    # Define policy parameters callback
    def policy_params_fn(current_step, make_policy, params):
        utils.visualize_policy(
            current_step=current_step,
            make_policy=make_policy,
            params=params,
            eval_env=eval_env,
            jit_step=jit_viz_step,
            jit_reset=jit_viz_reset,
            output_folder=output_dir
        )
        
        utils.save_checkpoint(
            current_step=current_step,
            make_policy=make_policy,
            params=params,
            checkpoint_path=output_dir
        )
    
    # Configure checkpoint loading if needed
    checkpoint_kwargs = {}
    if config.training.checkpoint_run_number is not None:
        # Get W&B entity if configured
        entity_name = wandb_config.get('entity') if wandb_config else None
        
        utils.download_checkpoint(
            entity_name=entity_name,
            project_name=wandb_config.get('project') if wandb_config else "pupperv3-mjx-rl",
            run_number=config.training.checkpoint_run_number,
            save_path="checkpoint"
        )
        
        checkpoint_kwargs["restore_checkpoint_path"] = Path("checkpoint").resolve()
    
    # Train policy
    make_inference_fn, params, stats = train_fn(
        environment=env,
        progress_fn=functools.partial(
            utils.progress,
            times=times,
            x_data=x_data,
            y_data=y_data,
            ydataerr=ydataerr,
            num_timesteps=config.training.ppo.num_timesteps,
            min_y=0,
            max_y=40,
        ),
        eval_env=eval_env,
        policy_params_fn=policy_params_fn,
        **checkpoint_kwargs
    )
    
    # Log training time
    if wandb_config:
        wandb.run.summary["time_to_jit"] = (times[1] - times[0]).total_seconds()
        wandb.run.summary["time_to_train"] = (times[-1] - times[1]).total_seconds()
    
    # Save final model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = os.path.join(output_dir, f'final_model_{timestamp}')
    model.save_params(model_path, params)
    
    # Log final model to W&B
    if wandb_config:
        wandb.save(model_path)
    
    return make_inference_fn, params, stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Pupper V3 RL policy")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--wandb-entity", type=str, default="rishi-shah", help="W&B entity name")
    parser.add_argument("--wandb-project", type=str, default="pupperv3-mjx-rl", help="W&B project name")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config if args.config else None)
    
    # Configure W&B
    wandb_config = None
    if args.wandb_entity or args.wandb_project:
        wandb_config = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "config": config.to_dict(),
            "save_code": True,
        }
    
    # Train policy
    make_inference_fn, params, stats = train(
        config=config,
        output_dir=args.output_dir,
        wandb_config=wandb_config
    )
    
    # Export policy for RTNeural
    rtneural_params = export.convert_params(
        params=params,
        activation=config.policy.activation,
        action_scale=config.policy.action_scale,
        kp=config.training.position_control_kp,
        kd=config.training.dof_damping,
        default_pose=config.training.default_pose,
        joint_upper_limits=config.simulation.joint_upper_limits,
        joint_lower_limits=config.simulation.joint_lower_limits,
        use_imu=config.policy.use_imu,
        observation_history=config.policy.observation_history,
        maximum_pitch_command=config.training.maximum_pitch_command,
        maximum_roll_command=config.training.maximum_roll_command,
        final_activation="tanh",
    )
    
    # Save RTNeural parameters
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rtneural_path = os.path.join(args.output_dir, f'rtneural_policy_{timestamp}.json')
    with open(rtneural_path, "w") as f:
        import json
        json.dump(rtneural_params, f)
    
    # Log RTNeural parameters to W&B
    if wandb_config:
        wandb.save(rtneural_path)
        wandb.finish()
    
    print(f"Training completed. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
