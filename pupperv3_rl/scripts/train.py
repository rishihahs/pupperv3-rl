#!/usr/bin/env python
"""Training script for Pupper V3 RL."""

import os
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

from pupperv3_rl.config.loader import load_config
from pupperv3_rl.training.utils import activation_fn_map, progress, save_checkpoint, visualize_policy, download_checkpoint
from pupperv3_rl.models.modifications import set_mjx_custom_options
from pupperv3_rl.models.obstacles import add_boxes_to_model
from pupperv3_rl.models.heightfield import create_random_heightfield, create_step_heightfield, add_heightfield_to_model
from pupperv3_rl.environment.randomization import domain_randomize
from pupperv3_rl.export.converter import convert_params

# Import these to register the environment
from pupperv3_rl.environment.pupperv3_env import PupperV3Env

# For PPO training
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


def prepare_model(config: config_dict.ConfigDict) -> str:
    """Prepares model by modifying XML and adding elements.
    
    Args:
        config: Configuration.
        
    Returns:
        Path to prepared model.
    """
    import xml.etree.ElementTree as ET
    from etils import epath
    
    # Load original model
    model_repo = config.simulation.model_repo
    model_branch = config.simulation.model_branch
    model_path = config.simulation.model_path
    
    # Clone model repository if not exists
    if not os.path.exists("pupper_v3_description"):
        os.system(f"git clone {model_repo} -b {model_branch}")
        os.system("cd pupper_v3_description && git pull")
    
    # Load XML
    original_model_path = "pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml"
    xml_str = epath.Path(original_model_path).read_text()
    tree = ET.ElementTree(ET.fromstring(xml_str))
    
    # Set MJX options
    tree = set_mjx_custom_options(tree, 
                                max_contact_points=config.simulation.max_contact_points,
                                max_geom_pairs=config.simulation.max_geom_pairs)
    
    # Add obstacles
    tree = add_boxes_to_model(tree,
                             n_boxes=config.training.n_obstacles,
                             x_range=config.training.obstacle_x_range,
                             y_range=config.training.obstacle_y_range,
                             height=config.training.obstacle_height,
                             length=config.training.obstacle_length)
    
    # Add heightfield if enabled
    if config.training.height_field_random or config.training.height_field_steps:
        if config.training.height_field_random:
            heightfield = create_random_heightfield(
                grid_size=config.training.height_field_grid_size,
                radius_x=config.training.height_field_radius_x,
                radius_y=config.training.height_field_radius_y,
                elevation_z=config.training.height_field_elevation_z,
                seed=0
            )
        
        if config.training.height_field_steps:
            heightfield = create_step_heightfield(
                grid_size=config.training.height_field_grid_size,
                step_size=config.training.height_field_step_size,
                seed=0
            )
        
        tree = add_heightfield_to_model(
            tree,
            heightfield=heightfield,
            grid_size=config.training.height_field_grid_size,
            radius_x=config.training.height_field_radius_x,
            radius_y=config.training.height_field_radius_y,
            elevation_z=config.training.height_field_elevation_z,
            base_z=config.training.height_field_base_z,
            group=config.training.height_field_group
        )
    
    # Create directory for models if not exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Write modified model
    with open(model_path, 'w+') as file:
        tree.write(file, encoding='unicode')
    
    return model_path


def setup_environment(config: config_dict.ConfigDict, env_name: str = 'pupper') -> Tuple[Any, Any]:
    """Sets up training and evaluation environments.
    
    Args:
        config: Configuration.
        env_name: Environment name.
        
    Returns:
        Training and evaluation environments.
    """
    # Register environment
    envs.register_environment(env_name, PupperV3Env)
    
    # Determine joint limits
    import mjcf
    sys_temp = mjcf.load(config.simulation.model_path)
    joint_upper_limits = sys_temp.jnt_range[1:, 1]
    joint_lower_limits = sys_temp.jnt_range[1:, 0]
    
    # Create environment arguments
    env_kwargs = dict(
        path=config.simulation.model_path,
        action_scale=config.policy.action_scale,
        observation_history=config.policy.observation_history,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
        dof_damping=config.training.dof_damping,
        position_control_kp=config.training.position_control_kp,
        foot_site_names=config.simulation.foot_site_names,
        torso_name=config.simulation.torso_name,
        upper_leg_body_names=config.simulation.upper_leg_body_names,
        lower_leg_body_names=config.simulation.lower_leg_body_names,
        resample_velocity_step=config.training.resample_velocity_step,
        linear_velocity_x_range=config.training.lin_vel_x_range,
        linear_velocity_y_range=config.training.lin_vel_y_range,
        angular_velocity_range=config.training.ang_vel_yaw_range,
        zero_command_probability=config.training.zero_command_probability,
        stand_still_command_threshold=config.training.stand_still_command_threshold,
        maximum_pitch_command=config.training.maximum_pitch_command,
        maximum_roll_command=config.training.maximum_roll_command,
        start_position_config=config.training.start_position_config,
        default_pose=config.training.default_pose,
        desired_abduction_angles=config.training.desired_abduction_angles,
        reward_config=config.reward,
        angular_velocity_noise=config.training.angular_velocity_noise,
        gravity_noise=config.training.gravity_noise,
        motor_angle_noise=config.training.motor_angle_noise,
        last_action_noise=config.training.last_action_noise,
        kick_vel=config.training.kick_vel,
        kick_probability=config.training.kick_probability,
        terminal_body_z=config.training.terminal_body_z,
        early_termination_step_threshold=config.training.early_termination_step_threshold,
        terminal_body_angle=config.training.terminal_body_angle,
        foot_radius=config.simulation.foot_radius,
        environment_timestep=config.training.environment_dt,
        physics_timestep=config.simulation.physics_dt,
        latency_distribution=config.training.latency_distribution,
        imu_latency_distribution=config.training.imu_latency_distribution,
        desired_world_z_in_body_frame=jp.array(config.training.desired_world_z_in_body_frame),
        use_imu=config.policy.use_imu,
    )
    
    # Create environments
    env = envs.get_environment(env_name, **env_kwargs)
    eval_env = envs.get_environment(env_name, **env_kwargs)
    
    return env, eval_env


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
    env, eval_env = setup_environment(config)
    
    # Create JIT'd functions for visualization
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    
    # Create training function
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.policy.hidden_layer_sizes,
        activation=activation_fn_map(config.policy.activation)
    )
    
    train_fn = functools.partial(
        ppo.train,
        **(config.training.ppo.to_dict()),
        network_factory=make_networks_factory,
        randomization_fn=functools.partial(
            domain_randomize,
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
        visualize_policy(
            current_step=current_step,
            make_policy=make_policy,
            params=params,
            eval_env=eval_env,
            jit_step=jit_step,
            jit_reset=jit_reset,
            output_folder=output_dir
        )
        
        save_checkpoint(
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
        
        download_checkpoint(
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
            progress,
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
    model_path = os.path.join(output_dir, f'final_model_{timestamp}.npz')
    np.savez(model_path, **params)
    
    # Log final model to W&B
    if wandb_config:
        wandb.save(model_path)
    
    return make_inference_fn, params, stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Pupper V3 RL policy")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--wandb-entity", type=str, help="W&B entity name")
    parser.add_argument("--wandb-project", type=str, default="pupperv3-mjx-rl", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, help="W&B run name")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config if args.config else None)
    
    # Configure W&B
    wandb_config = None
    if args.wandb_entity or args.wandb_project:
        wandb_config = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.wandb_run_name,
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
    rtneural_params = convert_params(
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
