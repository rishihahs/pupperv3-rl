"""Configuration loading and processing utilities."""

import os
import importlib.util
from ml_collections import config_dict
from typing import Optional, Union, Dict, Any, Tuple
import xml.etree.ElementTree as ET
from etils import epath
import jax
import jax.numpy as jp

from brax import envs
from brax.io import mjcf
from pupperv3_mjx import environment, utils, obstacles

from pupperv3_rl.config.defaults import get_default_config


def load_config_from_file(file_path: str) -> config_dict.ConfigDict:
    """Loads configuration from a Python file.
    
    Args:
        file_path: Path to the Python configuration file.
        
    Returns:
        ConfigDict containing the configuration.
    """
    spec = importlib.util.spec_from_file_location("config", file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if hasattr(config_module, "get_config"):
        return config_module.get_config()
    else:
        raise ValueError(f"Configuration file {file_path} does not contain a get_config() function.")

def merge_configs(
    base_config: config_dict.ConfigDict,
    override_config: Optional[config_dict.ConfigDict] = None,
    override_dict: Optional[Dict[str, Any]] = None
) -> config_dict.ConfigDict:
    """Merges configurations with overrides.
    
    Args:
        base_config: Base configuration.
        override_config: Configuration to override the base.
        override_dict: Dictionary of key-value pairs to override.
        
    Returns:
        Merged configuration.
    """
    config = base_config.copy_and_resolve_references()
    
    if override_config is not None:
        # Recursively merge ConfigDict objects
        for key, value in override_config.items():
            if isinstance(value, config_dict.ConfigDict) and key in config:
                config[key] = merge_configs(config[key], value)
            else:
                config[key] = value
    
    if override_dict is not None:
        # Apply flat dictionary of overrides
        for key_path, value in override_dict.items():
            keys = key_path.split(".")
            curr_dict = config
            for key in keys[:-1]:
                if key not in curr_dict:
                    curr_dict[key] = config_dict.ConfigDict()
                curr_dict = curr_dict[key]
            curr_dict[keys[-1]] = value
    
    return config

def load_config(
    config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> config_dict.ConfigDict:
    """Loads and processes configuration.
    
    Args:
        config_file: Path to configuration file.
        overrides: Dictionary of key-value pairs to override.
        
    Returns:
        Processed configuration.
    """
    config = get_default_config()
    
    if config_file is not None and os.path.exists(config_file):
        file_config = load_config_from_file(config_file)
        config = merge_configs(config, file_config)
    
    if overrides is not None:
        config = merge_configs(config, override_dict=overrides)
    
    return config


def prepare_model(config: config_dict.ConfigDict) -> str:
    """Prepares model by modifying XML and adding elements.
    
    Args:
        config: Configuration.
        
    Returns:
        Path to prepared model.
    """
    
    # Load original model
    model_repo = config.simulation.model_repo
    model_branch = config.simulation.model_branch
    model_path = config.simulation.model_path
    
    # Clone model repository if not exists
    if not os.path.exists("pupper_v3_description"):
        os.system(f"git clone {model_repo} -b {model_branch}")
        os.system("cd pupper_v3_description && git pull")
    
    # Load XML
    original_model_path = config.simulation.original_model_path
    xml_str = epath.Path(original_model_path).read_text()
    
    # Parse the modified XML
    tree = ET.ElementTree(ET.fromstring(xml_str))
    
    # Set MJX options
    tree = utils.set_mjx_custom_options(tree, 
                                max_contact_points=config.simulation.max_contact_points,
                                max_geom_pairs=config.simulation.max_geom_pairs)
    
    # Add obstacles
    tree = obstacles.add_boxes_to_model(tree,
                             n_boxes=config.training.n_obstacles,
                             x_range=config.training.obstacle_x_range,
                             y_range=config.training.obstacle_y_range,
                             height=config.training.obstacle_height,
                             length=config.training.obstacle_length)
    
    # Add heightfield if enabled
    if config.training.height_field_random or config.training.height_field_steps:
        if config.training.height_field_random:
            heightfield = heightfield.create_random_heightfield(
                grid_size=config.training.height_field_grid_size,
                radius_x=config.training.height_field_radius_x,
                radius_y=config.training.height_field_radius_y,
                elevation_z=config.training.height_field_elevation_z,
                seed=0
            )
        
        if config.training.height_field_steps:
            heightfield = heightfield.create_step_heightfield(
                grid_size=config.training.height_field_grid_size,
                step_size=config.training.height_field_step_size,
                seed=0
            )
        
        tree = heightfield.add_heightfield_to_model(
            tree,
            heightfield=heightfield,
            grid_size=config.training.height_field_grid_size,
            radius_x=config.training.height_field_radius_x,
            radius_y=config.training.height_field_radius_y,
            elevation_z=config.training.height_field_elevation_z,
            base_z=config.training.height_field_base_z,
            group=config.training.height_field_group
        )
    
    # Convert back to string and write to file
    xml_output = ET.tostring(tree.getroot(), encoding='unicode')

    model_path = model_path + '.updated.xml'
    
    with open(model_path, 'w+') as file:
        file.write(xml_output)
    
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
    envs.register_environment(env_name, environment.PupperV3Env)
    
    # Get absolute path to model
    model_path = os.path.abspath(config.simulation.model_path)
    
    # Get joint limits from the model 
    try:
        sys_temp = mjcf.load(config.simulation.original_model_path)
        joint_upper_limits = sys_temp.jnt_range[1:, 1]
        joint_lower_limits = sys_temp.jnt_range[1:, 0]
        print(f"Successfully loaded model with brax.io.mjcf")
    except Exception as e:
        print(f"Error loading model with brax.io.mjcf: {e}")
    
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
    
    return env
