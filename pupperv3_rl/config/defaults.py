"""Default configuration values for Pupper V3 RL training."""

from ml_collections import config_dict
import jax.numpy as jp

def get_simulation_config():
    """Returns the default simulation configuration."""
    config = config_dict.ConfigDict()
    
    # Model configuration
    config.model_repo = 'https://github.com/g-levine/pupper_v3_description'
    config.model_branch = 'master'
    config.original_model_path = "pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml"
    config.model_path = "pupper_v3_description/description/mujoco_xml/model_with_obstacles.xml"
    
    # Model body names
    config.upper_leg_body_names = ["leg_front_r_2", "leg_front_l_2", "leg_back_r_2", "leg_back_l_2"]
    config.lower_leg_body_names = ["leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3"]
    config.foot_site_names = [
        "leg_front_r_3_foot_site",
        "leg_front_l_3_foot_site",
        "leg_back_r_3_foot_site",
        "leg_back_l_3_foot_site",
    ]
    config.torso_name = "base_link"
    
    # Foot radius
    config.foot_radius = 0.02
    
    # Collision detection
    config.max_contact_points = 20
    config.max_geom_pairs = 20
    
    # Physics timestep
    config.physics_dt = 0.004  # Physics dt [s]
    
    return config

def get_training_config():
    """Returns the default training configuration."""
    from pupperv3_mjx.domain_randomization import StartPositionRandomization
    
    config = config_dict.ConfigDict()
    
    # Checkpoint
    config.checkpoint_run_number = None
    
    # Environment timestep
    config.environment_dt = 0.02
    
    # PPO params
    config.ppo = config_dict.ConfigDict()
    config.ppo.num_timesteps = 500_000_000
    config.ppo.episode_length = 500
    config.ppo.num_evals = 11
    config.ppo.reward_scaling = 1
    config.ppo.normalize_observations = True
    config.ppo.action_repeat = 1
    config.ppo.unroll_length = 20
    config.ppo.num_minibatches = 32
    config.ppo.num_updates_per_batch = 4
    config.ppo.discounting = 0.97
    config.ppo.learning_rate = 3.0e-5
    config.ppo.entropy_cost = 1e-2
    config.ppo.num_envs = 8192
    config.ppo.batch_size = 256
    
    # Command sampling
    config.resample_velocity_step = config.ppo.episode_length // 2
    config.lin_vel_x_range = [-0.75, 0.75]
    config.lin_vel_y_range = [-0.5, 0.5]
    config.ang_vel_yaw_range = [-2.0, 2.0]
    config.zero_command_probability = 0.02
    config.stand_still_command_threshold = 0.05
    
    # Orientation command sampling in degrees
    config.maximum_pitch_command = 0.0
    config.maximum_roll_command = 0.0
    
    # Desired body orientation
    config.desired_world_z_in_body_frame = (0.0, 0.0, 1.0)
    
    # Termination
    config.terminal_body_z = 0.05
    config.terminal_body_angle = 0.70
    config.early_termination_step_threshold = config.ppo.episode_length // 2
    
    # Joint PD overrides
    config.dof_damping = 0.25
    config.position_control_kp = 5.0
    
    # Default joint angles
    config.default_pose = jp.array(
        [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52]
    )
    
    # Desired abduction angles
    config.desired_abduction_angles = jp.array([0.0, 0.0, 0.0, 0.0])
    
    # Height field
    config.height_field_random = False
    config.height_field_steps = False
    config.height_field_step_size = 4
    config.height_field_grid_size = 256
    config.height_field_group = "0"
    config.height_field_radius_x = 10.0
    config.height_field_radius_y = 10.0
    config.height_field_elevation_z = 0.02
    config.height_field_base_z = 0.2
    
    # Domain randomization
    config.kick_probability = 0.04
    config.kick_vel = 0.10
    config.angular_velocity_noise = 0.1
    config.gravity_noise = 0.05
    config.motor_angle_noise = 0.05
    config.last_action_noise = 0.01
    
    # Motors
    config.position_control_kp_multiplier_range = (0.6, 1.1)
    config.position_control_kd_multiplier_range = (0.8, 1.5)
    
    # Starting position
    config.start_position_config = StartPositionRandomization(
        x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, z_min=0.15, z_max=0.20
    )
    
    # Latency distribution
    config.latency_distribution = jp.array([0.2, 0.8])
    config.imu_latency_distribution = jp.array([0.5, 0.5])
    
    # Body CoM
    config.body_com_x_shift_range = (-0.02, 0.03)
    config.body_com_y_shift_range = (-0.005, 0.005)
    config.body_com_z_shift_range = (-0.005, 0.005)
    
    # Mass and inertia randomization
    config.body_mass_scale_range = (0.9, 1.3)
    config.body_inertia_scale_range = (0.9, 1.3)
    
    # Friction
    config.friction_range = (0.6, 1.4)
    
    # Obstacles
    config.n_obstacles = 0
    config.obstacle_x_range = (-3.0, 3.0)
    config.obstacle_y_range = (-3.0, 3.0)
    config.obstacle_height = 0.04
    config.obstacle_length = 2.0
    
    return config

def get_policy_config():
    """Returns the default policy configuration."""
    config = config_dict.ConfigDict()
    
    config.use_imu = True
    config.observation_history = 4
    config.action_scale = 0.75
    config.hidden_layer_sizes = (256, 128, 128, 128)
    config.activation = "elu"
    
    return config

def get_reward_config():
    """Returns the default reward configuration."""
    config = config_dict.ConfigDict()
    config.rewards = config_dict.ConfigDict()
    config.rewards.scales = config_dict.ConfigDict()
    
    # Track linear velocity
    config.rewards.scales.tracking_lin_vel = 1.5
    
    # Track the angular velocity along z-axis
    config.rewards.scales.tracking_ang_vel = 0.8
    
    # Track body orientation
    config.rewards.scales.tracking_orientation = 0.5
    
    # Regularization terms
    config.rewards.scales.lin_vel_z = -0.1
    config.rewards.scales.ang_vel_xy = -0.002
    config.rewards.scales.orientation = -0.0
    config.rewards.scales.torques = -0.025
    config.rewards.scales.joint_acceleration = -1e-6
    config.rewards.scales.mechanical_work = 0
    config.rewards.scales.action_rate = -0.1
    config.rewards.scales.feet_air_time = 0.02
    config.rewards.scales.stand_still = -0.00
    config.rewards.scales.stand_still_joint_velocity = -0.2
    config.rewards.scales.abduction_angle = -0.01
    config.rewards.scales.termination = -100.0
    config.rewards.scales.foot_slip = -0.2
    config.rewards.scales.knee_collision = -10.0
    config.rewards.scales.body_collision = -0.5
    
    # Tracking reward parameters
    config.rewards.tracking_sigma = 0.25
    
    return config

def get_default_config():
    """Returns the complete default configuration."""
    config = config_dict.ConfigDict()
    config.simulation = get_simulation_config()
    config.training = get_training_config()
    config.policy = get_policy_config()
    config.reward = get_reward_config()
    
    return config
