"""Pupper V3 Reinforcement Learning Environment."""

import functools
import os
import time
from typing import Dict, Tuple, Optional, Callable, Any, List, Union

import jax
import jax.numpy as jp
import numpy as np
from brax import base
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax import math
from etils import epath
from flax import struct
from mujoco import mjx

@struct.dataclass
class PupperState:
    """State of the Pupper environment."""
    pipeline_state: MjxState
    obs: jp.ndarray
    reward: jp.ndarray
    done: jp.ndarray
    metrics: Dict[str, jp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

class PupperV3Env(Env):
    """Pupper V3 Robot Environment."""

    def __init__(
        self,
        path: str,
        action_scale: Union[float, List[float]] = 0.75,
        observation_history: int = 4,
        joint_lower_limits: List[float] = None,
        joint_upper_limits: List[float] = None,
        dof_damping: float = 0.25,
        position_control_kp: float = 5.0,
        foot_site_names: List[str] = None,
        torso_name: str = "base_link",
        upper_leg_body_names: List[str] = None,
        lower_leg_body_names: List[str] = None,
        resample_velocity_step: int = 150,
        linear_velocity_x_range: Tuple[float, float] = (-0.75, 0.75),
        linear_velocity_y_range: Tuple[float, float] = (-0.5, 0.5),
        angular_velocity_range: Tuple[float, float] = (-2.0, 2.0),
        zero_command_probability: float = 0.02,
        stand_still_command_threshold: float = 0.05,
        maximum_pitch_command: float = 0.0,
        maximum_roll_command: float = 0.0,
        start_position_config = None,
        default_pose = None,
        desired_abduction_angles = None,
        reward_config = None,
        angular_velocity_noise: float = 0.1,
        gravity_noise: float = 0.05,
        motor_angle_noise: float = 0.05,
        last_action_noise: float = 0.01,
        kick_vel: float = 0.1,
        kick_probability: float = 0.04,
        terminal_body_z: float = 0.05,
        early_termination_step_threshold: int = 150,
        terminal_body_angle: float = 0.7,
        foot_radius: float = 0.02,
        environment_timestep: float = 0.02,
        physics_timestep: float = 0.004,
        latency_distribution = None,
        imu_latency_distribution = None,
        desired_world_z_in_body_frame = None,
        use_imu: bool = True,
        **kwargs
    ):
        """Initialize the Pupper environment.
        
        Args:
            path: Path to the Pupper model XML file.
            action_scale: Scale of actions (either a single value or per-joint).
            observation_history: Number of observation history steps to include.
            joint_lower_limits: Lower limits for joint positions.
            joint_upper_limits: Upper limits for joint positions.
            dof_damping: Joint damping coefficient.
            position_control_kp: Position control proportional gain.
            foot_site_names: Names of the foot site bodies.
            torso_name: Name of the torso body.
            upper_leg_body_names: Names of the upper leg bodies.
            lower_leg_body_names: Names of the lower leg bodies.
            resample_velocity_step: Step interval for resampling velocity commands.
            linear_velocity_x_range: Range for linear velocity in x direction.
            linear_velocity_y_range: Range for linear velocity in y direction.
            angular_velocity_range: Range for angular velocity.
            zero_command_probability: Probability of zero command.
            stand_still_command_threshold: Threshold for stand still command.
            maximum_pitch_command: Maximum pitch command in degrees.
            maximum_roll_command: Maximum roll command in degrees.
            start_position_config: Configuration for start position randomization.
            default_pose: Default joint pose.
            desired_abduction_angles: Desired abduction angles.
            reward_config: Reward configuration.
            angular_velocity_noise: Angular velocity noise scale.
            gravity_noise: Gravity noise scale.
            motor_angle_noise: Motor angle noise scale.
            last_action_noise: Last action noise scale.
            kick_vel: Kick velocity scale.
            kick_probability: Probability of kick.
            terminal_body_z: Terminal body height.
            early_termination_step_threshold: Step threshold for early termination.
            terminal_body_angle: Terminal body angle.
            foot_radius: Foot radius.
            environment_timestep: Environment timestep.
            physics_timestep: Physics timestep.
            latency_distribution: Action latency distribution.
            imu_latency_distribution: IMU latency distribution.
            desired_world_z_in_body_frame: Desired world z axis in body frame.
            use_imu: Whether to use IMU in observation.
        """
        # Load model
        model_path = epath.Path(path)
        xml_string = model_path.read_text()
        self.sys = mjx.load_model_from_xml(xml_string)
        
        # Store parameters
        self.action_scale = action_scale
        self.observation_history = observation_history
        self.joint_lower_limits = jp.array(joint_lower_limits) if joint_lower_limits else None
        self.joint_upper_limits = jp.array(joint_upper_limits) if joint_upper_limits else None
        self.dof_damping = dof_damping
        self.position_control_kp = position_control_kp
        
        # Find body and joint indices
        self.foot_site_ids = [self.sys.site(name).id for name in foot_site_names] if foot_site_names else []
        self.torso_id = self.sys.body(torso_name).id if torso_name else None
        self.upper_leg_ids = [self.sys.body(name).id for name in upper_leg_body_names] if upper_leg_body_names else []
        self.lower_leg_ids = [self.sys.body(name).id for name in lower_leg_body_names] if lower_leg_body_names else []
        
        # Command parameters
        self.resample_velocity_step = resample_velocity_step
        self.linear_velocity_x_range = linear_velocity_x_range
        self.linear_velocity_y_range = linear_velocity_y_range
        self.angular_velocity_range = angular_velocity_range
        self.zero_command_probability = zero_command_probability
        self.stand_still_command_threshold = stand_still_command_threshold
        self.maximum_pitch_command = maximum_pitch_command
        self.maximum_roll_command = maximum_roll_command
        
        # Default poses and desired positions
        self.start_position_config = start_position_config
        self.default_pose = default_pose if default_pose is not None else jp.zeros(self.sys.nu)
        self.desired_abduction_angles = desired_abduction_angles
        
        # Reward configuration
        self.reward_config = reward_config
        
        # Randomization parameters
        self.angular_velocity_noise = angular_velocity_noise
        self.gravity_noise = gravity_noise
        self.motor_angle_noise = motor_angle_noise
        self.last_action_noise = last_action_noise
        self.kick_vel = kick_vel
        self.kick_probability = kick_probability
        
        # Termination parameters
        self.terminal_body_z = terminal_body_z
        self.early_termination_step_threshold = early_termination_step_threshold
        self.terminal_body_angle = terminal_body_angle
        self.foot_radius = foot_radius
        
        # Timesteps
        self.dt = environment_timestep
        self.physics_dt = physics_timestep
        self.physics_steps_per_env_step = max(1, int(self.dt / self.physics_dt))
        
        # Latency parameters
        self.latency_distribution = latency_distribution
        self.imu_latency_distribution = imu_latency_distribution
        self.max_latency = len(latency_distribution) if latency_distribution is not None else 1
        self.max_imu_latency = len(imu_latency_distribution) if imu_latency_distribution is not None else 1
        
        # Orientation parameters
        self.desired_world_z_in_body_frame = desired_world_z_in_body_frame if desired_world_z_in_body_frame is not None else jp.array([0.0, 0.0, 1.0])
        self.use_imu = use_imu
        
        # Initialize observation and action dimensions
        self._reset_observation_dim()

    def _reset_observation_dim(self):
        """Resets observation dimension based on configuration."""
        # Base observation: joint positions, velocities, last action
        obs_dim = self.sys.nu * 2 + self.sys.nu
        
        # Add IMU observation if enabled
        if self.use_imu:
            # Projection of world axes in body frame (9 values)
            # Angular velocity (3 values)
            # Linear acceleration (3 values)
            obs_dim += 9 + 3 + 3
        
        # Add command dimensions
        # Linear velocity x, y and angular velocity z, which are set as tracking target
        obs_dim += 3
        
        # Desired orientation (projection of world z in body frame)
        obs_dim += 3
        
        # Gravity direction in body frame
        obs_dim += 3
        
        # Progress in the episode normalized to [0, 1]
        obs_dim += 1
        
        # Action latency as one-hot encoding
        obs_dim += self.max_latency
        
        # IMU latency as one-hot encoding
        obs_dim += self.max_imu_latency
        
        # History dimension
        self.obs_dim = obs_dim * self.observation_history
        
        # Action dimension
        self.action_dim = self.sys.nu

    def reset(self, rng: jp.ndarray) -> PupperState:
        """Resets the environment to an initial state.
        
        Args:
            rng: Random number generator key.
            
        Returns:
            Initial state.
        """
        # Split RNG keys
        rng, rng_init, rng_command = jax.random.split(rng, 3)
        
        # Initialize state
        state = self._initialize_state(rng_init)
        
        # Sample command
        command = self._sample_command(rng_command)
        
        # Create and initialize metrics
        metrics = self._initialize_metrics()
        
        # Create info dictionary
        info = {
            'command': command,
            'desired_world_z_in_body_frame': self.desired_world_z_in_body_frame,
            'step_count': jp.zeros((), dtype=int),
            'prev_action': jp.zeros(self.sys.nu),
            'action_history': jp.zeros((self.max_latency, self.sys.nu)),
            'imu_history': jp.zeros((self.max_imu_latency, 15)),  # IMU history buffer
            'last_foot_positions': jp.zeros((4, 3)),
            'foot_air_time': jp.zeros(4),
            'foot_contact': jp.zeros(4, dtype=bool),
            'prev_feet_positions': jp.zeros((4, 3)),
        }
        
        # Create observation
        obs = self._get_observation(state, info)
        
        return PupperState(
            pipeline_state=state,
            obs=obs,
            reward=jp.zeros(()),
            done=jp.zeros((), dtype=bool),
            metrics=metrics,
            info=info
        )
    
    def _initialize_state(self, rng: jp.ndarray) -> MjxState:
        """Initializes the MJX state.
        
        Args:
            rng: Random number generator key.
            
        Returns:
            Initialized MJX state.
        """
        # Initialize with default data
        state = mjx.get_state(self.sys)
        
        # Randomize starting position if config is provided
        if self.start_position_config:
            rng, rng_pos = jax.random.split(rng)
            
            # Sample random position
            x = jax.random.uniform(
                rng_pos, 
                (), 
                minval=self.start_position_config.x_min, 
                maxval=self.start_position_config.x_max
            )
            
            rng, rng_pos = jax.random.split(rng)
            y = jax.random.uniform(
                rng_pos, 
                (), 
                minval=self.start_position_config.y_min, 
                maxval=self.start_position_config.y_max
            )
            
            rng, rng_pos = jax.random.split(rng)
            z = jax.random.uniform(
                rng_pos, 
                (), 
                minval=self.start_position_config.z_min, 
                maxval=self.start_position_config.z_max
            )
            
            # Set torso position
            state = state.replace(
                qpos=state.qpos.at[:3].set(jp.array([x, y, z]))
            )
        
        # Set default joint positions
        if self.default_pose is not None:
            state = state.replace(
                qpos=state.qpos.at[7:].set(self.default_pose)
            )
        
        # Return initialized state
        return state
    
    def _sample_command(self, rng: jp.ndarray) -> jp.ndarray:
        """Samples a command.
        
        Args:
            rng: Random number generator key.
            
        Returns:
            Command as [linear_vel_x, linear_vel_y, angular_vel_z].
        """
        # Split RNG
        rng, rng_cmd = jax.random.split(rng)
        
        # Decide if command should be zero
        zero_command = jax.random.uniform(rng_cmd) < self.zero_command_probability
        
        # Split RNG for each command dimension
        rng, rng_x, rng_y, rng_z = jax.random.split(rng, 4)
        
        # Sample command
        cmd_x = jax.random.uniform(
            rng_x,
            (),
            minval=self.linear_velocity_x_range[0],
            maxval=self.linear_velocity_x_range[1]
        )
        
        cmd_y = jax.random.uniform(
            rng_y,
            (),
            minval=self.linear_velocity_y_range[0],
            maxval=self.linear_velocity_y_range[1]
        )
        
        cmd_z = jax.random.uniform(
            rng_z,
            (),
            minval=self.angular_velocity_range[0],
            maxval=self.angular_velocity_range[1]
        )
        
        # Create command
        command = jp.array([cmd_x, cmd_y, cmd_z])
        
        # Apply zero command if selected
        command = jp.where(zero_command, jp.zeros_like(command), command)
        
        return command
    
    def _initialize_metrics(self) -> Dict[str, jp.ndarray]:
        """Initializes metrics dictionary.
        
        Returns:
            Dictionary of metrics.
        """
        return {
            'episode_length': jp.zeros(()),
            'reward_tracking_lin_vel': jp.zeros(()),
            'reward_tracking_ang_vel': jp.zeros(()),
            'reward_tracking_orientation': jp.zeros(()),
            'reward_lin_vel_z': jp.zeros(()),
            'reward_ang_vel_xy': jp.zeros(()),
            'reward_orientation': jp.zeros(()),
            'reward_torques': jp.zeros(()),
            'reward_joint_acceleration': jp.zeros(()),
            'reward_mechanical_work': jp.zeros(()),
            'reward_action_rate': jp.zeros(()),
            'reward_feet_air_time': jp.zeros(()),
            'reward_stand_still': jp.zeros(()),
            'reward_stand_still_joint_velocity': jp.zeros(()),
            'reward_abduction_angle': jp.zeros(()),
            'reward_foot_slip': jp.zeros(()),
            'reward_foot_pos_diff': jp.zeros(()),
            'reward_knee_collision': jp.zeros(()),
            'reward_body_collision': jp.zeros(()),
            'reward_termination': jp.zeros(()),
            'lin_vel_x': jp.zeros(()),
            'lin_vel_y': jp.zeros(()),
            'ang_vel_z': jp.zeros(()),
            'command_x': jp.zeros(()),
            'command_y': jp.zeros(()),
            'command_z': jp.zeros(()),
        }

    def step(self, state: PupperState, action: jp.ndarray) -> PupperState:
        """Steps the environment.
        
        Args:
            state: Current state.
            action: Action to apply.
            
        Returns:
            Next state.
        """
        # Update info
        info = dict(state.info)
        info['step_count'] = info['step_count'] + 1
        
        # Update action history
        info['action_history'] = jp.roll(info['action_history'], shift=1, axis=0)
        info['action_history'] = info['action_history'].at[0].set(action)
        
        # Apply action latency
        action_with_latency = self._apply_action_latency(info['action_history'], info)
        
        # Scale and clip action
        scaled_action = self._scale_and_clip_action(action_with_latency)
        
        # Apply step
        pipeline_state = self._step_physics(state.pipeline_state, scaled_action, info)
        
        # Check termination
        done = self._check_termination(pipeline_state, info)
        
        # Compute reward
        reward, reward_components = self._compute_reward(pipeline_state, action_with_latency, info, done)
        
        # Update metrics
        metrics = dict(state.metrics)
        metrics['episode_length'] = info['step_count']
        
        # Update metrics with reward components
        for key, value in reward_components.items():
            if 'reward_' + key in metrics:
                metrics['reward_' + key] = value
        
        # Update command metrics
        metrics['lin_vel_x'] = pipeline_state.qvel[0]
        metrics['lin_vel_y'] = pipeline_state.qvel[1]
        metrics['ang_vel_z'] = pipeline_state.qvel[5]
        metrics['command_x'] = info['command'][0]
        metrics['command_y'] = info['command'][1]
        metrics['command_z'] = info['command'][2]
        
        # Resample command if necessary
        if info['step_count'] % self.resample_velocity_step == 0:
            rng = jax.random.PRNGKey(int(info['step_count']))
            info['command'] = self._sample_command(rng)
        
        # Get observation
        obs = self._get_observation(pipeline_state, info)
        
        # Update prev_action
        info['prev_action'] = action_with_latency
        
        return PupperState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info
        )

    def _apply_action_latency(self, action_history: jp.ndarray, info: Dict[str, Any]) -> jp.ndarray:
        """Applies latency to the action.
        
        Args:
            action_history: History of actions.
            info: Information dictionary.
            
        Returns:
            Action with latency applied.
        """
        if self.latency_distribution is None:
            return action_history[0]
        
        # Sample from latency distribution
        rng = jax.random.PRNGKey(int(info['step_count']))
        latency_idx = jax.random.choice(rng, len(self.latency_distribution), p=self.latency_distribution)
        
        # Get action with latency
        return action_history[latency_idx]

    def _scale_and_clip_action(self, action: jp.ndarray) -> jp.ndarray:
        """Scales and clips the action.
        
        Args:
            action: Action to scale and clip.
            
        Returns:
            Scaled and clipped action.
        """
        # Clip action
        action = jp.clip(action, -1.0, 1.0)
        
        # Scale action
        if isinstance(self.action_scale, (list, tuple, np.ndarray, jp.ndarray)):
            scaled_action = action * jp.array(self.action_scale)
        else:
            scaled_action = action * self.action_scale
        
        # Map to joint position command
        if self.joint_lower_limits is not None and self.joint_upper_limits is not None:
            joint_range = self.joint_upper_limits - self.joint_lower_limits
            joint_middle = (self.joint_upper_limits + self.joint_lower_limits) / 2.0
            position_command = scaled_action * joint_range / 2.0 + joint_middle
        else:
            position_command = scaled_action
        
        return position_command

    def _step_physics(self, state: MjxState, action: jp.ndarray, info: Dict[str, Any]) -> MjxState:
        """Steps the physics simulation.
        
        Args:
            state: Current MJX state.
            action: Action to apply.
            info: Information dictionary.
            
        Returns:
            Next MJX state.
        """
        # Store previous foot positions
        info['prev_feet_positions'] = info['last_foot_positions'].copy()
        
        # Apply control using PD controller
        for _ in range(self.physics_steps_per_env_step):
            # Compute torques using PD controller
            position_error = action - state.qpos[7:]
            velocity_error = -state.qvel[6:]
            
            control = (
                self.position_control_kp * position_error +
                self.dof_damping * velocity_error
            )
            
            # Step physics
            state = mjx.step(self.sys, state, control)
            
            # Update IMU history
            self._update_imu_history(state, info)
            
            # Update foot positions and contacts
            self._update_foot_positions_and_contacts(state, info)
        
        return state

    def _update_imu_history(self, state: MjxState, info: Dict[str, Any]) -> None:
        """Updates IMU history.
        
        Args:
            state: Current MJX state.
            info: Information dictionary.
        """
        if not self.use_imu:
            return
        
        # Get rotation matrix from body to world
        torso_orientation = self._get_body_rotation_matrix(state, self.torso_id)
        
        # Orientation: projection of world axes in body frame
        world_x_in_body = torso_orientation[:, 0]
        world_y_in_body = torso_orientation[:, 1]
        world_z_in_body = torso_orientation[:, 2]
        
        # Angular velocity in body frame
        ang_vel_body = torso_orientation.T @ state.qvel[3:6]
        
        # Linear acceleration in body frame
        # (Here we'd need to compute acceleration, but for simplicity we use velocity)
        lin_acc_body = torso_orientation.T @ state.qvel[0:3]
        
        # Combine IMU readings
        imu_data = jp.concatenate([
            world_x_in_body,
            world_y_in_body,
            world_z_in_body,
            ang_vel_body,
            lin_acc_body
        ])
        
        # Update IMU history
        info['imu_history'] = jp.roll(info['imu_history'], shift=1, axis=0)
        info['imu_history'] = info['imu_history'].at[0].set(imu_data)

    def _update_foot_positions_and_contacts(self, state: MjxState, info: Dict[str, Any]) -> None:
        """Updates foot positions and contacts.
        
        Args:
            state: Current MJX state.
            info: Information dictionary.
        """
        # Update foot positions
        foot_positions = jp.zeros((4, 3))
        foot_contacts = jp.zeros(4, dtype=bool)
        
        # Get foot positions and check contacts
        for i, site_id in enumerate(self.foot_site_ids):
            # Get foot position
            foot_pos = mjx.site_pos(self.sys, state, site_id)
            foot_positions = foot_positions.at[i].set(foot_pos)
            
            # Check if foot is in contact (simple height check)
            foot_contacts = foot_contacts.at[i].set(foot_pos[2] <= self.foot_radius)
        
        # Track air time
        air_time = info['foot_air_time'].copy()
        air_time = jp.where(foot_contacts, jp.zeros_like(air_time), air_time + self.dt)
        info['foot_air_time'] = air_time
        
        # Update foot contacts and positions
        info['foot_contact'] = foot_contacts
        info['last_foot_positions'] = foot_positions

    def _get_body_rotation_matrix(self, state: MjxState, body_id: int) -> jp.ndarray:
        """Gets rotation matrix from body to world frame.
        
        Args:
            state: Current MJX state.
            body_id: Body ID.
            
        Returns:
            Rotation matrix.
        """
        # Extract body orientation quaternion
        if body_id == 0:  # World body
            quat = jp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        else:
            # Get body orientation from state
            # This is a simplified version - in practice, we'd need to extract
            # the quaternion from the state which depends on the specific structure
            # For now, let's use a placeholder
            quat = jp.array([state.qpos[3], state.qpos[4], state.qpos[5], state.qpos[6]])
        
        # Convert quaternion to rotation matrix
        return quaternion_to_rotation_matrix(quat)

    def _check_termination(self, state: MjxState, info: Dict[str, Any]) -> jp.ndarray:
        """Checks if episode should terminate.
        
        Args:
            state: Current MJX state.
            info: Information dictionary.
            
        Returns:
            Boolean indicating if episode should terminate.
        """
        # Termination conditions
        torso_position = state.qpos[:3]
        torso_height = torso_position[2]
        
        # Check height termination
        height_termination = torso_height < self.terminal_body_z
        
        # Check orientation termination
        torso_orientation = quaternion_to_rotation_matrix(state.qpos[3:7])
        world_z_in_body = torso_orientation[:, 2]
        orientation_dot = jp.dot(world_z_in_body, jp.array([0., 0., 1.]))
        orientation_angle = jp.arccos(jp.clip(orientation_dot, -1.0, 1.0))
        orientation_termination = orientation_angle > self.terminal_body_angle
        
        # Early termination only after threshold
        early_termination = jp.logical_or(height_termination, orientation_termination)
        early_termination = jp.logical_and(
            early_termination,
            info['step_count'] > self.early_termination_step_threshold
        )
        
        # Time limit termination
        time_limit = info['step_count'] >= self.ppo.episode_length if hasattr(self, 'ppo') else False
        
        # Combined termination
        return jp.logical_or(early_termination, time_limit)

    def _compute_reward(
        self, 
        state: MjxState, 
        action: jp.ndarray, 
        info: Dict[str, Any], 
        done: jp.ndarray
    ) -> Tuple[jp.ndarray, Dict[str, jp.ndarray]]:
        """Computes reward.
        
        Args:
            state: Current MJX state.
            action: Current action.
            info: Information dictionary.
            done: Whether episode is done.
            
        Returns:
            Total reward and dictionary of reward components.
        """
        if self.reward_config is None:
            return jp.zeros(()), {}
            
        # Get reward scales
        reward_scales = self.reward_config.rewards.scales
        
        # Initialize reward components
        reward_components = {}
        
        # Tracking rewards
        command = info['command']
        lin_vel = state.qvel[:3]
        ang_vel = state.qvel[3:6]
        
        # Linear velocity tracking
        tracking_error = jp.sum(jp.square(lin_vel[:2] - command[:2]))
        tracking_sigma = self.reward_config.rewards.tracking_sigma
        reward_components['tracking_lin_vel'] = jp.exp(-tracking_error / tracking_sigma)
        
        # Angular velocity tracking
        ang_vel_error = jp.square(ang_vel[2] - command[2])
        reward_components['tracking_ang_vel'] = jp.exp(-ang_vel_error / tracking_sigma)
        
        # Orientation tracking
        if hasattr(info, 'desired_world_z_in_body_frame'):
            desired_z = info['desired_world_z_in_body_frame']
            torso_orientation = quaternion_to_rotation_matrix(state.qpos[3:7])
            world_z_in_body = torso_orientation[:, 2]
            orientation_error = jp.sum(jp.square(world_z_in_body - desired_z))
            reward_components['tracking_orientation'] = jp.exp(-orientation_error / tracking_sigma)
        else:
            reward_components['tracking_orientation'] = jp.zeros(())
        
        # Regularization rewards
        
        # Linear velocity z
        reward_components['lin_vel_z'] = -jp.square(lin_vel[2])
        
        # Angular velocity xy
        reward_components['ang_vel_xy'] = -jp.sum(jp.square(ang_vel[:2]))
        
        # Orientation
        torso_orientation = quaternion_to_rotation_matrix(state.qpos[3:7])
        body_roll_pitch = jp.array([
            jp.arcsin(torso_orientation[2, 0]),  # roll
            jp.arcsin(-torso_orientation[2, 1])  # pitch
        ])
        reward_components['orientation'] = -jp.sum(jp.square(body_roll_pitch))
        
        # Torques
        reward_components['torques'] = -jp.sum(jp.square(state.qfrc_actuator[6:]))
        
        # Joint acceleration
        reward_components['joint_acceleration'] = -jp.sum(jp.square(state.qacc[6:]))
        
        # Mechanical work
        joint_velocities = state.qvel[6:]
        joint_torques = state.qfrc_actuator[6:]
        reward_components['mechanical_work'] = -jp.sum(jp.abs(joint_velocities * joint_torques))
        
        # Action rate
        prev_action = info['prev_action']
        reward_components['action_rate'] = -jp.sum(jp.square(action - prev_action))
        
        # Feet air time
        reward_components['feet_air_time'] = jp.sum(info['foot_air_time'])
        
        # Stand still
        command_norm = jp.sqrt(jp.sum(jp.square(command)))
        stand_still_condition = command_norm < self.stand_still_command_threshold
        
        # Joint deviation from default
        joint_pos_error = jp.sum(jp.abs(state.qpos[7:] - self.default_pose))
        reward_components['stand_still'] = -jp.where(
            stand_still_condition,
            joint_pos_error,
            jp.zeros(())
        )
        
        # Joint velocity at stand still
        joint_vel_error = jp.sum(jp.square(state.qvel[6:]))
        reward_components['stand_still_joint_velocity'] = -jp.where(
            stand_still_condition,
            joint_vel_error,
            jp.zeros(())
        )
        
        # Abduction angle
        if self.desired_abduction_angles is not None:
            # Extract abduction joint angles (assuming they are the first of each leg)
            abduction_indices = jp.array([0, 3, 6, 9])  # Assuming 3 joints per leg
            abduction_angles = state.qpos[7:][abduction_indices]
            abduction_error = jp.sum(jp.square(abduction_angles - self.desired_abduction_angles))
            reward_components['abduction_angle'] = -abduction_error
        else:
            reward_components['abduction_angle'] = jp.zeros(())
        
        # Foot slip
        foot_velocities = (info['last_foot_positions'] - info['prev_feet_positions']) / self.dt
        foot_slip = jp.sum(jp.where(
            info['foot_contact'],
            jp.sum(jp.square(foot_velocities[:, :2]), axis=1),
            jp.zeros(4)
        ))
        reward_components['foot_slip'] = -foot_slip
        
        # Foot position difference (to prevent too large steps)
        foot_pos_diff = jp.sum(jp.square(info['last_foot_positions'] - info['prev_feet_positions']))
        reward_components['foot_pos_diff'] = -foot_pos_diff
        
        # Knee collision (simplistic check)
        # In practice, would need proper collision detection
        reward_components['knee_collision'] = jp.zeros(())
        
        # Body collision (simplistic check)
        # In practice, would need proper collision detection
        reward_components['body_collision'] = jp.zeros(())
        
        # Termination penalty
        reward_components['termination'] = jp.where(done, -1.0, 0.0)
        
        # Compute total reward
        total_reward = jp.zeros(())
        for key, value in reward_components.items():
            if key in reward_scales:
                total_reward += reward_scales[key] * value
                
        return total_reward, reward_components

    def _get_observation(self, state: MjxState, info: Dict[str, Any]) -> jp.ndarray:
        """Computes observation.
        
        Args:
            state: Current MJX state.
            info: Information dictionary.
            
        Returns:
            Observation array.
        """
        # Base observation components
        joint_pos = state.qpos[7:]  # Joint positions
        joint_vel = state.qvel[6:]  # Joint velocities
        prev_action = info['prev_action']  # Previous action
        
        observation_components = [joint_pos, joint_vel, prev_action]
        
        # IMU data
        if self.use_imu:
            # Apply IMU latency
            if self.imu_latency_distribution is not None:
                rng = jax.random.PRNGKey(int(info['step_count']))
                latency_idx = jax.random.choice(rng, len(self.imu_latency_distribution), p=self.imu_latency_distribution)
                imu_data = info['imu_history'][latency_idx]
            else:
                imu_data = info['imu_history'][0]
            
            # Add IMU data to observation
            observation_components.append(imu_data)
        
        # Command
        observation_components.append(info['command'])
        
        # Desired orientation
        observation_components.append(info['desired_world_z_in_body_frame'])
        
        # Gravity direction in body frame
        # Simplification: use world_z_in_body from IMU data
        if self.use_imu:
            gravity_dir = imu_data[6:9]  # world_z_in_body
        else:
            # As fallback, use identity
            gravity_dir = jp.array([0., 0., 1.])
        observation_components.append(gravity_dir)
        
        # Progress
        if hasattr(self, 'ppo'):
            progress = jp.array([info['step_count'] / self.ppo.episode_length])
        else:
            progress = jp.array([0.0])
        observation_components.append(progress)
        
        # Action latency as one-hot
        if self.latency_distribution is not None:
            # For simplicity, use previous latency (in practice, would need tracking)
            latency_onehot = jp.zeros(self.max_latency)
            latency_onehot = latency_onehot.at[0].set(1.0)  # Placeholder
            observation_components.append(latency_onehot)
        
        # IMU latency as one-hot
        if self.imu_latency_distribution is not None:
            # For simplicity, use previous latency (in practice, would need tracking)
            imu_latency_onehot = jp.zeros(self.max_imu_latency)
            imu_latency_onehot = imu_latency_onehot.at[0].set(1.0)  # Placeholder
            observation_components.append(imu_latency_onehot)
        
        # Create current observation
        current_obs = jp.concatenate(observation_components)
        
        # If first step, create history buffer
        if 'observation_history' not in info:
            info['observation_history'] = jp.tile(current_obs, (self.observation_history, 1))
        else:
            # Update history
            info['observation_history'] = jp.roll(info['observation_history'], shift=1, axis=0)
            info['observation_history'] = info['observation_history'].at[0].set(current_obs)
        
        # Flatten history into single observation
        return jp.ravel(info['observation_history'])

def quaternion_to_rotation_matrix(q: jp.ndarray) -> jp.ndarray:
    """Converts quaternion to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z].
        
    Returns:
        3x3 rotation matrix.
    """
    # Normalize quaternion
    q = q / jp.sqrt(jp.sum(jp.square(q)))
    
    # Extract components
    w, x, y, z = q
    
    # Compute rotation matrix
    xx, xy, xz = x*x, x*y, x*z
    yy, yz, zz = y*y, y*z, z*z
    wx, wy, wz = w*x, w*y, w*z
    
    return jp.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])
