#!/usr/bin/env python
"""Simple testing script for Pupper V3 RL package."""

import os
import jax
import jax.numpy as jp
import numpy as np
import mediapy as media
from brax import envs

from pupperv3_rl.config import load_config
from pupperv3_rl.environment import PupperV3Env


def main():
    """Main entry point."""
    print("Testing Pupper V3 RL environment...")
    
    # Load configuration
    config = load_config()
    
    # Register environment
    envs.register_environment('pupper', PupperV3Env)
    
    # Prepare model XML path
    model_path = "pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.no_body.self_collision.two_iterations.xml"
    
    # Clone model repository if not exists
    if not os.path.exists("pupper_v3_description"):
        os.system(f"git clone {config.simulation.model_repo} -b {config.simulation.model_branch}")
        os.system("cd pupper_v3_description && git pull")
    
    # Determine joint limits from model
    import mjcf
    sys_temp = mjcf.load(model_path)
    joint_upper_limits = sys_temp.jnt_range[1:, 1]
    joint_lower_limits = sys_temp.jnt_range[1:, 0]
    
    # Environment arguments
    env_kwargs = dict(
        path=model_path,
        action_scale=config.policy.action_scale,
        observation_history=config.policy.observation_history,
        joint_lower_limits=np.array(joint_lower_limits).tolist(),
        joint_upper_limits=np.array(joint_upper_limits).tolist(),
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
    
    # Create environment
    print("Creating environment...")
    env = envs.get_environment('pupper', **env_kwargs)
    
    # JIT functions
    print("Compiling functions...")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # Run a simple simulation
    print("Running simulation...")
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    # Set command to walk forward
    state.info['command'] = jp.array([0.5, 0.0, 0.0])
    rollout = [state.pipeline_state]
    
    # Collect trajectory with sinusoidal actions
    n_steps = 100
    for i in range(n_steps):
        # Simple sinusoidal pattern
        ctrl = jp.ones(shape=(env.sys.nu,)) * jp.sin(i * env.dt * 2 * 3.14 * 2) * 0.25
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
    
    # Render and display video
    print("Rendering video...")
    video = env.render(rollout[::2], camera='tracking_cam')
    
    # Save video
    os.makedirs("test_output", exist_ok=True)
    video_path = "test_output/pupper_test.mp4"
    media.write_video(video_path, video, fps=1.0 / env.dt / 2)
    
    print(f"Test completed. Video saved to {video_path}")
    
    # Print observation shape
    print(f"Observation shape: {state.obs.shape}")
    
    # Print reward
    print(f"Final reward: {state.reward}")
    
    return 0


if __name__ == "__main__":
    main()
