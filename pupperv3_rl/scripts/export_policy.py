#!/usr/bin/env python
"""Export script for Pupper V3 RL policy."""

import os
import argparse
import json
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

from pupperv3_rl.config.loader import load_config
from pupperv3_rl.export.converter import convert_params

# For policy loading
from brax.training.agents.ppo import networks as ppo_networks


def load_policy_params(checkpoint_path: str):
    """Loads policy parameters from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        Policy parameters dictionary.
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load params
    with np.load(checkpoint_path) as data:
        params = {key: data[key] for key in data.keys()}
    
    return params


def export_policy(
    params: dict,
    config: config_dict.ConfigDict,
    output_path: str,
    final_activation: str = "tanh"
):
    """Exports policy to RTNeural format.
    
    Args:
        params: Policy parameters.
        config: Configuration.
        output_path: Output file path.
        final_activation: Final activation function.
    """
    # Convert parameters
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
        final_activation=final_activation,
    )
    
    # Save parameters
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rtneural_params, f, indent=2)
    
    print(f"Exported policy to {output_path}")
    
    # Print policy structure
    policy_layers = rtneural_params['layers']
    print("\nPolicy structure:")
    for i, layer in enumerate(policy_layers):
        print(f"  Layer {i+1}: Input {len(layer['biases'])} -> Output {len(layer['weights'])} (Activation: {layer['activation']})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export Pupper V3 RL policy")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default="policy.json", help="Output file path")
    parser.add_argument("--final-activation", type=str, default="tanh", help="Final layer activation function")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config if args.config else None)
    
    # Load policy parameters
    params = load_policy_params(args.checkpoint)
    
    # Export policy
    export_policy(
        params=params,
        config=config,
        output_path=args.output,
        final_activation=args.final_activation
    )


if __name__ == "__main__":
    main()
