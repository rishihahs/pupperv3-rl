"""Parameter conversion utilities for exporting policies."""

import json
import jax
import jax.numpy as jp
import numpy as np
from typing import Dict, Tuple, List, Any, Union, Optional

def convert_params(
    params: Dict[str, Any],
    activation: str = "elu",
    action_scale: Union[float, List[float]] = 0.75,
    kp: float = 5.0,
    kd: float = 0.25,
    default_pose: jp.ndarray = None,
    joint_upper_limits: List[float] = None,
    joint_lower_limits: List[float] = None,
    use_imu: bool = True,
    observation_history: int = 4,
    maximum_pitch_command: float = 0.0,
    maximum_roll_command: float = 0.0,
    final_activation: str = "tanh",
) -> Dict[str, Any]:
    """Converts policy parameters to RTNeural format.
    
    Args:
        params: Policy parameters.
        activation: Activation function.
        action_scale: Action scaling.
        kp: Position control proportional gain.
        kd: Position control derivative gain.
        default_pose: Default joint pose.
        joint_upper_limits: Upper limits for joint positions.
        joint_lower_limits: Lower limits for joint positions.
        use_imu: Whether to use IMU in observation.
        observation_history: Observation history length.
        maximum_pitch_command: Maximum pitch command in degrees.
        maximum_roll_command: Maximum roll command in degrees.
        final_activation: Final activation function.
        
    Returns:
        Dictionary with converted parameters.
    """
    # Extract policy network parameters
    policy_params = params['policy']
    
    # Convert activation function
    activation_mapping = {
        'relu': 'relu',
        'elu': 'elu',
        'tanh': 'tanh',
        'sigmoid': 'sigmoid',
        'swish': 'elu',  # RTNeural doesn't support swish, use elu as approximation
    }
    
    if activation not in activation_mapping:
        raise ValueError(f"Activation function {activation} not supported. "
                         f"Supported activations: {list(activation_mapping.keys())}")
    
    rt_activation = activation_mapping[activation]
    
    # Convert final activation
    if final_activation not in activation_mapping:
        raise ValueError(f"Final activation function {final_activation} not supported. "
                         f"Supported activations: {list(activation_mapping.keys())}")
    
    rt_final_activation = activation_mapping[final_activation]
    
    # Extract network structure
    network_structure = []
    
    # Process policy layers
    layer_idx = 0
    while f'Dense_{layer_idx}' in policy_params:
        layer_params = policy_params[f'Dense_{layer_idx}']
        
        if 'kernel' in layer_params and 'bias' in layer_params:
            weights = np.array(layer_params['kernel'])
            biases = np.array(layer_params['bias'])
            
            # For RTNeural, we need to transpose the weights
            weights = weights.T
            
            network_structure.append({
                'weights': weights.tolist(),
                'biases': biases.tolist(),
                'activation': rt_activation if layer_idx < len(policy_params) - 1 else rt_final_activation
            })
        
        layer_idx += 1
    
    # Create RTNeural parameter dictionary
    rtneural_params = {
        'layers': network_structure,
        'action_scale': action_scale if isinstance(action_scale, (int, float)) else list(action_scale),
        'kp': kp,
        'kd': kd,
        'default_pose': default_pose.tolist() if default_pose is not None else None,
        'joint_upper_limits': joint_upper_limits,
        'joint_lower_limits': joint_lower_limits,
        'use_imu': use_imu,
        'observation_history': observation_history,
        'maximum_pitch_command': maximum_pitch_command,
        'maximum_roll_command': maximum_roll_command,
    }
    
    return rtneural_params
