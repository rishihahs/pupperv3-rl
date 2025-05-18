"""Domain randomization utilities."""

import jax
import jax.numpy as jp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

@dataclass
class StartPositionRandomization:
    """Configuration for randomizing start positions."""
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0
    z_min: float = 0.15
    z_max: float = 0.20

def domain_randomize(
    params: Dict[str, Any],
    state,
    rng: jax.random.PRNGKey,
    friction_range: Tuple[float, float] = (0.6, 1.4),
    kp_multiplier_range: Tuple[float, float] = (0.6, 1.1),
    kd_multiplier_range: Tuple[float, float] = (0.8, 1.5),
    body_com_x_shift_range: Tuple[float, float] = (-0.02, 0.03),
    body_com_y_shift_range: Tuple[float, float] = (-0.005, 0.005),
    body_com_z_shift_range: Tuple[float, float] = (-0.005, 0.005),
    body_mass_scale_range: Tuple[float, float] = (0.9, 1.3),
    body_inertia_scale_range: Tuple[float, float] = (0.9, 1.3),
) -> Dict[str, Any]:
    """Applies domain randomization to the environment.
    
    Args:
        params: Environment parameters.
        state: Environment state.
        rng: Random number generator key.
        friction_range: Range for friction scaling.
        kp_multiplier_range: Range for position control Kp scaling.
        kd_multiplier_range: Range for position control Kd scaling.
        body_com_x_shift_range: Range for body center of mass X shift.
        body_com_y_shift_range: Range for body center of mass Y shift.
        body_com_z_shift_range: Range for body center of mass Z shift.
        body_mass_scale_range: Range for body mass scaling.
        body_inertia_scale_range: Range for body inertia scaling.
        
    Returns:
        Randomized parameters.
    """
    # Create multiple random keys
    keys = jax.random.split(rng, 8)
    
    # Randomize friction
    friction_scale = jax.random.uniform(
        keys[0],
        shape=(),
        minval=friction_range[0],
        maxval=friction_range[1]
    )
    
    # Randomize control gains
    kp_multiplier = jax.random.uniform(
        keys[1],
        shape=(),
        minval=kp_multiplier_range[0],
        maxval=kp_multiplier_range[1]
    )
    kd_multiplier = jax.random.uniform(
        keys[2],
        shape=(),
        minval=kd_multiplier_range[0],
        maxval=kd_multiplier_range[1]
    )
    
    # Randomize body center of mass
    body_com_shift = jp.array([
        jax.random.uniform(
            keys[3],
            shape=(),
            minval=body_com_x_shift_range[0],
            maxval=body_com_x_shift_range[1]
        ),
        jax.random.uniform(
            keys[4],
            shape=(),
            minval=body_com_y_shift_range[0],
            maxval=body_com_y_shift_range[1]
        ),
        jax.random.uniform(
            keys[5],
            shape=(),
            minval=body_com_z_shift_range[0],
            maxval=body_com_z_shift_range[1]
        )
    ])
    
    # Randomize body mass and inertia
    body_mass_scale = jax.random.uniform(
        keys[6],
        shape=(),
        minval=body_mass_scale_range[0],
        maxval=body_mass_scale_range[1]
    )
    body_inertia_scale = jax.random.uniform(
        keys[7],
        shape=(),
        minval=body_inertia_scale_range[0],
        maxval=body_inertia_scale_range[1]
    )
    
    # Update parameters
    params = dict(params)
    params['friction_scale'] = friction_scale
    params['kp_multiplier'] = kp_multiplier
    params['kd_multiplier'] = kd_multiplier
    params['body_com_shift'] = body_com_shift
    params['body_mass_scale'] = body_mass_scale
    params['body_inertia_scale'] = body_inertia_scale
    
    return params
