"""Utilities for adding heightfield to the model."""

import xml.etree.ElementTree as ET
import numpy as np
import jax
import jax.numpy as jp
from typing import Optional

def create_random_heightfield(
    grid_size: int = 256,
    radius_x: float = 10.0,
    radius_y: float = 10.0,
    elevation_z: float = 0.02,
    seed: int = 0
) -> np.ndarray:
    """Creates a random heightfield.
    
    Args:
        grid_size: Size of the grid.
        radius_x: X radius of the heightfield.
        radius_y: Y radius of the heightfield.
        elevation_z: Maximum elevation.
        seed: Random seed.
        
    Returns:
        Heightfield array.
    """
    # High-resolution noise
    noise = np.array(jax.random.uniform(
        jax.random.PRNGKey(seed),
        (grid_size, grid_size)
    ))
    
    # Low-resolution noise for areas
    area_noise = jax.random.uniform(
        jax.random.PRNGKey(seed + 1),
        (int(grid_size // radius_x), int(grid_size // radius_y))
    )
    
    # Upscale low-resolution noise
    upscaled_area_noise = np.array(jax.image.resize(
        image=area_noise,
        shape=(grid_size, grid_size),
        method="nearest"
    ))
    
    # Combine noises
    scaled_noise = noise * upscaled_area_noise
    
    return scaled_noise

def create_step_heightfield(
    grid_size: int = 256,
    step_size: int = 4,
    seed: int = 0
) -> np.ndarray:
    """Creates a heightfield with steps.
    
    Args:
        grid_size: Size of the grid.
        step_size: Size of steps.
        seed: Random seed.
        
    Returns:
        Heightfield array.
    """
    # Low-resolution steps
    steps = jax.random.uniform(
        jax.random.PRNGKey(seed),
        (grid_size // step_size, grid_size // step_size)
    )
    
    # Upscale to full resolution
    scaled_noise = np.array(jax.image.resize(
        image=steps,
        shape=(grid_size, grid_size),
        method="nearest"
    ))
    
    return scaled_noise

def add_heightfield_to_model(
    tree: ET.ElementTree,
    heightfield: np.ndarray,
    grid_size: int = 256,
    radius_x: float = 10.0,
    radius_y: float = 10.0,
    elevation_z: float = 0.02,
    base_z: float = 0.2,
    group: str = "0"
) -> ET.ElementTree:
    """Adds heightfield to the model.
    
    Args:
        tree: XML element tree of the model.
        heightfield: Heightfield array.
        grid_size: Size of the grid.
        radius_x: X radius of the heightfield.
        radius_y: Y radius of the heightfield.
        elevation_z: Maximum elevation.
        base_z: Base height.
        group: Collision group.
        
    Returns:
        Modified XML element tree.
    """
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset = root.find("asset")
    
    # Add heightfield asset
    ET.SubElement(
        asset,
        "hfield",
        name="hfield_geom",
        nrow=f"{grid_size}",
        ncol=f"{grid_size}",
        elevation=' '.join(heightfield.astype(str).flatten().tolist()),
        size=f"{radius_x} {radius_y} {elevation_z} {base_z}",
    )
    
    # Add heightfield geom
    ET.SubElement(
        worldbody,
        "geom",
        name="hfield_floor",
        type="hfield",
        hfield="hfield_geom",
        material="grid",
        conaffinity="1",
        contype="1",
        condim="3",
        group=group,
    )
    
    return tree
