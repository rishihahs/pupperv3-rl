"""Utilities for adding obstacles to the model."""

import xml.etree.ElementTree as ET
import numpy as np
from typing import Tuple

def add_boxes_to_model(
    tree: ET.ElementTree,
    n_boxes: int = 0,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    y_range: Tuple[float, float] = (-3.0, 3.0),
    height: float = 0.04,
    length: float = 2.0
) -> ET.ElementTree:
    """Adds obstacle boxes to the model.
    
    Args:
        tree: XML element tree of the model.
        n_boxes: Number of boxes to add.
        x_range: Range of x coordinates.
        y_range: Range of y coordinates.
        height: Height of boxes.
        length: Length of boxes.
        
    Returns:
        Modified XML element tree.
    """
    if n_boxes <= 0:
        return tree
    
    root = tree.getroot()
    worldbody = root.find("worldbody")
    
    # Create boxes with reproducible random positions
    np.random.seed(0)
    
    x_coords = np.random.uniform(x_range[0], x_range[1], n_boxes)
    y_coords = np.random.uniform(y_range[0], y_range[1], n_boxes)
    
    for i in range(n_boxes):
        box = ET.SubElement(
            worldbody,
            "geom",
            name=f"obstacle_{i}",
            type="box",
            size=f"{length/2} 0.1 {height/2}",
            pos=f"{x_coords[i]} {y_coords[i]} {height/2}",
            quat="1 0 0 0",
            material="grid",
            conaffinity="1",
            contype="1",
            group="1"
        )
    
    return tree
