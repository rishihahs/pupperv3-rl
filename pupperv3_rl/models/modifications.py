"""Utilities for modifying the robot model."""

import xml.etree.ElementTree as ET
from typing import Optional, Tuple

def set_mjx_custom_options(
    tree: ET.ElementTree,
    max_contact_points: int = 20,
    max_geom_pairs: int = 20
) -> ET.ElementTree:
    """Sets custom MJX options in the model.
    
    Args:
        tree: XML element tree of the model.
        max_contact_points: Maximum number of contact points.
        max_geom_pairs: Maximum number of geometry pairs.
        
    Returns:
        Modified XML element tree.
    """
    root = tree.getroot()
    
    # Find or create compiler element
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    
    # Set custom options
    compiler.set("mjxstrncpy", "1")  # Required for MJX
    
    # Find or create option element
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    
    # Set custom options
    option.set("cone", "elliptic")
    option.set("iterations", "1")
    option.set("ls_iterations", "5")
    option.set("impratio", "10")
    option.set("tolerance", "1e-6")
    option.set("noslip_tolerance", "1e-6")
    option.set("maxcontact", str(max_contact_points))
    option.set("maxgeompair", str(max_geom_pairs))
    
    return tree
