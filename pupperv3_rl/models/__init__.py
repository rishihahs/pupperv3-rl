"""Model modification module for Pupper V3 RL."""

from pupperv3_rl.models.modifications import set_mjx_custom_options
from pupperv3_rl.models.obstacles import add_boxes_to_model
from pupperv3_rl.models.heightfield import (
    create_random_heightfield,
    create_step_heightfield,
    add_heightfield_to_model
)
