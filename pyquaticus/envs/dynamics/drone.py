import numpy as np
from pyquaticus.utils.utils import clip, angle180
from pyquaticus.structs import RenderingPlayer


def drone_move_agents(
    env,
    player: RenderingPlayer,
    desired_speed: float,
    heading_error: float,
    dt: float,
) -> tuple[float, float, float]:
    # Use drone dynamics to go from desired_speed and heading_error
    # to new_speed, new_heading, and new_thrust

    pass
