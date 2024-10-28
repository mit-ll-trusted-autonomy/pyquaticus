import numpy as np
from pyquaticus.utils.utils import clip, angle180
from pyquaticus.structs import RenderingPlayer


def fixed_wing_move_agents(
    env,
    player: RenderingPlayer,
    desired_speed: float,
    heading_error: float,
    dt: float,
) -> tuple[float, float]:

    new_speed = clip(desired_speed, env.fixed_wing_min_speed, env.fixed_wing_max_speed)

    desired_turn_rate = np.deg2rad(heading_error / dt)

    desired_turn_radius = new_speed / desired_turn_rate

    new_turn_radius = max(desired_turn_radius, env.fixed_wing_min_turn_radius)

    new_turn_rate = np.rad2deg(new_speed / new_turn_radius)

    new_heading = player.heading + np.sign(heading_error) * new_turn_rate * dt

    return new_speed, new_heading
