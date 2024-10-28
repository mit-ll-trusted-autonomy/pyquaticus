import numpy as np
from pyquaticus.utils.utils import clip, angle180
from pyquaticus.structs import RenderingPlayer


def si_move_agents(
    env,
    player: RenderingPlayer,
    desired_speed: float,
    heading_error: float,
    dt: float,
) -> tuple[float, float]:
    # Use single-integrator dynamics to go from desired_speed and heading_error
    # to new_speed, new_heading

    new_speed = clip(desired_speed, -env.si_max_speed, env.si_max_speed)
    turn_rate = clip(heading_error / dt, -env.si_max_turn_rate, env.si_max_turn_rate)
    new_heading = player.heading + turn_rate * dt
    new_thrust = desired_speed

    return new_speed, new_heading
