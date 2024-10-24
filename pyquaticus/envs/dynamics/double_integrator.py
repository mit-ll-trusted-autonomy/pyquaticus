import numpy as np
from pyquaticus.utils.utils import clip, angle180
from pyquaticus.structs import RenderingPlayer


def di_move_agents(
    env,
    player: RenderingPlayer,
    desired_speed: float,
    heading_error: float,
    dt: float,
) -> tuple[float, float, float]:
    # Use double-integrator dynamics to go from desired_speed and heading_error
    # to new_speed, new_heading, and new_thrust

    desired_acc = (desired_speed - player.speed) / dt
    desired_omega = heading_error / dt
    desired_alpha = (desired_omega - player.turn_rate) / dt
    desired_acc = clip(desired_acc, -env.di_max_acc, env.di_max_acc)
    desired_alpha = clip(desired_alpha, -env.di_max_alpha, env.di_max_alpha)

    new_speed = player.speed + desired_acc * dt
    new_speed = clip(new_speed, -env.di_max_speed, env.di_max_speed)

    new_omega = player.turn_rate + desired_alpha * dt
    new_omega = clip(new_omega, -env.di_max_omega, env.di_max_omega)
    player.turn_rate = new_omega
    new_heading = player.heading + new_omega * dt

    return new_speed, new_heading, new_speed
