import numpy as np
from pyquaticus.utils.utils import clip, angle180, mag_heading_to_vec
from pyquaticus.structs import RenderingPlayer


def di_move_agents(
    env,
    player: RenderingPlayer,
    desired_speed: float,
    heading_error: float,
    dt: float,
) -> tuple[float, float]:
    """
    Use double-integrator unicycle dynamics to move the agent given a desired speed and heading error.

    Args:
        env: the PyQuaticusEnv
        player: the player to move
        desired speed: desired speed, in m/s
        heading_error: heading error, in deg
        dt: the length of time to simulate

    Returns:
        new_speed: current speed, in m/s
        new_heading: current heading, in degrees east of north
    """

    # Get and clip desired linear and angular acceleration
    desired_acc = (desired_speed - player.speed) / dt

    desired_turn_rate = heading_error / dt
    desired_alpha = (desired_turn_rate - player.turn_rate) / dt

    desired_acc = clip(desired_acc, -env.di_max_acc, env.di_max_acc)
    desired_alpha = clip(desired_alpha, -env.di_max_alpha, env.di_max_alpha)

    # Calculate new linear speed and turn rate
    new_speed = player.speed + desired_acc * dt
    new_speed = clip(new_speed, -env.di_max_speed, env.di_max_speed)

    new_turn_rate = player.turn_rate + desired_alpha * dt
    new_turn_rate = clip(new_turn_rate, -env.di_max_turn_rate, env.di_max_turn_rate)

    player.turn_rate = new_turn_rate
    new_heading = player.heading + new_turn_rate * dt

    # Propagate vehicle position based on new speed and heading
    hdg_rad = np.deg2rad(player.heading)
    new_hdg_rad = np.deg2rad(new_heading)
    avg_speed = (new_speed + player.speed) / 2.0
    if env.gps_env:
        avg_speed = avg_speed / env.meters_per_mercator_xy
    s = np.sin(new_hdg_rad) + np.sin(hdg_rad)
    c = np.cos(new_hdg_rad) + np.cos(hdg_rad)
    avg_hdg = np.arctan2(s, c)
    # Note: sine/cos swapped because of the heading / angle difference
    new_ag_pos = [
        player.pos[0] + np.sin(avg_hdg) * avg_speed * dt,
        player.pos[1] + np.cos(avg_hdg) * avg_speed * dt,
    ]

    player.prev_pos = player.pos
    player.pos = np.asarray(new_ag_pos)
    player.speed = clip(new_speed, 0.0, env.max_speed)
    player.heading = angle180(new_heading)

    return new_speed, new_heading
