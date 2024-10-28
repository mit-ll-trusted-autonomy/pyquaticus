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

    # Propagate vehicle position based on new_heading and new_speed
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
