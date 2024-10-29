import numpy as np
from pyquaticus.utils.utils import clip, angle180
from pyquaticus.structs import RenderingPlayer


def large_usv_move_agents(
    env,
    player: RenderingPlayer,
    desired_speed: float,
    heading_error: float,
    dt: float,
) -> tuple[float, float]:
    """
    Use altered MOOS-IVP dynamics to move the agent given a desired speed and heading error.
    Adapted from https://oceanai.mit.edu/ivpman/pmwiki/pmwiki.php?n=IvPTools.USimMarineV22

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

    # desired heading is relative to current heading
    speed_error = desired_speed - player.speed
    desired_speed = env._pid_controllers[player.id]["speed"](speed_error)
    desired_rudder = env._pid_controllers[player.id]["heading"](heading_error)

    desired_thrust = player.thrust + env.large_usv_speed_factor * desired_speed

    desired_thrust = clip(
        desired_thrust, -env.large_usv_max_thrust, env.large_usv_max_thrust
    )
    desired_rudder = clip(
        desired_rudder, -env.large_usv_max_rudder, env.large_usv_max_rudder
    )

    # propagate vehicle speed
    raw_speed = np.interp(
        desired_thrust, env.large_usv_thrust_map[0, :], env.large_usv_thrust_map[1, :]
    )
    new_speed = env._min(
        raw_speed * 1 - ((abs(desired_rudder) / 100) * env.large_usv_turn_loss),
        env.large_usv_max_speed,
    )
    if (new_speed - player.speed) / dt > env.large_usv_max_acc:
        new_speed = player.speed + env.large_usv_max_acc * dt
    elif (player.speed - new_speed) / dt > env.large_usv_max_dec:
        new_speed = player.speed - env.large_usv_max_dec * dt

    # propagate vehicle heading
    raw_d_hdg = desired_rudder * (env.large_usv_turn_rate / 100) * dt
    thrust_d_hdg = raw_d_hdg * (1 + (abs(desired_thrust) - 50) / 50)
    if desired_thrust < 0:
        thrust_d_hdg = -thrust_d_hdg

    # if not moving, then can't turn
    if (new_speed + player.speed) / 2.0 < 0.5:
        thrust_d_hdg = 0.0
    new_heading = angle180(player.heading + thrust_d_hdg)

    player.thrust = desired_thrust

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
