import numpy as np
from pyquaticus.utils.utils import clip, angle180
from pyquaticus.structs import RenderingPlayer


def heron_move_agents(
    env,
    player: RenderingPlayer,
    desired_speed: float,
    heading_error: float,
    dt: float,
) -> tuple[float, float]:
    # Use vehicle dynamics to go from desired_speed and heading_error
    # to new_speed, new_heading

    # desired heading is relative to current heading
    speed_error = desired_speed - player.speed
    desired_speed = env._pid_controllers[player.id]["speed"](speed_error)
    desired_rudder = env._pid_controllers[player.id]["heading"](heading_error)

    desired_thrust = player.thrust + env.heron_speed_factor * desired_speed

    desired_thrust = clip(desired_thrust, -env.heron_max_thrust, env.heron_max_thrust)
    desired_rudder = clip(desired_rudder, -env.heron_max_rudder, env.heron_max_rudder)

    # propagate vehicle speed
    raw_speed = np.interp(
        desired_thrust, env.heron_thrust_map[0, :], env.heron_thrust_map[1, :]
    )
    new_speed = env._min(
        raw_speed * 1 - ((abs(desired_rudder) / 100) * env.heron_turn_loss),
        env.heron_max_speed,
    )
    if (new_speed - player.speed) / dt > env.heron_max_acc:
        new_speed = player.speed + env.heron_max_acc * dt
    elif (player.speed - new_speed) / dt > env.heron_max_dec:
        new_speed = player.speed - env.heron_max_dec * dt

    # propagate vehicle heading
    raw_d_hdg = desired_rudder * (env.heron_turn_rate / 100) * dt
    thrust_d_hdg = raw_d_hdg * (1 + (abs(desired_thrust) - 50) / 50)
    if desired_thrust < 0:
        thrust_d_hdg = -thrust_d_hdg

    player.thrust = desired_thrust

    # if not moving, then can't turn
    if (new_speed + player.speed) / 2.0 < 0.5:
        thrust_d_hdg = 0.0
    new_heading = angle180(player.heading + thrust_d_hdg)

    return new_speed, new_heading