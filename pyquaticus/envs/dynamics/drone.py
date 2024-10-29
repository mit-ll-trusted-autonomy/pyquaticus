import numpy as np
from pyquaticus.utils.utils import clip, angle180
from pyquaticus.structs import RenderingPlayer


def drone_move_agents(
    env,
    player: RenderingPlayer,
    desired_speed: float,
    heading_error: float,
    dt: float,
) -> tuple[float, float]:
    """
    Use quadcopter dynamics to move the agent given a desired speed and heading error.
    Adapted from https://github.com/AtsushiSakai/PythonRobotics?tab=readme-ov-file#drone-3d-trajectory-following

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

    desired_speed = clip(desired_speed, 0, env.drone_max_speed)

    # Constants
    g = 9.81
    m = 0.2
    Ixx = 1
    Iyy = 1
    Izz = 1

    # PID control coefficients
    Kp_x = 1
    Kp_y = 1
    Kp_z = 1
    Kp_roll = 100
    Kp_pitch = 100
    Kp_yaw = 25

    Kd_x = 1
    Kd_y = 1
    Kd_z = 1
    Kd_roll = 5
    Kd_pitch = 5
    Kd_yaw = 5

    # Convert heading error (deg) to desired yaw
    player.yaw = np.deg2rad(player.heading)
    yaw_error = np.deg2rad(heading_error)
    des_yaw = player.yaw + yaw_error

    # Calculate desired acceleration in x and y directions
    des_x_vel = desired_speed * np.sin(des_yaw)
    cur_x_vel = player.x_vel
    des_x_acc = clip((des_x_vel - cur_x_vel) / dt, -10, 10)
    des_y_vel = desired_speed * np.cos(des_yaw)
    cur_y_vel = player.y_vel
    des_y_acc = clip((des_y_vel - cur_y_vel) / dt, -10, 10)

    # Placeholders for z for now so that it is easier to add in the future
    des_z_pos = 0
    des_z_vel = 0
    des_z_acc = 0
    z_pos = 0
    z_vel = 0

    # Calculate vertical thrust and roll, pitch, and yaw torques.
    thrust = m * (
        g + des_z_acc + Kp_z * (des_z_pos - z_pos) + Kd_z * (des_z_vel - z_vel)
    )

    roll_torque = (
        Kp_roll
        * (
            ((des_x_acc * np.cos(des_yaw) + des_y_acc * np.sin(des_yaw)) / g)
            - player.roll
        )
        - Kd_roll * player.roll_rate
    )

    pitch_torque = (
        Kp_pitch
        * (
            ((des_x_acc * np.sin(des_yaw) - des_y_acc * np.cos(des_yaw)) / g)
            - player.pitch
        )
        - Kd_pitch * player.pitch_rate
    )

    yaw_torque = Kp_yaw * (des_yaw - player.yaw) - Kd_yaw * player.yaw_rate

    # Get roll, pitch, and yaw rates from torques and moments of inertia
    player.roll_rate += roll_torque * dt / Ixx
    player.pitch_rate += pitch_torque * dt / Iyy
    player.yaw_rate += yaw_torque * dt / Izz

    # Propagate roll, pitch, and yaw (and heading for proper rendering)
    player.roll += player.roll_rate * dt
    player.pitch += player.pitch_rate * dt
    player.yaw += player.yaw_rate * dt
    player.yaw = np.arctan2(np.sin(player.yaw), np.cos(player.yaw))
    player.heading = np.rad2deg(player.yaw)

    # Transform into world frame to get x, y, and z accelerations, velocities, and positions
    R = rotation_matrix(player.roll, player.pitch, player.yaw)
    acc = (np.matmul(R, np.array([0, 0, thrust]).T) - np.array([0, 0, m * g]).T) / m
    x_acc = acc[0]
    y_acc = acc[1]
    z_acc = acc[2]
    player.x_vel = cur_x_vel + x_acc * dt
    player.y_vel = cur_y_vel + y_acc * dt
    z_vel += z_acc * dt
    z_pos += z_vel * dt
    x_pos = player.pos[0] + cur_x_vel * dt
    y_pos = player.pos[1] + cur_y_vel * dt

    player.prev_pos = player.pos
    player.pos = np.asarray([x_pos, y_pos])
    player.speed = np.sqrt(np.power(cur_x_vel, 2) + np.power(cur_y_vel, 2))

    return player.speed, player.heading


# Adapted from https://github.com/AtsushiSakai/PythonRobotics?tab=readme-ov-file#drone-3d-trajectory-following
def rotation_matrix(roll, pitch, yaw):
    """
    Calculates the rotation matrix.

    Args
        Roll: Angular position about the x-axis in radians.
        Pitch: Angular position about the y-axis in radians.
        Yaw: Angular position about the z-axis in radians.

    Returns
        3x3 rotation matrix as NumPy array
    """
    return np.array(
        [
            [
                np.cos(yaw) * np.cos(roll),
                -np.sin(yaw) * np.cos(pitch)
                + np.cos(yaw) * np.sin(roll) * np.sin(pitch),
                np.sin(yaw) * np.sin(pitch)
                + np.cos(yaw) * np.sin(roll) * np.cos(pitch),
            ],
            [
                np.sin(yaw) * np.cos(roll),
                np.cos(yaw) * np.cos(pitch)
                + np.sin(yaw) * np.sin(roll) * np.sin(pitch),
                -np.cos(yaw) * np.sin(pitch)
                + np.sin(yaw) * np.sin(roll) * np.cos(pitch),
            ],
            [-np.sin(roll), np.cos(roll) * np.sin(pitch), np.cos(roll) * np.cos(yaw)],
        ]
    )
