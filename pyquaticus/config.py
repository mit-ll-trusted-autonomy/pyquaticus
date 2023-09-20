import copy
import numpy as np

from pyquaticus.utils.utils import get_screen_res

MAX_SPEED = 5.0

config_dict_std = {
    "world_size": [110.0, 55.0],  # meters
    "pixel_size": 10,  # pixels/meter
    "scrimmage_line": 55.0,  # horizontal location (meters)
    "agent_radius": 2.0,  # meters
    "catch_radius": 10.0,  # meters
    "flag_keepout": 0.0,  # minimum distance (meters) between agent and flag centers
    "max_speed": MAX_SPEED,  # meters / s
    "own_side_accel": (
        1.0
    ),  # [0, 1] percentage of max acceleration that can be used on your side of scrimmage
    "opp_side_accel": (
        1.0
    ),  # [0, 1] percentage of max acceleration that can be used on opponent's side of scrimmage
    "wall_bounce": (
        0.5
    ),  # [0, 1] percentage of current speed (x or y) at which an agent is repelled from a wall (vertical or horizontal)
    "tau": (
        1 / 10
    ),  # length of timestep (seconds) between state updates and for updating action input from demonstrator or rl
    "max_time": 240.0,  # maximum time (seconds) per episode
    "max_screen_size": get_screen_res(),
    "random_init": (
        False
    ),  # randomly initialize agents' positions for ctf mode (within fairness constraints)
    "save_traj": False,  # save traj as pickle
    "render_fps": 30,
    "normalize": True,  # Flag for normalizing the observation space.
    "tagging_cooldown": (
        10.0
    ),  # Cooldown on an agent after they tag another agent, to prevent consecutive tags
    # MOOS dynamics parameters
    "speed_factor": 20.0,  # Multiplicative factor for desired_speed -> desired_thrust
    "thrust_map": np.array(  # Piecewise linear mapping from desired_thrust to speed
        [[-100, 0, 20, 40, 60, 80, 100], [-2, 0, 1, 2, 3, 5, 5]]
    ),
    "max_thrust": 70,  # Limit on vehicle thrust
    "max_rudder": 100,  # Limit on vehicle rudder actuation
    "turn_loss": 0.85,
    "turn_rate": 70,
    "max_acc": 1,  # m / s**2
    "max_dec": 1,  # m / s**2
    "suppress_numpy_warnings": (
        True  # Option to stop numpy from printing warnings to the console
    ),
    "teleport_on_tag" : False, 
    # Option for the agent when tagged, either out of bounds or by opponent, to teleport home or not
}
""" Standard configuration setup """


def get_std_config() -> dict:
    """Gets a copy of the standard configuration, ideal for minor modifications to the standard configuration."""
    return copy.deepcopy(config_dict_std)


# action space key combos
# maps discrete action id to (speed, heading)
ACTION_MAP = []
for spd in [MAX_SPEED, MAX_SPEED / 2.0]:
    for hdg in range(180, -180, -45):
        ACTION_MAP.append([spd, hdg])
# add a none action
ACTION_MAP.append([0.0, 0.0])
