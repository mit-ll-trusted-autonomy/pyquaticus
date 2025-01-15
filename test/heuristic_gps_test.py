import sys
import os
import os.path
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.waypoint_policy import WaypointPolicy
from pyquaticus.envs.pyquaticus import Team
from collections import OrderedDict
from pyquaticus.config import config_dict_std, ACTION_MAP

config = config_dict_std
config["gps_env"] = True
config["default_init"] = False
config["env_bounds"] = "auto"
config["env_bounds"] = [-93.5417, 49.2916, -93.4848, 49.3151]
config["env_bounds_unit"] = "ll"
config["blue_flag_home"] = (49.3016, -93.5217)
config["red_flag_home"] = (49.3051, -93.5048)
config["flag_homes_unit"] = "ll"
config["sim_speedup_factor"] = 80
config["max_time"] = 2400
config["action_type"] = "continuous"
config["dynamics"] = "heron"

init_dict = {
    "agent_pos_unit": "ll",
    "agent_position": {
        0: (49.3010, -93.5210),
        1: (49.3016, -93.5210),
        2: (49.3022, -93.5210),
        3: (49.3045, -93.5055),
        4: (49.3051, -93.5055),
        5: (49.3057, -93.5055),
    },
    "agent_heading": {
        0: 90,
        1: 90,
        2: 90,
        3: -90,
        4: -90,
        5: -90,
    }
}


env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config, render_mode="human")
term = {0: False, 1: False}
trunc = {0: False, 1: False}
obs, info = env.reset(options={"init_dict": init_dict})

red_one = BaseDefender(
    3,
    Team.RED_TEAM,
    env,
    mode="easy",
)
red_two = BaseDefender(
    4,
    Team.RED_TEAM,
    env,
    mode="easy",
)
red_three = BaseDefender(
    5,
    Team.RED_TEAM,
    env,
    mode="easy",
)

blue_one = Heuristic_CTF_Agent(
    0,
    Team.BLUE_TEAM,
    env,
    mode="nothing",
)
blue_two = WaypointPolicy(
    1,
    Team.BLUE_TEAM,
    env,
    capture_radius=5,
    slip_radius=10
)
blue_three = Heuristic_CTF_Agent(
    2,
    Team.BLUE_TEAM,
    env,
    mode="nothing",
)

blue_two.update_state(obs, info)
blue_two.plan(wp=env.flag_homes[Team.RED_TEAM], num_iters=500)

while not (any(term.values()) or any(trunc.values())):

    three = red_one.compute_action(obs, info)
    four = red_two.compute_action(obs, info)
    five = red_three.compute_action(obs, info)
    zero = blue_one.compute_action(obs, info)
    one = blue_two.compute_action(obs, info)
    two = blue_three.compute_action(obs, info)

    obs, reward, term, trunc, info = env.step(
        {0: zero, 1: one, 2: two, 3: three, 4: four, 5: five}
    )

env.close()
