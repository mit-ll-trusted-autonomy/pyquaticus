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
import numpy as np

config_dict = config_dict_std
config_dict["max_time"] = 600.0
config_dict["max_score"] = 100
config_dict["render_agent_ids"] = True
config_dict["dynamics"] = "si"
config_dict["action_type"] = "continuous"
continuous = True
config_dict["lidar_obs"] = True
config_dict["sim_speedup_factor"] = 4
config_dict["tau"] = 0.05
config_dict["normalize"] = False
# config_dict["catch_radius"] = 1
config_dict["obstacles"] = {
    "polygon": [
        (
            (100, 55),
            (130, 35),
            (125, 25),
            (105, 35),
            (100, 25),
            (105, 15),
            (100, 5),
            (90, 25),
        )
    ]
}

env = pyquaticus_v0.PyQuaticusEnv(
    team_size=2, config_dict=config_dict, render_mode="human"
)

obs, info = env.reset()

H_one = BaseAttacker(
    2,
    Team.RED_TEAM,
    env,
    mode="easy",
)
H_two = BaseDefender(
    3,
    Team.RED_TEAM,
    env,
    mode="easy",
)

R_one = WaypointPolicy(
    0,
    Team.BLUE_TEAM,
    env,
    capture_radius=5,
    slip_radius=10,
    avoid_radius=4,
)
R_two = BaseAttacker(
    1,
    Team.BLUE_TEAM,
    env,
    mode="nothing",
)

R_one.update_state(obs, info)

R_one.plan(
    wp=env.flag_homes[Team.RED_TEAM], num_iters=500
)

while True:

    two = H_one.compute_action(obs, info)
    three = H_two.compute_action(obs, info)
    zero = R_one.compute_action(obs, info)
    one = R_two.compute_action(obs, info)

    obs, reward, term, trunc, info = env.step({0: zero, 1: one, 2: two, 3: three})
    k = list(term.keys())

    if term[k[0]] or trunc[k[0]]:
        break

env.close()
