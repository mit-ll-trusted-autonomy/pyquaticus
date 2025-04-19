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
from pyquaticus.config import ACTION_MAP
import numpy as np

config_dict = {}
config_dict["max_time"] = 600.0
config_dict["max_score"] = 100
config_dict["render_agent_ids"] = True
config_dict["dynamics"] = "si"
continuous = True
config_dict["lidar_obs"] = True
config_dict["sim_speedup_factor"] = 4
config_dict["tau"] = 0.05
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

obs, info = env.reset(return_info=True)

B_one = BaseDefender(
    "agent_1",
    Team.BLUE_TEAM,
    env,
    mode="medium",
    continuous=True
)
B_zero = BaseDefender(
    "agent_0",
    Team.BLUE_TEAM,
    env,
    mode="medium",
    continuous=True
)

R_three = WaypointPolicy(
    "agent_3",
    Team.RED_TEAM,
    env,
    capture_radius=4,
    slip_radius=8,
    avoid_radius=4,
    continuous=True
)
R_two = BaseAttacker(
    "agent_2",
    Team.RED_TEAM,
    env,
    mode="nothing",
    continuous=True
)

R_three.update_state(obs, info)

R_three.plan(
    wp=env.flag_homes[Team.RED_TEAM], num_iters=500
)

while True:

    two = R_two.compute_action(obs, info)
    three = R_three.compute_action(obs, info)
    zero = B_zero.compute_action(obs, info)
    one = B_one.compute_action(obs, info)

    obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three})
    k = list(term.keys())

    if term[k[0]] or trunc[k[0]]:
        break

env.close()
