import sys
import os
import os.path
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.wp_follower import WaypointFollower
from pyquaticus.envs.pyquaticus import Team
from collections import OrderedDict
from pyquaticus.config import config_dict_std, ACTION_MAP
import numpy as np

config_dict = config_dict_std
config_dict["max_time"] = 600.0
config_dict["max_score"] = 100
config_dict["render_agent_ids"] = True
config_dict["dynamics"] = "heron"
# config_dict["action_type"] = "continuous"
config_dict["lidar_obs"] = True
config_dict["sim_speedup_factor"] = 4
config_dict["tau"] = 0.05
config_dict["normalize"] = False
# config_dict["catch_radius"] = 1

env = pyquaticus_v0.PyQuaticusEnv(team_size=2, config_dict=config_dict, render_mode='human')

obs, info = env.reset()

H_one = BaseAttacker(2, Team.RED_TEAM, 3, [0, 1], env.agent_obs_normalizer, env.global_state_normalizer, mode="easy", continuous=False)
H_two = BaseDefender(3, Team.RED_TEAM, 2, [0, 1], env.agent_obs_normalizer, env.global_state_normalizer, mode="medium", continuous=False)

R_one = Heuristic_CTF_Agent(0, Team.BLUE_TEAM, 1, [2, 3], env.agent_obs_normalizer, env.global_state_normalizer, mode="easy", continuous=False)
R_two = BaseAttacker(1, Team.BLUE_TEAM, 0, [2, 3], env.agent_obs_normalizer, env.global_state_normalizer, mode="nothing", continuous=False)

while True:
    # print(global_state)

    two = H_one.compute_action(obs, info)
    three = H_two.compute_action(obs, info)
    zero = R_one.compute_action(obs, info)
    one = R_two.compute_action(obs, info)

    obs, reward, term, trunc, info = env.step({0: zero, 1: one, 2: two, 3: three})
    k = list(term.keys())

    if term[k[0]] or trunc[k[0]]:
        break

env.close()
