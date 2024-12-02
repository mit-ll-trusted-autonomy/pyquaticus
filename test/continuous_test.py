import sys
import os
import os.path
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.wp_follower_continuous import WaypointFollowerContinuous
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
config_dict["sim_speedup_factor"] = 1
config_dict["tau"] = 0.05
# config_dict["catch_radius"] = 1

env = pyquaticus_v0.PyQuaticusEnv(team_size=2, config_dict=config_dict,render_mode='human')
term_g = {0:False,1:False}
truncated_g = {0:False,1:False}
term = term_g
trunc = truncated_g
obs, info = env.reset()
temp_captures = env.state["captures"]
temp_grabs = env.state["grabs"]
temp_tags = env.state["tags"]

H_one = BaseAttacker(2, Team.RED_TEAM, mode="easy")
H_two = BaseAttacker(3, Team.RED_TEAM, mode="easy")

R_one = BaseDefender(0, Team.BLUE_TEAM, mode="easy")
R_two = BaseDefender(1, Team.BLUE_TEAM, mode="easy")
step = 0
while True:
    new_obs = {}
    for k in obs:
        new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

    global_state = env.global_state_normalizer.unnormalized(info["global_state"])

    # print(global_state)

    two = H_one.compute_action(global_state)
    three = H_two.compute_action(global_state)
    zero = R_one.compute_action(global_state)
    one = R_two.compute_action(global_state)

    
    obs, reward, term, trunc, info = env.step({0:zero,1:one, 2:two, 3:three})
    k =  list(term.keys())

    step += 1
    if term[k[0]] == True or trunc[k[0]]==True:
        break
for i in range(len(env.state["captures"])):
    temp_captures[i] += env.state["captures"][i]
for i in range(len(env.state["grabs"])):
    temp_grabs[i] += env.state["grabs"][i]
for i in range(len(env.state["tags"])):
    temp_tags[i] += env.state["tags"][i]

env.close()