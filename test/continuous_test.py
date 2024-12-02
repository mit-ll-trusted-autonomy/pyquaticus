import sys
import os
import os.path
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.discrete.base_attack import BaseAttacker
from pyquaticus.base_policies.discrete.base_defend import BaseDefender
from pyquaticus.base_policies.discrete.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.continuous.base_attack_continuous import BaseAttackerContinuous
from pyquaticus.base_policies.continuous.base_defend_continuous import BaseDefenderContinuous
from pyquaticus.base_policies.continuous.base_combined_continuous import Heuristic_CTF_Agent_Continuous
from pyquaticus.base_policies.continuous.wp_follower_continuous import WaypointFollowerContinuous
from pyquaticus.envs.pyquaticus import Team
from collections import OrderedDict
from pyquaticus.config import config_dict_std, ACTION_MAP
import numpy as np

config_dict = config_dict_std
config_dict["max_time"] = 600.0
config_dict["max_score"] = 100
config_dict["render_agent_ids"] = True
config_dict["dynamics"] = "heron"
config_dict["action_type"] = "continuous"
config_dict["sim_speedup_factor"] = 3
config_dict["tau"] = 0.05
# config_dict["catch_radius"] = 1

env = pyquaticus_v0.PyQuaticusEnv(team_size=2, config_dict=config_dict,render_mode='human')
term_g = {0:False,1:False}
truncated_g = {0:False,1:False}
term = term_g
trunc = truncated_g
obs = env.reset()
temp_captures = env.state["captures"]
temp_grabs = env.state["grabs"]
temp_tags = env.state["tags"]

H_one = WaypointFollowerContinuous(2, Team.RED_TEAM, wps=[np.array([50, 0]), np.array([75, 25]), np.array([50, 50])])
H_two = BaseDefenderContinuous(3, Team.RED_TEAM, mode="nothing")

R_one = WaypointFollowerContinuous(0, Team.BLUE_TEAM, wps=[np.array([100, 0]), np.array([125, 25]), np.array([100, 50])])
R_two = Heuristic_CTF_Agent_Continuous(1, Team.BLUE_TEAM, mode="nothing")
step = 0
while True:
    new_obs = {}
    for k in obs:
        new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

    two = H_one.compute_action(new_obs)
    three = H_two.compute_action(new_obs)
    zero = R_one.compute_action(new_obs)
    one = R_two.compute_action(new_obs)

    
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