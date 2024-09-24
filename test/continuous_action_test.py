import sys
import os
import os.path
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.pyquaticus import Team
from collections import OrderedDict
from pyquaticus.config import config_dict_std, ACTION_MAP

config_dict = config_dict_std
config_dict["max_time"] = 600.0
config_dict["max_score"] = 100
config_dict["render_agent_ids"] = True
config_dict["action_type"] = "continuous"

env = pyquaticus_v0.PyQuaticusEnv(team_size=1, config_dict=config_dict,render_mode='human')
term_g = {0:False,1:False}
truncated_g = {0:False,1:False}
term = term_g
trunc = truncated_g
obs = env.reset()

step = 0
while True:

    blue_action = [1,90] # speed, heading (in degrees)
    red_action = [0.7,45]
    
    obs, reward, term, trunc, info = env.step({0:blue_action,1:red_action})
    k =  list(term.keys())

    step += 1
    if term[k[0]] == True or trunc[k[0]]==True:
        break
env.close()