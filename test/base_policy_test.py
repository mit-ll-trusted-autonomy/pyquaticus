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
config_dict["sim_speedup_factor"] = 5

env = pyquaticus_v0.PyQuaticusEnv(team_size=2, config_dict=config_dict,render_mode='human')


obs, info = env.reset()

r_one = BaseAttacker(2, Team.RED_TEAM, env, mode='easy')
r_two = BaseAttacker(3, Team.RED_TEAM, env, mode='nothing')

b_one = Heuristic_CTF_Agent(0, Team.BLUE_TEAM, env, mode='hard')
b_two = BaseDefender(1, Team.BLUE_TEAM, env, mode='nothing')

while True:

    two = r_one.compute_action(obs, info)
    three = r_two.compute_action(obs, info)
    zero = b_one.compute_action(obs, info)
    one = b_two.compute_action(obs, info)

    
    obs, reward, term, trunc, info = env.step({0:zero,1:one, 2:two, 3:three})
    k =  list(term.keys())

    if term[k[0]] == True or trunc[k[0]]==True:
        break

env.close()
