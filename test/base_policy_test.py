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
from pyquaticus.config import ACTION_MAP

config_dict = {}
config_dict["max_time"] = 600.0
config_dict["max_score"] = 100
config_dict["sim_speedup_factor"] = 8

env = pyquaticus_v0.PyQuaticusEnv(team_size=2, config_dict=config_dict,render_mode='human')
term_g = {0:False,1:False}
truncated_g = {0:False,1:False}
term = term_g
trunc = truncated_g

obs, info = env.reset()

temp_captures = env.state["captures"]
temp_grabs = env.state["grabs"]
temp_tags = env.state["tags"]

r_one = BaseAttacker('agent_2', env, mode='competition_easy')
r_two = BaseDefender('agent_3', env, mode='competition_easy')

b_one = BaseDefender('agent_0', env, mode='competition_easy')
b_two = BaseAttacker('agent_1', env, mode='competition_easy')
step = 0
while True:

    two = r_one.compute_action(obs, info)
    three = r_two.compute_action(obs, info)
    zero = b_one.compute_action(obs, info)
    one = b_two.compute_action(obs, info)

    
    obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three})
    k =  list(term.keys())

    if term[k[0]] == True or trunc[k[0]]==True:
        break

env.close()
