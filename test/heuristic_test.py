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
config_dict["render_agent_ids"] = True
config_dict["dynamics"] = ["si", "si", "si", "si"]
config_dict["sim_speedup_factor"] = 3

env = pyquaticus_v0.PyQuaticusEnv(team_size=2, config_dict=config_dict,render_mode='human')
term_g = {'agent_0':False,'agent_1':False}
truncated_g = {'agent_0':False,'agent_1':False}
term = term_g
trunc = truncated_g

reset_opts = {'normalize_obs': False, 'normalize_state': False}

obs, info = env.reset(options=reset_opts)

temp_captures = env.state["captures"]
temp_grabs = env.state["grabs"]
temp_tags = env.state["tags"]

H_one = Heuristic_CTF_Agent('agent_2', env, mode="hard", continuous=True)
H_two = Heuristic_CTF_Agent('agent_3', env, mode="hard", continuous=True)

R_one = Heuristic_CTF_Agent('agent_0', env, mode="hard", continuous=True)
R_two = Heuristic_CTF_Agent('agent_1', env, mode="hard", continuous=True)

step = 0
while True:
    two = H_one.compute_action(obs, info)
    three = H_two.compute_action(obs, info)
    zero = R_one.compute_action(obs, info)
    one = R_two.compute_action(obs, info)

    
    obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':three})
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

for i in range(len(env.state["captures"])):
    temp_captures[i] += env.state["captures"][i]
for i in range(len(env.state["grabs"])):
    temp_grabs[i] += env.state["grabs"][i]
for i in range(len(env.state["tags"])):
    temp_tags[i] += env.state["tags"][i]

env.close()