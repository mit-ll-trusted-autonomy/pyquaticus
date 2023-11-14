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

env = pyquaticus_v0.PyQuaticusEnv(team_size=2, config_dict=config_dict,render_mode='human')
term_g = {0:False,1:False}
truncated_g = {0:False,1:False}
term = term_g
trunc = truncated_g
obs = env.reset()
temp_score = env.game_score

r_one = BaseAttacker(2, Team.RED_TEAM, mode='competition_easy')
r_two = BaseDefender(3, Team.RED_TEAM, mode='competition_easy')

b_one = BaseDefender(0, Team.BLUE_TEAM, mode='competition_easy')
b_two = BaseAttacker(1, Team.BLUE_TEAM, mode='competition_easy')
step = 0
while True:
    new_obs = {}
    for k in obs:
        new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

    two = r_one.compute_action(new_obs)
    three = r_two.compute_action(new_obs)
    zero = r_one.compute_action(new_obs)
    one = r_two.compute_action(new_obs)

    
    obs, reward, term, trunc, info = env.step({0:zero,1:one, 2:two, 3:three})
    k =  list(term.keys())

    step += 1
    if term[k[0]] == True or trunc[k[0]]==True:
        break
for k in env.game_score:
    temp_score[k] += env.game_score[k]
env.close()
