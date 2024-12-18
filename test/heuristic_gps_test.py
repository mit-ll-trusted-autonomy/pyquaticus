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

config = config_dict_std
config["gps_env"] = True
config["default_init"] = False
config["env_bounds"] = "auto"
config["blue_flag_home"] = (42.352229714597705, -70.99992567997114)
config["red_flag_home"] = (42.32710627259394, -70.96739585043458)
config["flag_homes_unit"] = "ll"
config["sim_speedup_factor"] = 5
config["default_dynamics"] = False
config["dynamics_dict"] = {
    0: "heron",
    1: "large_usv",
    2: "si",
    3: "drone",
    4: "drone",
    5: "drone",
}


env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config, render_mode="human")
term_g = {'agent_0': False, 'agent_1': False}
truncated_g = {'agent_0': False, 'agent_1': False}
term = term_g
trunc = truncated_g
obs,_ = env.reset()
temp_captures = env.state["captures"]
temp_grabs = env.state["grabs"]
temp_tags = env.state["tags"]

H_one = Heuristic_CTF_Agent('agent_3', Team.RED_TEAM, mode="hard")
H_two = Heuristic_CTF_Agent('agent_4', Team.RED_TEAM, mode="hard")
H_three = Heuristic_CTF_Agent('agent_5', Team.RED_TEAM, mode="hard")

R_one = Heuristic_CTF_Agent('agent_0', Team.BLUE_TEAM, mode="hard")
R_two = Heuristic_CTF_Agent('agent_1', Team.BLUE_TEAM, mode="hard")
R_three = Heuristic_CTF_Agent('agent_2', Team.BLUE_TEAM, mode="hard")
step = 0
while True:
    new_obs = {}
    for k in obs:
        new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

    three = H_one.compute_action(new_obs)
    four = H_two.compute_action(new_obs)
    five = H_three.compute_action(new_obs)
    zero = R_one.compute_action(new_obs)
    one = R_two.compute_action(new_obs)
    two = R_three.compute_action(new_obs)

    obs, reward, term, trunc, info = env.step(
        {'agent_0': zero, 'agent_1': one, 'agent_2': two, 'agent_3': three, 'agent_4': four, 'agent_5': five}
    )
    k = list(term.keys())

    step += 1
    if term[k[0]] == True or trunc[k[0]] == True:
        break

env.close()
