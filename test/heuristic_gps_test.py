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
from pyquaticus.config_aas import config_dict_aas

config = config_dict_std
config["gps_env"] = True
config["default_init"] = False
config["env_bounds"] = "auto"
config["blue_flag_home"] = (49.301673369138015, -93.52174888478079)
config["red_flag_home"] = (49.30515588662647, -93.50487665880466)
config["flag_homes_unit"] = "ll"
config["sim_speedup_factor"] = 5
config["default_dynamics"] = False
config["dynamics_dict"] = {
    0: "heron",
    1: "large_usv",
    2: "si",
    3: "fixed_wing",
    4: "fixed_wing",
    5: "fixed_wing",
}


env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config, render_mode="human")
term_g = {0: False, 1: False}
truncated_g = {0: False, 1: False}
term = term_g
trunc = truncated_g
obs = env.reset()
temp_captures = env.state["captures"]
temp_grabs = env.state["grabs"]
temp_tags = env.state["tags"]

H_one = Heuristic_CTF_Agent(3, Team.RED_TEAM, mode="hard")
H_two = Heuristic_CTF_Agent(4, Team.RED_TEAM, mode="hard")
H_three = Heuristic_CTF_Agent(5, Team.RED_TEAM, mode="hard")

R_one = Heuristic_CTF_Agent(0, Team.BLUE_TEAM, mode="hard")
R_two = Heuristic_CTF_Agent(1, Team.BLUE_TEAM, mode="hard")
R_three = Heuristic_CTF_Agent(2, Team.BLUE_TEAM, mode="hard")
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
        {0: zero, 1: one, 2: two, 3: three, 4: four, 5: five}
    )
    k = list(term.keys())

    step += 1
    if term[k[0]] == True or trunc[k[0]] == True:
        break

env.close()
