import sys
import os
import os.path
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.waypoint_policy import WaypointPolicy
from pyquaticus.envs.pyquaticus import Team
from collections import OrderedDict
from pyquaticus.config import get_std_config, ACTION_MAP

config = get_std_config()
config["gps_env"] = True
config["default_init"] = False
config["env_bounds"] = "auto"
config["env_bounds"] = [-93.5417, 49.2916, -93.4848, 49.3151]
config["env_bounds_unit"] = "ll"
config["blue_flag_home"] = (49.3016, -93.5217)
config["red_flag_home"] = (49.3051, -93.5048)
config["flag_homes_unit"] = "ll"
config["sim_speedup_factor"] = 80
config["max_time"] = 2400
config["dynamics"] = "large_usv"
config["catch_radius"] = 200

init_dict = {
    "agent_pos_unit": "ll",
    "agent_position": {
        0: (49.3010, -93.5210),
        1: (49.3016, -93.5210),
        2: (49.3022, -93.5210),
        3: (49.3045, -93.5055),
        4: (49.3051, -93.5055),
        5: (49.3057, -93.5055),
    },
    "agent_heading": {
        0: 90,
        1: 90,
        2: 90,
        3: -90,
        4: -90,
        5: -90,
    }
}


env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config, render_mode="human")
term_g = {'agent_0': False, 'agent_1': False}
truncated_g = {'agent_0': False, 'agent_1': False}
term = term_g
trunc = truncated_g
obs, info = env.reset()

temp_captures = env.state["captures"]
temp_grabs = env.state["grabs"]
temp_tags = env.state["tags"]


H_one = Heuristic_CTF_Agent('agent_3', env, mode="hard", continuous=True)
H_two = Heuristic_CTF_Agent('agent_4', env, mode="hard", continuous=True)
H_three = Heuristic_CTF_Agent('agent_5', env, mode="hard", continuous=True)

R_one = Heuristic_CTF_Agent('agent_0', env, mode="hard", continuous=True)
R_two = Heuristic_CTF_Agent('agent_1', env, mode="hard", continuous=True)
R_three = Heuristic_CTF_Agent('agent_2', env, mode="hard", continuous=True)

step = 0
while True:

    three = H_one.compute_action(obs, info)
    four = H_two.compute_action(obs, info)
    five = H_three.compute_action(obs, info)
    zero = R_one.compute_action(obs, info)
    one = R_two.compute_action(obs, info)
    two = R_three.compute_action(obs, info)

    obs, reward, term, trunc, info = env.step(
        {'agent_0': zero, 'agent_1': one, 'agent_2': two, 'agent_3': three, 'agent_4': four, 'agent_5': five}
    )
    k =  list(term.keys())

    if term[k[0]] == True or trunc[k[0]]==True:
        break

env.close()
