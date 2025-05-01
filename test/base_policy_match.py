from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.deprecated.base_attack import (
    BaseAttacker as OldBaseAttacker,
)
from pyquaticus.base_policies.deprecated.base_defend import (
    BaseDefender as OldBaseDefender,
)
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.deprecated.base_combined import (
    Heuristic_CTF_Agent as OldHeuristicAgent,
)
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.base_policies.key_agent import KeyAgent
import numpy as np

config_dict = {}
config_dict["max_time"] = 600.0
config_dict["max_score"] = 100
config_dict["sim_speedup_factor"] = 5
config_dict["normalize_obs"] = False
config_dict["normalize_state"] = True
config_dict["render_agent_ids"] = True

env = pyquaticus_v0.PyQuaticusEnv(
    team_size=2, config_dict=config_dict, render_mode="human", action_space="continuous"
)


term_g = {0: False, 1: False}
truncated_g = {0: False, 1: False}
term = term_g
trunc = truncated_g

obs, info = env.reset()

r_one_old = OldBaseAttacker(
    "agent_2",
    Team.RED_TEAM,
    2,
    mode="hard",
    continuous=True,
    aquaticus_field_points=env.aquaticus_field_points,
)
r_one_new = BaseAttacker(
    "agent_2", Team.RED_TEAM, env, mode="hard", continuous=True
)

r_two_old = OldBaseDefender(
    "agent_3",
    Team.RED_TEAM,
    2,
    mode="medium",
    continuous=True,
    flag_keepout=env.flag_keepout_radius,
    aquaticus_field_points=env.aquaticus_field_points,
)
r_two = BaseDefender(
    "agent_3", Team.RED_TEAM, env, mode="medium", continuous=True
)

b_one_old = OldHeuristicAgent(
    "agent_0",
    Team.BLUE_TEAM,
    2,
    aquaticus_field_points=env.aquaticus_field_points,
    mode="hard",
    continuous=True,
    flag_keepout=env.flag_keepout_radius,
)
b_one = Heuristic_CTF_Agent(
    "agent_0", Team.BLUE_TEAM, env, mode="hard", continuous=True
)

b_two = KeyAgent()
step = 0
while True:
    two = r_one_new.compute_action(obs, info)
    two_old = r_one_old.compute_action(obs)
    if isinstance(two, str) or isinstance(two_old, str):
        if two != two_old:
            print("---")
            print(two)
            print(two_old)
            raise Exception("Attack policies don't match")
    elif not np.all(np.isclose(np.array(two), np.array(two_old), rtol=0.001, atol=0.001)):
        print("---")
        print(f"{two}")
        print(f"{two_old}")
        # raise Exception("Attack policies don't match")

    three = r_two.compute_action(obs, info)
    three_old = r_two_old.compute_action(obs)
    if isinstance(three, str) or isinstance(three_old, str):
        if three != three_old:
            print("---")
            print(three)
            print(three_old)
            raise Exception("Defend policies don't match")
    elif not np.all(np.isclose(np.array(three), np.array(three_old), rtol=0.001, atol=0.001)):
        print("---")
        print(f"{three}")
        print(f"{three_old}")
        raise Exception("Defend policies don't match")

    zero = b_one.compute_action(obs, info)
    zero_old = b_one_old.compute_action(obs)
    if isinstance(zero, str) or isinstance(zero_old, str):
        if zero != zero_old:
            print("---")
            print(zero)
            print(zero_old)
            raise Exception("Heuristic policies don't match")
    elif not np.all(np.isclose(np.array(zero), np.array(zero_old), rtol=0.001, atol=0.001)):
        print("---")
        print(f"{zero}")
        print(f"{zero_old}")
        raise Exception("Heuristic policies don't match")
    one = b_two.compute_action(obs, info)

    obs, reward, term, trunc, info = env.step(
        {"agent_0": zero, "agent_1": one, "agent_2": two, "agent_3": three}
    )
    k = list(term.keys())

    if term[k[0]] == True or trunc[k[0]] == True:
        break

env.close()
