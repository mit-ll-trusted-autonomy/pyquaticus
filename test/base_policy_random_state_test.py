from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.base_policies.deprecated.base import BaseAgentPolicy as OldBaseAgentPolicy
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
import numpy as np
import time

config_dict = {}
config_dict["normalize_obs"] = False
config_dict["normalize_state"] = True
config_dict["default_init"] = False
config_dict["on_sides_init"] = False
config_dict["render_agent_ids"] = True

env = pyquaticus_v0.PyQuaticusEnv(
    team_size=2, config_dict=config_dict, render_mode="human", action_space="continuous"
)

num_checks = 10000

modes = ["easy", "medium", "hard", "competition_easy", "competition_medium"]
ids = ["agent_0", "agent_1", "agent_2", "agent_3"]
new_policies: list[BaseAgentPolicy] = []
old_policies: list[OldBaseAgentPolicy] = []

afp = env.aquaticus_field_points
fk = env.flag_keepout_radius
cr = env.catch_radius

for mode in modes:
    for id in ids:
        if id in ids[:2]:
            team = Team.BLUE_TEAM
        else:
            team = Team.RED_TEAM

        ms = env.max_speeds[ids.index(id)]
        
        new_policies.append(BaseAttacker(id, env, continuous=True, mode=mode))
        new_policies.append(BaseAttacker(id, env, continuous=False, mode=mode))
        new_policies.append(BaseDefender(id, env, fk, cr, continuous=True, mode=mode))
        new_policies.append(BaseDefender(id, env, fk, cr, continuous=False, mode=mode))
        old_policies.append(OldBaseAttacker(id, team, ms, afp, continuous=True, mode=mode))
        old_policies.append(OldBaseAttacker(id, team, ms, afp, continuous=False, mode=mode))
        old_policies.append(OldBaseDefender(id, team, ms, afp, continuous=True, mode=mode, flag_keepout=fk, catch_radius=cr))
        old_policies.append(OldBaseDefender(id, team, ms, afp, continuous=False, mode=mode, flag_keepout=fk, catch_radius=cr))

        if mode in modes[:3]:
            new_policies.append(Heuristic_CTF_Agent(id, env, fk, cr, continuous=True, mode=mode))
            new_policies.append(Heuristic_CTF_Agent(id, env, fk, cr, continuous=False, mode=mode))
            old_policies.append(OldHeuristicAgent(id, team, ms, afp, continuous=True, mode=mode, flag_keepout=fk, catch_radius=cr))
            old_policies.append(OldHeuristicAgent(id, team, ms, afp, continuous=False, mode=mode, flag_keepout=fk, catch_radius=cr))

np.random.seed(3589)
for i in range(num_checks):
    obs, info = env.reset()
    for j in range(len(new_policies)):
        new = new_policies[j].compute_action(obs, info)
        old = old_policies[j].compute_action(obs)
        if isinstance(new, str) or isinstance(old, str):
            if new != old:
                print(new)
                print(old)
                raise Exception("policies don't match")
        elif not np.all(np.isclose(np.array(new), np.array(old), rtol=0.001, atol=0.001)):
            print(f"{new}")
            print(f"{old}")
            print(type(new_policies[j]))
            print(new_policies[j].mode)
            print(type(old_policies[j]))
            print(old_policies[j].mode)
            print(new_policies[j].id)
            time.sleep(20)
            raise Exception(f"policies don't match")