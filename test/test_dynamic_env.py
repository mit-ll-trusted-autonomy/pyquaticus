# SPDX-License-Identifier: BSD-3-Clause
"""Quick test for Dynamic PyQuaticus env."""

import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std
from pyquaticus.envs.dynamic_pyquaticus import DynamicPyQuaticusEnv
from pyquaticus.envs.graph_obs_wrapper import GraphObsWrapper
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper


def test_dynamic_env():
    cfg = config_dict_std.copy()
    cfg["max_score"] = 2
    cfg["max_time"] = 60
    cfg["sim_speedup_factor"] = 4
    reward_config = {f"agent_{i}": rew.caps_and_grabs for i in range(6)}

    env = DynamicPyQuaticusEnv(
        team_size_range=(1, 3),
        tag_removes_agent=False,
        config_dict=cfg,
        reward_config=reward_config,
    )
    obs, info = env.reset(seed=42)
    assert len(obs) == 6
    assert "num_blue_active" in info["agent_0"]
    assert "disabled_agents" in info["agent_0"]
    assert info["agent_0"]["num_blue_active"] >= 1 and info["agent_0"]["num_blue_active"] <= 3

    for _ in range(10):
        actions = {aid: 16 for aid in env.agents}  # no-op
        obs, rewards, term, trunc, info = env.step(actions)
    env.close()
    print("DynamicPyQuaticusEnv: OK")


def test_graph_wrapper():
    cfg = config_dict_std.copy()
    cfg["max_score"] = 2
    cfg["max_time"] = 60
    reward_config = {f"agent_{i}": rew.caps_and_grabs for i in range(6)}

    env = DynamicPyQuaticusEnv(
        team_size_range=(2, 3),
        config_dict=cfg,
        reward_config=reward_config,
    )
    env = ParallelPettingZooWrapper(env)
    env = GraphObsWrapper(env, flatten_for_fc=True)

    obs, info = env.reset(seed=42)
    assert obs["agent_0"].shape == (6 * 10 + 6,)  # flat graph
    for _ in range(5):
        actions = {aid: 16 for aid in env.agents}
        obs, rewards, term, trunc, info = env.step(actions)
    env.close()
    print("GraphObsWrapper: OK")


if __name__ == "__main__":
    test_dynamic_env()
    test_graph_wrapper()
    print("All tests passed.")
