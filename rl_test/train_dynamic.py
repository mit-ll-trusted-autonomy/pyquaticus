# SPDX-License-Identifier: BSD-3-Clause
"""
Train MARL policies on Dynamic PyQuaticus with our GNN policy only.

Uses graph observations and the custom GNN model (message passing, self-node embedding).
Usage:
  python rl_test/train_dynamic.py
  python rl_test/train_dynamic.py --render
"""

import argparse
import logging
import os
import time

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

try:
    from ray.rllib.algorithms.registry import POLICIES
except ImportError:
    try:
        from ray.rllib.policy.registry import POLICIES
    except ImportError:
        POLICIES = None
try:
    from ray.rllib.models.catalog import ModelCatalog
    from pyquaticus.models.gnn_model import GNNModel
    ModelCatalog.register_custom_model("gnn_model", GNNModel)
except Exception as e:
    raise RuntimeError("GNN model registration failed (need ray/rllib and pyquaticus.models.gnn_model).") from e

import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std
from pyquaticus.envs.dynamic_pyquaticus import DynamicPyQuaticusEnv
from pyquaticus.envs.graph_obs_wrapper import GraphObsWrapper
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper


class RandPolicy(Policy):
    """Random policy for opponent agents."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        n = len(obs_batch)
        if hasattr(self.action_space, "n"):
            return [np.random.randint(0, self.action_space.n) for _ in range(n)], [], {}
        return [self.action_space.sample() for _ in range(n)], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


# Register so checkpoints can load red_policy (RandPolicy) when restoring
if POLICIES is not None:
    POLICIES["RandPolicy"] = RandPolicy


def make_env(config=None, render_mode=None, sim_speedup=4):
    cfg = config_dict_std.copy()
    cfg["sim_speedup_factor"] = sim_speedup
    cfg["max_score"] = 3
    cfg["max_time"] = 240
    cfg["tagging_cooldown"] = 60
    cfg["tag_on_oob"] = True

    reward_config = {
        "agent_0": rew.caps_and_grabs, "agent_1": rew.caps_and_grabs, "agent_2": rew.caps_and_grabs,
        "agent_3": rew.caps_and_grabs, "agent_4": rew.caps_and_grabs, "agent_5": rew.caps_and_grabs,
    }

    env = DynamicPyQuaticusEnv(
        team_size_range=(1, 3),
        tag_removes_agent=False,
        reinforcement_interval=0,
        config_dict=cfg,
        reward_config=reward_config,
        render_mode=render_mode,
    )
    # Graph obs for GNN policy (no flatten)
    env = GraphObsWrapper(env, flatten_for_fc=False)
    env = ParallelPettingZooWrapper(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on Dynamic PyQuaticus")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--iters", type=int, default=2000, help="Training iterations")
    parser.add_argument("--save-every", type=int, default=250, help="Save checkpoint every N iters")
    parser.add_argument("--out-dir", type=str, default="./ray_dynamic/", help="Output directory")
    parser.add_argument("--runners", type=int, default=20, help="Number of parallel env runners (more = faster if you have CPUs)")
    parser.add_argument("--speedup", type=int, default=4, help="Sim speedup factor (4=default; 8=env steps 2x faster, minimal impact on learning)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    ray.init(ignore_reinit_error=True)

    RENDER = "human" if args.render else None
    SPEEDUP = max(1, int(args.speedup))

    def env_creator(cfg=None):
        return make_env(cfg, render_mode=RENDER, sim_speedup=SPEEDUP)

    register_env("dynamic_pyquaticus", env_creator)
    env = make_env(render_mode=RENDER, sim_speedup=SPEEDUP)
    # Reset to ensure agents are initialized
    obs, info = env.reset()
    # Get spaces - RLLib wrapper may expose as method or dict
    agent_id = list(obs.keys())[0]
    if hasattr(env, "observation_space") and callable(env.observation_space):
        obs_space = env.observation_space(agent_id)
    elif hasattr(env, "observation_spaces") and isinstance(env.observation_spaces, dict):
        obs_space = env.observation_spaces[agent_id]
    else:
        # Fallback: get from par_env
        par_env = getattr(env, "par_env", env)
        if hasattr(par_env, "observation_space") and callable(par_env.observation_space):
            obs_space = par_env.observation_space(agent_id)
        else:
            obs_space = par_env.observation_spaces[agent_id]
    
    if hasattr(env, "action_space") and callable(env.action_space):
        act_space = env.action_space(agent_id)
    elif hasattr(env, "action_spaces") and isinstance(env.action_spaces, dict):
        act_space = env.action_spaces[agent_id]
    else:
        par_env = getattr(env, "par_env", env)
        if hasattr(par_env, "action_space") and callable(par_env.action_space):
            act_space = par_env.action_space(agent_id)
        else:
            act_space = par_env.action_spaces[agent_id]
    env.close()

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id in ["agent_0", "agent_1", "agent_2"]:
            return "blue_policy"
        return "red_policy"

    policies = {
        "blue_policy": (None, obs_space, act_space, {}),
        "red_policy": (RandPolicy, obs_space, act_space, {}),
    }

    ppo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="dynamic_pyquaticus")
        .env_runners(num_env_runners=args.runners, num_cpus_per_env_runner=0.25)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["blue_policy"],
        )
    ).training(
        model={
            "custom_model": "gnn_model",
            "custom_model_config": {"gnn_hidden": 64, "gnn_layers": 2},
        }
    )
    algo = ppo_config.build_algo()

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.iters + 1):
        start = time.time()
        try:
            result = algo.train()
            elapsed = time.time() - start
            ep_rew = result.get("env_runners", {}).get("episode_return_mean", 0)
            if i % 50 == 0:
                print(f"Iter {i}: return_mean={ep_rew:.2f}, time={elapsed:.1f}s/iter (est. ~{50*elapsed:.0f}s per 50 iters)")
            if i > 0 and i % args.save_every == 0:
                path = os.path.join(args.out_dir, f"iter_{i}")
                algo.save(path)
                print(f"Saved checkpoint to {path}")
        except Exception as e:
            print(f"ERROR at iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            print("\nCheck Ray worker logs for more details:")
            print("  - Look for 'RolloutWorker pid=...' error messages above")
            print("  - Or check: ray logs")
            break

    ray.shutdown()
    print("Training complete.")
