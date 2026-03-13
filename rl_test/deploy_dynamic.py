# SPDX-License-Identifier: BSD-3-Clause
"""
Run the trained blue policy from a checkpoint against random or heuristic red agents.

Usage:
  python rl_test/deploy_dynamic.py
  python rl_test/deploy_dynamic.py ./ray_dynamic/iter_500
  python rl_test/deploy_dynamic.py ./ray_dynamic/iter_500 --no-render
  python rl_test/deploy_dynamic.py ./ray_dynamic/iter_132 --red-heuristic --red-heuristic-mode easy
  python rl_test/deploy_dynamic.py ./ray_dynamic/iter_360 --red-dummy
"""

import argparse
import os

import numpy as np
from gymnasium.spaces import Discrete
from ray.rllib.policy.policy import Policy
from ray.rllib.models.catalog import ModelCatalog

from pyquaticus.models.gnn_model import GNNModel
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std
from pyquaticus.envs.dynamic_pyquaticus import DynamicPyQuaticusEnv
from pyquaticus.envs.graph_obs_wrapper import GraphObsWrapper
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper

BLUE_AGENTS = ["agent_0", "agent_1", "agent_2"]
RED_AGENTS = ["agent_3", "agent_4", "agent_5"]

# Register GNN model so Policy.from_checkpoint can reconstruct it
ModelCatalog.register_custom_model("gnn_model", GNNModel)

# Fallback action space (discrete 17) if env does not expose it
_DEFAULT_ACTION_SPACE = Discrete(17)


def _get_action_space(env, agent_id):
    """Get action space for an agent; works with ParallelPettingZooWrapper and GraphObsWrapper."""
    if hasattr(env, "action_space") and callable(env.action_space):
        try:
            return env.action_space(agent_id)
        except Exception:
            pass
    spaces = getattr(env, "action_spaces", None)
    if spaces and isinstance(spaces, dict) and agent_id in spaces:
        return spaces[agent_id]
    par = getattr(env, "par_env", None)
    if par is not None:
        return _get_action_space(par, agent_id)
    return _DEFAULT_ACTION_SPACE


def make_env(render_mode="human", red_gets_raw_obs=False, red_dummy=False):
    cfg = config_dict_std.copy()
    cfg["sim_speedup_factor"] = 4
    cfg["max_score"] = 3
    cfg["max_time"] = 240
    cfg["tagging_cooldown"] = 60
    cfg["tag_on_oob"] = True
    if red_dummy:
        cfg["red_dummy_mode"] = True
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
    env = GraphObsWrapper(env, flatten_for_fc=False, red_gets_raw_obs=red_gets_raw_obs)
    env = ParallelPettingZooWrapper(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Deploy trained blue policy from checkpoint")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="./ray_dynamic/iter_500",
        help="Checkpoint dir (e.g. ./ray_dynamic/iter_500) or path to blue_policy (e.g. ./ray_dynamic/iter_500/policies/blue_policy)",
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering (faster)")
    parser.add_argument("--max-episodes", type=int, default=0, help="Stop after N episodes (0 = run until Ctrl+C)")
    parser.add_argument("--red-heuristic", action="store_true", help="Use easy/medium/hard heuristic for Red instead of random")
    parser.add_argument("--red-heuristic-mode", type=str, default="easy", choices=["easy", "medium", "hard"], help="Heuristic difficulty (default: easy)")
    parser.add_argument("--red-dummy", action="store_true", help="No Red opponents (Blue plays alone; same as training with --red-dummy)")
    args = parser.parse_args()

    if args.red_heuristic and args.red_dummy:
        print("Error: Cannot use both --red-heuristic and --red-dummy.")
        return

    # Resolve policy path
    path = os.path.abspath(args.checkpoint)
    if os.path.isdir(path) and not path.endswith("blue_policy"):
        policy_path = os.path.join(path, "policies", "blue_policy")
    else:
        policy_path = path
    if not os.path.isdir(policy_path):
        print(f"Error: Policy not found at {policy_path}")
        print("Usage: python rl_test/deploy_dynamic.py [./ray_dynamic/iter_500] [--red-heuristic]")
        return

    print(f"Loading blue policy from: {policy_path}")
    blue_policy = Policy.from_checkpoint(policy_path)

    render_mode = None if args.no_render else "human"
    env = make_env(render_mode=render_mode, red_gets_raw_obs=args.red_heuristic, red_dummy=args.red_dummy)

    # Red heuristic: need base env (DynamicPyQuaticusEnv) and raw obs for Red
    red_heuristics = None
    if args.red_dummy:
        print("Red team: dummy (no opponents; Blue plays alone).")
    elif args.red_heuristic:
        base_env = getattr(getattr(env, "par_env", env), "par_env", getattr(env, "par_env", env))
        red_heuristics = {
            "agent_3": Heuristic_CTF_Agent("agent_3", base_env, mode=args.red_heuristic_mode),
            "agent_4": Heuristic_CTF_Agent("agent_4", base_env, mode=args.red_heuristic_mode),
            "agent_5": Heuristic_CTF_Agent("agent_5", base_env, mode=args.red_heuristic_mode),
        }
        print(f"Red team: heuristic (combined CTF, {args.red_heuristic_mode} mode).")

    obs, info = env.reset()

    episode = 0
    step = 0
    max_steps_per_episode = 5000
    ep_blue_reward = 0.0
    ep_red_reward = 0.0

    try:
        while True:
            # Blue: use trained policy
            actions = {}
            for aid in obs:
                if aid in BLUE_AGENTS:
                    action = blue_policy.compute_single_action(obs[aid], explore=False)
                    if isinstance(action, (list, tuple)):
                        action = action[0]
                    if hasattr(action, "item"):
                        action = int(action.item())
                    actions[aid] = action
                else:
                    # Red: heuristic or random
                    if red_heuristics is not None and aid in red_heuristics:
                        # Heuristic expects info[agent_id]["global_state"]; wrapper may alter info shape, so get from base env
                        global_state = base_env._history_to_state()
                        heuristic_info = {aid: {"global_state": global_state}}
                        action = red_heuristics[aid].compute_action(obs[aid], heuristic_info)
                        actions[aid] = int(action) if hasattr(action, "item") else int(action)
                    else:
                        space = _get_action_space(env, aid)
                        samp = space.sample()
                        actions[aid] = int(samp) if hasattr(samp, "item") else int(samp)

            obs, rewards, term, trunc, info = env.step(actions)
            step += 1
            for a in BLUE_AGENTS:
                if a in rewards:
                    ep_blue_reward += rewards[a]
            for a in RED_AGENTS:
                if a in rewards:
                    ep_red_reward += rewards[a]

            done = any(term.values()) or any(trunc.values()) or step >= max_steps_per_episode
            if done:
                episode += 1
                print(f"Episode {episode} finished (steps={step})  Blue total: {ep_blue_reward:.1f}  Red total: {ep_red_reward:.1f}")
                if args.max_episodes and episode >= args.max_episodes:
                    break
                obs, info = env.reset()
                step = 0
                ep_blue_reward = 0.0
                ep_red_reward = 0.0
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
    print("Done.")


if __name__ == "__main__":
    main()
