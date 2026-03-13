# SPDX-License-Identifier: BSD-3-Clause
"""
Train MARL policies on Dynamic PyQuaticus with our GNN policy only.

Uses graph observations and the custom GNN model (message passing, self-node embedding).

Usage:
  python rl_test/train_dynamic.py
  python rl_test/train_dynamic.py --render
  # Overnight: progress is logged to out_dir/train.log (disable with --no-log-file)
  python rl_test/train_dynamic.py --speedup 8 --runners 16
  # Save more often so you can resume if you have to stop early (e.g. --save-every 100):
  python rl_test/train_dynamic.py --speedup 8 --runners 16 --save-every 100
  # Resume after a crash (continues from next iteration, saves to same out_dir).
  # You can change Red difficulty when resuming (e.g. --red-heuristic-mode medium); Blue is restored, Red is rebuilt from current args.
  python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_1250
  # Train vs built-in heuristic (default: easy first, then resume with --red-heuristic-mode medium; save every 12):
  python rl_test/train_dynamic.py --red-heuristic
  python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_N --red-heuristic --red-heuristic-mode medium
  # Quick smoke test before a long run:
  python rl_test/train_dynamic.py --iters 100

  # Save checkpoint right now (while training is running): create file SAVE_NOW in out_dir.
  # E.g. from another terminal:  echo. > ray_dynamic/SAVE_NOW   (Windows)
  #                             touch ray_dynamic/SAVE_NOW      (Linux/Mac)
  # Next completed iteration will save to iter_N and delete SAVE_NOW.
"""

import argparse
import logging
import os
import re
import time

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
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
from pyquaticus.config import config_dict_std, ACTION_MAP
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


class DoNothingPolicy(Policy):
    """Deterministic policy that always selects the no-op action for opponents."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # In discrete mode, the last ACTION_MAP entry is defined as the "none" action.
        if hasattr(action_space, "n"):
            self._noop_action = len(ACTION_MAP) - 1
        else:
            # For continuous spaces, the environment treats [0, 0] as no movement.
            self._noop_action = np.array([0.0, 0.0], dtype=np.float32)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        n = len(obs_batch)
        if isinstance(self._noop_action, np.ndarray):
            actions = [self._noop_action.copy() for _ in range(n)]
        else:
            actions = [self._noop_action for _ in range(n)]
        return actions, [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


# Register so checkpoints can load red_policy (RandPolicy) when restoring
if POLICIES is not None:
    POLICIES["RandPolicy"] = RandPolicy


def make_env(config=None, render_mode=None, sim_speedup=4, red_gets_raw_obs=False):
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
    # Graph obs for Blue (GNN); optionally pass raw obs for Red (heuristic policies)
    env = GraphObsWrapper(env, flatten_for_fc=False, red_gets_raw_obs=red_gets_raw_obs)
    env = ParallelPettingZooWrapper(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on Dynamic PyQuaticus")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--iters", type=int, default=2000, help="Training iterations")
    parser.add_argument("--save-every", type=int, default=12, help="Save checkpoint every N iters")
    parser.add_argument("--out-dir", type=str, default="./ray_dynamic/", help="Output directory for checkpoints and train.log")
    parser.add_argument("--runners", type=int, default=8, help="Number of parallel env runners (8=stable default; increase if PC has headroom)")
    parser.add_argument("--speedup", type=int, default=8, help="Sim speedup factor (8=env steps 2x faster, minimal impact on learning)")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="Resume from checkpoint (e.g. ./ray_dynamic/iter_1250)")
    parser.add_argument("--no-log-file", action="store_true", help="Disable writing progress to out_dir/train.log")
    parser.add_argument("--red-heuristic", action="store_true", help="Use built-in heuristic (combined CTF) for Red instead of random")
    parser.add_argument("--red-heuristic-mode", type=str, default="easy", choices=["easy", "medium", "hard"], help="Heuristic difficulty when --red-heuristic (default: easy)")
    parser.add_argument("--red-dummy", action="store_true", help="Use do-nothing policy for Red (always no-op)")
    args = parser.parse_args()

    if args.red_heuristic and args.red_dummy:
        raise SystemExit("Cannot enable both --red-heuristic and --red-dummy at the same time.")

    # Out-dir: use parent of resume path if resuming and out-dir not explicitly set
    if args.resume and args.out_dir == "./ray_dynamic/":
        args.out_dir = os.path.dirname(os.path.normpath(os.path.abspath(args.resume))) or "."
    os.makedirs(args.out_dir, exist_ok=True)

    # Log file: same messages to stdout and to out_dir/train.log (unless --no-log-file)
    log_file = None
    if not args.no_log_file:
        log_path = os.path.join(args.out_dir, "train.log")
        try:
            log_file = open(log_path, "a", encoding="utf-8")
        except OSError:
            log_file = None

    def log(msg):
        print(msg)
        if log_file is not None:
            try:
                log_file.write(msg + "\n")
                log_file.flush()
            except OSError:
                pass

    logging.basicConfig(level=logging.ERROR)
    ray.init(ignore_reinit_error=True)

    RENDER = "human" if args.render else None
    SPEEDUP = max(1, int(args.speedup))

    def env_creator(cfg=None):
        return make_env(cfg, render_mode=RENDER, sim_speedup=SPEEDUP, red_gets_raw_obs=args.red_heuristic)

    register_env("dynamic_pyquaticus", env_creator)
    env = make_env(render_mode=RENDER, sim_speedup=SPEEDUP, red_gets_raw_obs=args.red_heuristic)
    # Reset to ensure agents are initialized
    obs, info = env.reset()
    # Get spaces - Blue uses graph obs, Red uses raw obs when --red-heuristic
    agent_id_blue = "agent_0"
    agent_id_red = "agent_3"
    if hasattr(env, "observation_space") and callable(env.observation_space):
        obs_space_blue = env.observation_space(agent_id_blue)
        obs_space_red = env.observation_space(agent_id_red)
    elif hasattr(env, "observation_spaces") and isinstance(env.observation_spaces, dict):
        obs_space_blue = env.observation_spaces[agent_id_blue]
        obs_space_red = env.observation_spaces.get(agent_id_red, env.observation_spaces[agent_id_blue])
    else:
        par_env = getattr(env, "par_env", env)
        if hasattr(par_env, "observation_space") and callable(par_env.observation_space):
            obs_space_blue = par_env.observation_space(agent_id_blue)
            obs_space_red = par_env.observation_space(agent_id_red)
        else:
            obs_space_blue = par_env.observation_spaces[agent_id_blue]
            obs_space_red = par_env.observation_spaces.get(agent_id_red, obs_space_blue)
    
    if hasattr(env, "action_space") and callable(env.action_space):
        act_space = env.action_space(agent_id_blue)
    elif hasattr(env, "action_spaces") and isinstance(env.action_spaces, dict):
        act_space = env.action_spaces[agent_id_blue]
    else:
        par_env = getattr(env, "par_env", env)
        if hasattr(par_env, "action_space") and callable(par_env.action_space):
            act_space = par_env.action_space(agent_id_blue)
        else:
            act_space = par_env.action_spaces[agent_id_blue]
    # Base env (for heuristic Red) = innermost PyQuaticus env, before env.close()
    base_env = getattr(getattr(env, "par_env", env), "par_env", getattr(env, "par_env", env)) if args.red_heuristic else None
    env.close()

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id in ["agent_0", "agent_1", "agent_2"]:
            return "blue_policy"
        if args.red_heuristic:
            return "red_policy_3" if agent_id == "agent_3" else "red_policy_4" if agent_id == "agent_4" else "red_policy_5"
        if args.red_dummy:
            return "red_dummy_policy"
        return "red_policy"

    if args.red_heuristic:
        from pyquaticus.base_policies.base_policy_wrappers import CombinedGen
        mode = args.red_heuristic_mode
        RedPolicy3 = CombinedGen("agent_3", base_env, mode)
        RedPolicy4 = CombinedGen("agent_4", base_env, mode)
        RedPolicy5 = CombinedGen("agent_5", base_env, mode)
        RedPolicy3.__name__ = "HeuristicRed_3"
        RedPolicy4.__name__ = "HeuristicRed_4"
        RedPolicy5.__name__ = "HeuristicRed_5"
        if POLICIES is not None:
            POLICIES["HeuristicRed_3"] = RedPolicy3
            POLICIES["HeuristicRed_4"] = RedPolicy4
            POLICIES["HeuristicRed_5"] = RedPolicy5
        policies = {
            "blue_policy": (None, obs_space_blue, act_space, {}),
            "red_policy_3": (RedPolicy3, obs_space_red, act_space, {}),
            "red_policy_4": (RedPolicy4, obs_space_red, act_space, {}),
            "red_policy_5": (RedPolicy5, obs_space_red, act_space, {}),
        }
        log(f"Red team using built-in heuristic (combined CTF, {mode} mode).")
    elif args.red_dummy:
        policies = {
            "blue_policy": (None, obs_space_blue, act_space, {}),
            "red_dummy_policy": (DoNothingPolicy, obs_space_blue, act_space, {}),
        }
        log("Red team using do-nothing policy (always no-op actions).")
    else:
        policies = {
            "blue_policy": (None, obs_space_blue, act_space, {}),
            "red_policy": (RandPolicy, obs_space_blue, act_space, {}),
        }

    # Build or load algorithm
    i_start = 0
    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.isdir(resume_path):
            log(f"ERROR: Resume path is not a directory: {resume_path}")
            ray.shutdown()
            raise SystemExit(1)
        # Parse iteration from path (e.g. .../iter_1250 -> 1250)
        basename = os.path.basename(resume_path)
        m = re.match(r"iter_(\d+)$", basename)
        if not m:
            log(f"ERROR: Resume path should end with iter_N (e.g. iter_1250), got: {basename}")
            ray.shutdown()
            raise SystemExit(1)
        i_start = int(m.group(1)) + 1
        log(f"Resuming from {resume_path} (next iteration {i_start})")
        # Load only Blue so Red can use current args (e.g. --red-heuristic-mode). Curriculum easy→medium→hard keeps Blue progress.
        try:
            algo = PPO.from_checkpoint(
                resume_path,
                policy_ids=["blue_policy"],
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["blue_policy"],
            )
        except AttributeError:
            from ray.rllib.algorithms.algorithm import Algorithm
            algo = Algorithm.from_checkpoint(
                resume_path,
                policy_ids=["blue_policy"],
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["blue_policy"],
            )
        # Re-add Red policies from current args (heuristic mode or random).
        for pid, spec in policies.items():
            if pid == "blue_policy":
                continue
            policy_obj, obs_sp, act_sp, cfg = spec
            if isinstance(policy_obj, type):
                algo.add_policy(
                    pid,
                    policy_cls=policy_obj,
                    observation_space=obs_sp,
                    action_space=act_sp,
                    config=cfg,
                )
            else:
                algo.add_policy(
                    pid,
                    policy=policy_obj,
                    observation_space=obs_sp,
                    action_space=act_sp,
                    config=cfg,
                )
        mode_str = getattr(args, "red_heuristic_mode", None) if args.red_heuristic else "random"
        log(f"Red opponent: {mode_str} (current args). Blue weights restored from checkpoint.")
    else:
        # With --render, env has pygame Surface which can't be pickled; use 0 remote workers so rollout runs in driver.
        num_runners = 0 if args.render else args.runners
        if args.render and args.runners != 0:
            log("Rendering enabled: using 0 remote env runners (pygame cannot be pickled for Ray workers).")
        ppo_config = (
            PPOConfig()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(env="dynamic_pyquaticus")
            .env_runners(num_env_runners=num_runners, num_cpus_per_env_runner=0.25)
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["blue_policy"],
            )
        ).training(
            model={
                "custom_model": "gnn_model",
                "custom_model_config": {"gnn_hidden": 64, "gnn_layers": 2},
            },
            # 4000 steps per iteration for more stable learning (longer per iter)
            train_batch_size=4000,
        )
        algo = ppo_config.build_algo()
        log("Training started (new run).")

    try:
        for i in range(i_start, args.iters + 1):
            start = time.time()
            try:
                result = algo.train()
                elapsed = time.time() - start
                ep_rew = result.get("env_runners", {}).get("episode_return_mean", 0)
                if ep_rew is None or (isinstance(ep_rew, float) and not np.isfinite(ep_rew)):
                    ep_rew_str = "n/a (no completed episodes yet)"
                else:
                    ep_rew_str = f"{float(ep_rew):.2f}"
                # Print every 25 iters (and iter 0)
                if i % 25 == 0 or i == 0:
                    log(f"Iter {i}: return_mean={ep_rew_str}, time={elapsed:.1f}s/iter (est. ~{50*elapsed:.0f}s per 50 iters)")
                if i > 0 and i % args.save_every == 0:
                    path = os.path.join(args.out_dir, f"iter_{i}")
                    algo.save(path)
                    log(f"Saved checkpoint to {path}")
                # "Save now" trigger: create out_dir/SAVE_NOW to save at end of current iter
                save_now_file = os.path.join(args.out_dir, "SAVE_NOW")
                if os.path.isfile(save_now_file):
                    path = os.path.join(args.out_dir, f"iter_{i}")
                    algo.save(path)
                    log(f"Saved checkpoint (on request) to {path}")
                    try:
                        os.remove(save_now_file)
                    except OSError:
                        pass
            except Exception as e:
                log(f"ERROR at iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                log("Check Ray worker logs for more details (RolloutWorker pid=... or: ray logs)")
                break
        else:
            log("Training complete.")
    finally:
        ray.shutdown()
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                pass
