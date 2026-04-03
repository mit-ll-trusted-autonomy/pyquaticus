# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
Throughput benchmark: measures how many environment steps per second can be
achieved when running N vectorized environments simultaneously.

Usage:
    python throughput_benchmark.py                    # 10,000 envs, 500 steps
    python throughput_benchmark.py --n-envs 1000 --steps 1000
    python throughput_benchmark.py --team-size 2 --timewarp 10
"""

import argparse
import time
import numpy as np
from collections import OrderedDict

import pyquaticus.config
from pyquaticus import pyquaticus_v0


def build_actions(env, n_envs):
    """Pre-sample a batch of random actions for all agents and all envs."""
    actions = OrderedDict()
    for player_id in env.players:
        space = env.action_space(player_id)
        # batch-sample using the space's high/low bounds to avoid n_envs Python calls
        if hasattr(space, 'n'):
            # Discrete
            actions[player_id] = np.random.randint(0, space.n, size=n_envs)
        else:
            # Box (continuous)
            lo, hi = space.low, space.high
            actions[player_id] = np.random.uniform(lo, hi, size=(n_envs,) + lo.shape).astype(space.dtype)
    return actions


def run_benchmark(n_envs: int, n_steps: int, team_size: int, timewarp: int, action_repeat: int):
    config = dict(pyquaticus.config.config_dict_std)
    config["timewarp"] = timewarp
    config["tag_on_oob"] = True
    config["tau"] = 0.05
    config["dynamics"] = "surveyor"
    config["env_bounds"] = [100, 100]

    print(f"\nBuilding environment: {n_envs:,} envs | {team_size}v{team_size} | "
          f"timewarp={timewarp} | action_repeat={action_repeat}")
    t0 = time.perf_counter()
    env = pyquaticus_v0.PyQuaticusEnv(
        render_mode=None,
        n_envs=n_envs,
        team_size=team_size,
        action_repeat=action_repeat,
        config_dict=config,
    )
    build_time = time.perf_counter() - t0
    print(f"  Environment built in {build_time:.2f}s")

    env.reset()
    actions = build_actions(env, n_envs)

    # --- warm-up (a few steps so JIT caches, allocations, etc. settle) ---
    warmup = min(10, n_steps)
    for _ in range(warmup):
        env.step(actions)

    # --- timed run ---
    print(f"  Running {n_steps:,} steps...")
    step_times = []
    t_total_start = time.perf_counter()
    for i in range(n_steps):
        # re-sample actions every step (realistic workload)
        actions = build_actions(env, n_envs)
        t_step = time.perf_counter()
        obs, rews, terms, truncs, infos = env.step(actions)
        step_times.append(time.perf_counter() - t_step)

        # reset any finished envs without breaking the batch
        done_mask = np.zeros(n_envs, dtype=bool)
        for v in terms.values():
            done_mask |= np.asarray(v, dtype=bool)
        for v in truncs.values():
            done_mask |= np.asarray(v, dtype=bool)
        done_envs = np.where(done_mask)[0].tolist()
        if done_envs:
            env.reset(env_idx=done_envs[0] if len(done_envs) == 1 else None)

    t_total = time.perf_counter() - t_total_start

    env.close()

    step_times = np.array(step_times)
    env_steps_per_sec   = n_envs * n_steps / t_total            # env-steps/s
    agent_steps_per_sec = env_steps_per_sec * team_size * 2     # agent-steps/s (both teams)

    print(f"\n{'=' * 60}")
    print(f"  n_envs          : {n_envs:>12,}")
    print(f"  team_size       : {team_size:>12} v {team_size}  ({team_size*2} agents/env)")
    print(f"  steps run       : {n_steps:>12,}")
    print(f"  total wall time : {t_total:>12.3f} s")
    print(f"  --- per-step latency ---")
    print(f"  mean            : {step_times.mean()*1e3:>11.2f} ms")
    print(f"  median          : {np.median(step_times)*1e3:>11.2f} ms")
    print(f"  p95             : {np.percentile(step_times, 95)*1e3:>11.2f} ms")
    print(f"  max             : {step_times.max()*1e3:>11.2f} ms")
    print(f"  --- throughput ---")
    print(f"  env-steps/s     : {env_steps_per_sec:>12,.0f}")
    print(f"  agent-steps/s   : {agent_steps_per_sec:>12,.0f}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="PyQuaticus vectorized environment throughput benchmark")
    parser.add_argument("--n-envs",       type=int, default=10_000, help="Number of parallel environments (default: 10000)")
    parser.add_argument("--steps",        type=int, default=500,    help="Number of steps to benchmark (default: 500)")
    parser.add_argument("--team-size",    type=int, default=1,      help="Agents per team (default: 1)")
    parser.add_argument("--timewarp",     type=int, default=10,     help="Simulation timewarp factor (default: 10)")
    parser.add_argument("--action-repeat",type=int, default=1,      help="Action repeat / frame skip (default: 1)")
    args = parser.parse_args()

    run_benchmark(
        n_envs=args.n_envs,
        n_steps=args.steps,
        team_size=args.team_size,
        timewarp=args.timewarp,
        action_repeat=args.action_repeat,
    )


if __name__ == "__main__":
    main()
