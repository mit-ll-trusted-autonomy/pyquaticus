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

import argparse
import gymnasium as gym
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF1Policy, PPOTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec
import os
import pyquaticus.utils.rewards as rew


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 1v1 policy in a 1v1 PyQuaticus environment')
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    reward_config = {0:rew.sparse, 2:rew.custom_v1, 5:None} # Example Reward Config
    args = parser.parse_args()

    RENDER_MODE = 'human' if args.render else None #set to 'human' if you want rendered output
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=1)
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=1))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    policies = {0:PolicySpec(policy_class=None, observation_space=None, action_space=None), 1:PolicySpec(policy_class=None, observation_space=None, action_space=None),}
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0 or agent_id == 'agent-0':
            return "agent-0-policy"
        else:
            return "agent-1-policy"
    
    env.close()
    policies = {'agent-0-policy':(None, obs_space, act_space, {}), 'agent-1-policy':(None, obs_space, act_space, {})}
    ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=5).resources(num_cpus_per_worker=2, num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy", "agent-1-policy"],)
    algo = ppo_config.build()

    for i in range(10000):
        algo.train()
        if np.mod(i, 100) == 0:
            chkpt_file = algo.save('./ray_test/')
            print(f'Saved to {chkpt_file}', flush=True)
