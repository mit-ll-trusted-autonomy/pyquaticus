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
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec
import os

from ray.rllib.algorithms.ppo import PPO


RENDER_MODE = 'human'
#RENDER_MODE = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 1v1 PyQuaticus environment')
    parser.add_argument('checkpoint', help='Please enter the path to the model you would like to load in') 
    test = True
    reward_config = {0:None, 1:None}
    args = parser.parse_args()
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=1)
    env = ParallelPettingZooEnv(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=1))
    register_env('pyquaticus', lambda config: ParallelPettingZooEnv(env_creator(config)))
    
    
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    policies = {0:PolicySpec(policy_class=None, observation_space=obs_space, action_space=act_space), 1:PolicySpec(policy_class=None, observation_space=obs_space, action_space=act_space),}
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0 or agent_id == 'agent-0':
            return "agent-0-policy"
        else:
            return "agent-1-policy"

    env.close()
    policies = {'agent-0-policy':(None, obs_space, act_space, {}), 'agent-1-policy':(None, obs_space, act_space, {})}
    ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=1).resources(num_cpus_per_worker=4, num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy", "agent-1-policy"],)
    ppo_config.evaluation(evaluation_interval=1)
    algo = ppo_config.build()
    print("args: ", args)
    algo.restore('./ray_test/'+str(args.checkpoint)+'/')
    results = algo.evaluate()




