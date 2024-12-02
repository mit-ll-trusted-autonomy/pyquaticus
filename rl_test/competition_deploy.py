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
from pyquaticus.base_policies.discrete.base_policies import DefendGen, AttackGen
from pyquaticus.base_policies.discrete.base_attack import BaseAttacker
from pyquaticus.base_policies.discrete.base_defend import BaseDefender
from pyquaticus.base_policies.discrete.base_combined import Heuristic_CTF_Agent
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy

RENDER_MODE = 'human'
#RENDER_MODE = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 2v2 PyQuaticus environment')
    parser.add_argument('policy_one', help='Please enter the path to the model you would like to load in')
    parser.add_argument('policy_two', help='Please enter the path to the model you would like to load in') 
    test = True
    reward_config = {0:None, 1:None}
    args = parser.parse_args()

    env = pyquaticus_v0.PyQuaticusEnv(render_mode='human', team_size=2)
    term_g = {0:False,1:False}
    truncated_g = {0:False,1:False}
    term = term_g
    trunc = truncated_g
    obs = env.reset()
    temp_score = env.game_score
    H_one = BaseDefender(2, Team.RED_TEAM, mode='competition_easy')
    H_two = BaseAttacker(3, Team.RED_TEAM, mode='competition_easy')
    policy_one = Policy.from_checkpoint(args.policy_one)
    policy_two = Policy.from_checkpoint(args.policy_two)
    step = 0
    max_step = 2500
    while True:
        new_obs = {}
        #Get Unnormalized Observation for heuristic agents (H_one, and H_two)
        for k in obs:
            new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

        #Get learning agent action from policy
        zero = policy_one.compute_single_action(obs[0])[0]
        one = policy_two.compute_single_action(obs[1])[0]
        #Compute Heuristic agent actions
        two = H_one.compute_action(new_obs)
        three = H_two.compute_action(new_obs)
        #Step the environment
        obs, reward, term, trunc, info = env.step({0:zero,1:one, 2:two, 3:three})
        k =  list(term.keys())
        if step >= max_step:
            break
        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            for k in env.game_score:
                temp_score[k] += env.game_score[k]
            env.reset()
    for k in env.game_score:
        temp_score[k] += env.game_score[k]
    env.close()
    easy_score = temp_score['blue_captures'] - temp_score['red_captures']


