# (C) 2021 Massachusetts Institute of Technology.

# Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

# The software/firmware is provided to you on an As-Is basis

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
import ray
from ray.rllib.algorithms.ppo import PPOConfig
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
from ray.rllib.policy.policy import PolicySpec, Policy
import os
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
from pyquaticus.config import config_dict_std
import logging
class RandPolicy(Policy):
    """
    Example wrapper for training against a random policy.

    To use a base policy, insantiate it inside a wrapper like this,
    and call it from self.compute_actions

    See policies and policy_mapping_fn for how policies are associated
    with agents
    """
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [-1 for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 2v2 policy in a 2v2 PyQuaticus environment')
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    reward_config = {'blue_0':rew.sparse, 'blue_1':rew.sparse, 'red_2':None, 'red_3':None} # Example Reward Config
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    RENDER_MODE = 'human' if args.render else None #set to 'human' if you want rendered output
    
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 1
    config_dict['max_score'] = 3
    config_dict['max_time']=240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob']=True
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode=RENDER_MODE, reward_config=reward_config, team_size=2)
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode=RENDER_MODE, reward_config=reward_config, team_size=2))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    obs_space = env.observation_space['blue_0']
    act_space = env.action_space['blue_0']
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 'blue_0':
            return "agent-0-policy"
        if agent_id == 'blue_1':
            return "agent-1-policy"
        return "random"
        #elif agent_id == 2 or agent_id == 'agent-2':
            # change this to agent-1-policy to train both agents at once
        #    return "easy-defend-policy"
        #else:
        #    return "easy-attack-policy"
    
    policies = {'agent-0-policy':(None, obs_space, act_space, {}), 
                'agent-1-policy':(None, obs_space, act_space, {}),
                'random':(RandPolicy, obs_space, act_space, {})}
                #Examples of Heuristic Opponents in Rllib Training (See two lines below)
                #'easy-defend-policy': (DefendGen(2, Team.RED_TEAM, 'easy', 2, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                #'easy-attack-policy': (AttackGen(3, Team.RED_TEAM, 'easy', 2, env.par_env.agent_obs_normalizer), obs_space, act_space, {})}
    env.close()
    #Not using the Alpha Rllib (api_stack False) 
    ppo_config = PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False).environment(env='pyquaticus').env_runners(num_env_runners=14, num_cpus_per_env_runner=1.0)
    #If your system allows changing the number of rollouts can significantly reduce training times (num_rollout_workers=15)
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy", "agent-1-policy"],)
    algo = ppo_config.build()
    start = 0
    end = 0
    for i in range(8001):
        print("Looping: ", i)
        start = time.time()
        algo.train()
        end = time.time()
        print("End Loop: ", end-start)
        if np.mod(i, 500) == 0:
            print("Saving Checkpoint: ", i)
            chkpt_file = algo.save('./ray_test/iter_'+str(i)+'/')
