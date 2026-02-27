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
Evaluate a trained hierarchical PPO policy for 3v3 PyQuaticus.

This script loads a checkpoint and evaluates the trained policy against
baseline/random opponents over multiple episodes.
"""

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
from pyquaticus.base_policies.base_policy_wrappers import DefendGen, AttackGen
from pyquaticus.config import config_dict_std
import logging
from pyquaticus.hierarchical.role_wrapper import wrap_env_with_roles
from pyquaticus.hierarchical.roles import ATTACK, DEFEND, INTERCEPT, ROLE_NAMES


class RandPolicy(Policy):
    """Random policy for baseline opponents."""
    
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
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


def create_role_reward_config():
    """Create reward configuration that uses hierarchical role rewards."""
    def role_reward_wrapper(agent_id, team, agents, agent_inds_of_team, state, 
                           prev_state, env_size, agent_radius, catch_radius, 
                           scrimmage_coords, max_speeds, tagging_cooldown):
        role_id = ATTACK  # Default
        return rew.hierarchical_role_reward(
            agent_id, team, agents, agent_inds_of_team, state, prev_state,
            env_size, agent_radius, catch_radius, scrimmage_coords, 
            max_speeds, tagging_cooldown, role_id
        )
    return role_reward_wrapper


def env_creator_with_roles(config):
    """Create environment with hierarchical roles."""
    try:
        base_env = pyquaticus_v0.PyQuaticusEnv(**config)
        role_env = wrap_env_with_roles(base_env)
        return ParallelPettingZooWrapper(role_env)
    except Exception as e:
        print(f"Error creating environment with roles: {e}")
        print("Falling back to environment without roles")
        base_env = pyquaticus_v0.PyQuaticusEnv(**config)
        return ParallelPettingZooWrapper(base_env)


def extract_role_from_obs(observation):
    """Extract role ID from observation (last 3 elements should be one-hot)."""
    try:
        if observation is None:
            return ATTACK
        
        obs_array = np.asarray(observation)
        if len(obs_array.shape) == 1 and obs_array.shape[0] >= 3:
            role_one_hot = obs_array[-3:]
            role_id = np.argmax(role_one_hot)
            if role_id in [ATTACK, DEFEND, INTERCEPT]:
                return role_id
        return ATTACK  # Default
    except Exception:
        return ATTACK  # Default on any error


def evaluate_episode(algo, env, render=False, debug_roles=False):
    """
    Evaluate a single episode.
    
    Returns:
        dict: Episode statistics
    """
    try:
        obs, info = env.reset()
        terminated = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}
        
        episode_rewards = {agent: 0.0 for agent in env.agents}
        episode_steps = 0
        role_counts = {agent: {ATTACK: 0, DEFEND: 0, INTERCEPT: 0} for agent in env.agents}
        
        while not all(terminated.values()) and not all(truncated.values()):
            # Get actions from trained policy for our agents
            actions = {}
            
            for agent_id in env.agents:
                try:
                    if agent_id in ['agent_0', 'agent_1', 'agent_2']:
                        # Use trained policy
                        agent_obs = obs[agent_id]
                        action, _, _ = algo.compute_single_action(
                            agent_obs,
                            policy_id="hierarchical_policy"
                        )
                        actions[agent_id] = action
                        
                        # Track roles for debugging
                        if debug_roles and agent_id in ['agent_0', 'agent_1', 'agent_2']:
                            role_id = extract_role_from_obs(agent_obs)
                            role_counts[agent_id][role_id] += 1
                            
                    else:
                        # Use random policy for opponents
                        agent_obs = obs[agent_id]
                        action, _, _ = algo.compute_single_action(
                            agent_obs,
                            policy_id="random_policy"
                        )
                        actions[agent_id] = action
                except Exception as e:
                    print(f"Warning: Error getting action for {agent_id}: {e}")
                    # Use random action as fallback
                    actions[agent_id] = env.action_space[agent_id].sample()
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate rewards
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            
            episode_steps += 1
            
            # Render if requested
            if render and episode_steps % 10 == 0:  # Render every 10 steps to avoid slowdown
                try:
                    env.render()
                except Exception:
                    pass  # Ignore rendering errors
        
        # Get final scores
        final_scores = {}
        try:
            if hasattr(env, 'par_env') and hasattr(env.par_env, 'state'):
                state = env.par_env.state
                if 'captures' in state:
                    final_scores['blue'] = state['captures'][Team.BLUE_TEAM.value]
                    final_scores['red'] = state['captures'][Team.RED_TEAM.value]
        except Exception:
            final_scores = {'blue': 0, 'red': 0}
        
        # Determine winner
        blue_wins = final_scores.get('blue', 0) > final_scores.get('red', 0)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'final_scores': final_scores,
            'blue_wins': blue_wins,
            'role_counts': role_counts if debug_roles else None
        }
    except Exception as e:
        print(f"Error during episode evaluation: {e}")
        # Return default values on error
        return {
            'episode_rewards': {agent: 0.0 for agent in env.agents},
            'episode_steps': 0,
            'final_scores': {'blue': 0, 'red': 0},
            'blue_wins': False,
            'role_counts': None
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate hierarchical PPO policy for 3v3 PyQuaticus')
    parser.add_argument('--checkpoint', help='Path to checkpoint file', type=str, required=True)
    parser.add_argument('--episodes', help='Number of evaluation episodes', type=int, default=100)
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    parser.add_argument('--debug-roles', help='Print role distribution debug info', action='store_true')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Environment configuration (same as training)
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True
    config_dict['team_size'] = 3
    
    role_reward_func = create_role_reward_config()
    reward_config = {
        'agent_0': role_reward_func,
        'agent_1': role_reward_func, 
        'agent_2': role_reward_func,
        'agent_3': None,
        'agent_4': None,
        'agent_5': None
    }
    
    env_config = {
        'config_dict': config_dict,
        'render_mode': 'human' if args.render else None,
        'reward_config': reward_config,
        'team_size': 3
    }
    
    # Register environment
    register_env('pyquaticus_hierarchical', lambda config: env_creator_with_roles(config))
    
    # Create environment for evaluation
    env = env_creator_with_roles(env_config)
    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id in ['agent_0', 'agent_1', 'agent_2']:
            return "hierarchical_policy"
        return "random_policy"
    
    policies = {
        'hierarchical_policy': (None, obs_space, act_space, {}),
        'random_policy': (RandPolicy, obs_space, act_space, {"no_checkpoint": True})
    }
    
    # Build algorithm and restore checkpoint
    ppo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env='pyquaticus_hierarchical')
        .env_runners(num_env_runners=1)
    )
    
    ppo_config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=[],  # No training during evaluation
    )
    
    algo = ppo_config.build_algo()
    algo.restore(args.checkpoint)
    
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Running {args.episodes} episodes...")
    print("=" * 50)
    
    # Evaluation metrics
    total_blue_wins = 0
    total_red_wins = 0
    episode_rewards_list = []
    episode_steps_list = []
    all_role_counts = {agent: {ATTACK: 0, DEFEND: 0, INTERCEPT: 0} for agent in ['agent_0', 'agent_1', 'agent_2']}
    
    # Initialize Ray with error handling
    try:
        if not ray.is_initialized():
            ray.init()
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        print("Continuing without Ray...")
    
    try:
        # Run evaluation episodes
        for episode in range(args.episodes):
            episode_stats = evaluate_episode(
                algo, env, render=args.render, debug_roles=args.debug_roles
            )
            
            # Update metrics
            if episode_stats['blue_wins']:
                total_blue_wins += 1
            else:
                total_red_wins += 1
            
            episode_rewards_list.append(episode_stats['episode_rewards'])
            episode_steps_list.append(episode_stats['episode_steps'])
            
            # Accumulate role counts
            if args.debug_roles and episode_stats['role_counts']:
                for agent_id in ['agent_0', 'agent_1', 'agent_2']:
                    for role_id in [ATTACK, DEFEND, INTERCEPT]:
                        all_role_counts[agent_id][role_id] += episode_stats['role_counts'][agent_id][role_id]
            
            # Progress update
            if (episode + 1) % 20 == 0:
                current_winrate = total_blue_wins / (episode + 1) * 100
                print(f"Episode {episode + 1}/{args.episodes} - Blue win rate: {current_winrate:.1f}%")
        
        # Final statistics
        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        
        blue_winrate = total_blue_wins / args.episodes * 100
        red_winrate = total_red_wins / args.episodes * 100
        
        print(f"Total Episodes: {args.episodes}")
        print(f"Blue Team Wins: {total_blue_wins} ({blue_winrate:.1f}%)")
        print(f"Red Team Wins: {total_red_wins} ({red_winrate:.1f}%)")
        
        # Average rewards per agent
        print("\nAverage Episode Rewards:")
        for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']:
            agent_rewards = [rewards.get(agent_id, 0.0) for rewards in episode_rewards_list]
            if agent_rewards:
                avg_reward = np.mean(agent_rewards)
                print(f"  {agent_id}: {avg_reward:.3f}")
        
        # Average episode length
        if episode_steps_list:
            avg_steps = np.mean(episode_steps_list)
            print(f"\nAverage Episode Length: {avg_steps:.1f} steps")
        
        # Role distribution debug info
        if args.debug_roles:
            print("\nRole Distribution (over all episodes):")
            for agent_id in ['agent_0', 'agent_1', 'agent_2']:
                total_role_steps = sum(all_role_counts[agent_id].values())
                if total_role_steps > 0:
                    print(f"  {agent_id}:")
                    for role_id in [ATTACK, DEFEND, INTERCEPT]:
                        count = all_role_counts[agent_id][role_id]
                        percentage = count / total_role_steps * 100
                        print(f"    {ROLE_NAMES[role_id]}: {count} steps ({percentage:.1f}%)")
        
        print("\nEvaluation completed!")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        # Clean up
        try:
            env.close()
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
