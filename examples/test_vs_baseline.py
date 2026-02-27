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
# Part 252.227-7013 or 7024 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
Test trained hierarchical policy against intelligent baseline opponents.
"""

import argparse
import gymnasium as gym
import numpy as np
import pygame
import ray
from ray.rllib.algorithms.ppo import PPOConfig
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

class BaselinePolicy(Policy):
    """Wrapper for baseline policies."""
    
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.policy = config.get("baseline_policy")
        if self.policy:
            self.policy.set_team(config.get("team", Team.BLUE_TEAM))
            self.policy.set_agent_obs_normalizer(config.get("obs_normalizer"))

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        for obs in obs_batch:
            if self.policy:
                action = self.policy.compute_action(obs)
                actions.append(action)
            else:
                actions.append(self.action_space.sample())
        return actions, [], {}

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
        base_env = pyquaticus_v0.PyQuaticusEnv(**config)
        return ParallelPettingZooWrapper(base_env)

def evaluate_episode(algo, env, render=False):
    """Evaluate one episode against baseline opponents."""
    obs, info = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    
    episode_rewards = {agent: 0.0 for agent in env.agents}
    episode_steps = 0
    
    while not all(terminated.values()) and not all(truncated.values()):
        # Get actions
        actions = {}
        
        for agent_id in env.agents:
            try:
                if agent_id in ['agent_0', 'agent_1', 'agent_2']:
                    # Use trained hierarchical policy
                    agent_obs = obs[agent_id]
                    action, _, _ = algo.compute_single_action(
                        agent_obs,
                        policy_id="hierarchical_policy"
                    )
                    actions[agent_id] = action
                else:
                    # Use baseline policies
                    agent_obs = obs[agent_id]
                    action, _, _ = algo.compute_single_action(
                        agent_obs,
                        policy_id="baseline_policy"
                    )
                    actions[agent_id] = action
            except Exception as e:
                print(f"Warning: Error getting action for {agent_id}: {e}")
                actions[agent_id] = env.action_space[agent_id].sample()
        
        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Accumulate rewards
        for agent_id, reward in rewards.items():
            episode_rewards[agent_id] += reward
        
        episode_steps += 1
        
        # Render if requested
        if render and episode_steps % 10 == 0:
            try:
                env.render()
            except Exception:
                pass
    
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
    
    blue_wins = final_scores.get('blue', 0) > final_scores.get('red', 0)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'final_scores': final_scores,
        'blue_wins': blue_wins
    }

def main():
    parser = argparse.ArgumentParser(description='Test hierarchical policy vs baseline opponents')
    parser.add_argument('--checkpoint', help='Path to trained checkpoint', type=str, required=True)
    parser.add_argument('--episodes', help='Number of evaluation episodes', type=int, default=100)
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    parser.add_argument('--opponent', help='Opponent type (random/attack/defend/mixed)', type=str, default='mixed')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Environment configuration
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
        'agent_3': role_reward_func,  # Baseline opponents also use role rewards
        'agent_4': role_reward_func,
        'agent_5': role_reward_func
    }
    
    env_config = {
        'config_dict': config_dict,
        'render_mode': 'human' if args.render else None,
        'reward_config': reward_config,
        'team_size': 3
    }
    
    # Register environment
    register_env('pyquaticus_hierarchical', lambda config: env_creator_with_roles(config))
    
    # Create environment
    env = env_creator_with_roles(env_config)
    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id in ['agent_0', 'agent_1', 'agent_2']:
            return "hierarchical_policy"
        return "baseline_policy"
    
    # Create baseline policies based on opponent type
    baseline_policies = {}
    
    if args.opponent == 'random':
        # Random baseline
        class RandomPolicy(Policy):
            def __init__(self, observation_space, action_space, config):
                Policy.__init__(self, observation_space, action_space, config)
            def compute_actions(self, obs_batch, state_batches, **kwargs):
                return [self.action_space.sample() for _ in obs_batch], [], {}
            def get_weights(self): return {}
            def learn_on_batch(self, samples): return {}
            def set_weights(self, weights): pass
        
        baseline_policies['baseline_policy'] = (RandomPolicy, obs_space, act_space, {"no_checkpoint": True})
        
    elif args.opponent == 'attack':
        # Attack policies
        for i in range(3):
            agent_id = f'agent_{i+3}'
            attack_policy = AttackGen(3, Team.RED_TEAM, 'easy', i+3, env.par_env.agent_obs_normalizer if hasattr(env, 'par_env') else None)
            baseline_policies[f'baseline_policy_{i}'] = (BaselinePolicy, obs_space, act_space, {
                "baseline_policy": attack_policy,
                "team": Team.RED_TEAM,
                "obs_normalizer": env.par_env.agent_obs_normalizer if hasattr(env, 'par_env') else None
            })
    
    elif args.opponent == 'defend':
        # Defend policies
        for i in range(3):
            agent_id = f'agent_{i+3}'
            defend_policy = DefendGen(2, Team.RED_TEAM, 'easy', i+3, env.par_env.agent_obs_normalizer if hasattr(env, 'par_env') else None)
            baseline_policies[f'baseline_policy_{i}'] = (BaselinePolicy, obs_space, act_space, {
                "baseline_policy": defend_policy,
                "team": Team.RED_TEAM,
                "obs_normalizer": env.par_env.agent_obs_normalizer if hasattr(env, 'par_env') else None
            })
    
    else:  # mixed
        # Mix of attack and defend
        attack_policy = AttackGen(3, Team.RED_TEAM, 'easy', 3, env.par_env.agent_obs_normalizer if hasattr(env, 'par_env') else None)
        defend_policy = DefendGen(2, Team.RED_TEAM, 'easy', 4, env.par_env.agent_obs_normalizer if hasattr(env, 'par_env') else None)
        random_policy = RandomPolicy(obs_space, act_space, {"no_checkpoint": True})
        
        baseline_policies['baseline_policy_0'] = (BaselinePolicy, obs_space, act_space, {
            "baseline_policy": attack_policy, "team": Team.RED_TEAM,
            "obs_normalizer": env.par_env.agent_obs_normalizer if hasattr(env, 'par_env') else None
        })
        baseline_policies['baseline_policy_1'] = (BaselinePolicy, obs_space, act_space, {
            "baseline_policy": defend_policy, "team": Team.RED_TEAM,
            "obs_normalizer": env.par_env.agent_obs_normalizer if hasattr(env, 'par_env') else None
        })
        baseline_policies['baseline_policy_2'] = (RandomPolicy, obs_space, act_space, {"no_checkpoint": True})
    
    # Policy configuration
    policies = {
        'hierarchical_policy': (None, obs_space, act_space, {}),
    }
    policies.update(baseline_policies)
    
    def dynamic_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id in ['agent_0', 'agent_1', 'agent_2']:
            return "hierarchical_policy"
        elif args.opponent == 'mixed':
            if agent_id == 'agent_3':
                return "baseline_policy_0"
            elif agent_id == 'agent_4':
                return "baseline_policy_1"
            else:
                return "baseline_policy_2"
        else:
            return "baseline_policy"
    
    # Build algorithm
    ppo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env='pyquaticus_hierarchical')
        .env_runners(num_env_runners=1)
    )
    
    ppo_config.multi_agent(
        policies=policies,
        policy_mapping_fn=dynamic_policy_mapping_fn,
        policies_to_train=[],  # No training during evaluation
    )
    
    algo = ppo_config.build_algo()
    algo.restore(args.checkpoint)
    
    print(f"Testing hierarchical policy vs {args.opponent} opponents")
    print(f"Running {args.episodes} episodes...")
    print("=" * 50)
    
    # Evaluation
    total_blue_wins = 0
    total_red_wins = 0
    episode_rewards_list = []
    episode_steps_list = []
    
    for episode in range(args.episodes):
        episode_stats = evaluate_episode(algo, env, render=args.render)
        
        if episode_stats['blue_wins']:
            total_blue_wins += 1
        else:
            total_red_wins += 1
        
        episode_rewards_list.append(episode_stats['episode_rewards'])
        episode_steps_list.append(episode_stats['episode_steps'])
        
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
    print(f"Hierarchical (Blue) Wins: {total_blue_wins} ({blue_winrate:.1f}%)")
    print(f"Baseline (Red) Wins: {total_red_wins} ({red_winrate:.1f}%)")
    
    # Average rewards
    print("\nAverage Episode Rewards:")
    for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']:
        agent_rewards = [rewards.get(agent_id, 0.0) for rewards in episode_rewards_list]
        if agent_rewards:
            avg_reward = np.mean(agent_rewards)
            team = "Hierarchical" if int(agent_id.split('_')[1]) < 3 else f"Baseline-{args.opponent}"
            print(f"  {agent_id} ({team}): {avg_reward:.3f}")
    
    if episode_steps_list:
        avg_steps = np.mean(episode_steps_list)
        print(f"\nAverage Episode Length: {avg_steps:.1f} steps")
    
    print(f"\n✅ Evaluation completed!")
    
    env.close()
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

if __name__ == '__main__':
    main()
