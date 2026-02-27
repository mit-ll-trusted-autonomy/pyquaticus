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

This script loads a PyTorch checkpoint and evaluates the trained policy against
baseline/random opponents over multiple episodes.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import time
import os
from pyquaticus.envs.pyquaticus import Team
from pyquaticus import pyquaticus_v0
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus.hierarchical.role_wrapper import wrap_env_with_roles
from pyquaticus.hierarchical.roles import ATTACK, DEFEND, INTERCEPT, ROLE_NAMES
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std
import logging


class SimplePolicy(nn.Module):
    """Simple neural network policy for PPO (same as training)."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(SimplePolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def create_role_reward_config():
    """Create reward configuration that uses hierarchical role rewards."""
    
    def role_reward_wrapper(
        agent_id,
        team,
        agents,
        agent_inds_of_team,
        state,
        prev_state,
        env_size,
        agent_radius,
        catch_radius,
        scrimmage_coords,
        max_speeds,
        tagging_cooldown,
    ):
        role_id = ATTACK  # Default role
        
        try:
            if hasattr(state, "get") and "agent_observations" in state:
                agent_obs = state["agent_observations"].get(agent_id)
                if agent_obs is not None and len(agent_obs) >= 3:
                    role_one_hot = agent_obs[-3:]
                    role_id = int(np.argmax(role_one_hot))
        except Exception:
            pass
        
        base_reward = rew.hierarchical_role_reward(
            agent_id, team, agents, agent_inds_of_team, state, prev_state,
            env_size, agent_radius, catch_radius, scrimmage_coords,
            max_speeds, tagging_cooldown, role_id,
        )
        
        role_scaling = {0: 1.2, 1: 1.1, 2: 1.15}
        scaling_factor = role_scaling.get(role_id, 1.0)
        
        return base_reward * scaling_factor
    
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


def load_checkpoint(checkpoint_path, obs_dim, action_dim):
    """Load trained policies from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create agents and load weights
    agents = {}
    for agent_id in ["agent_0", "agent_1", "agent_2"]:
        agent = SimplePolicy(obs_dim, action_dim)
        if agent_id in checkpoint:
            agent.load_state_dict(checkpoint[agent_id])
            agent.eval()  # Set to evaluation mode
            print(f"Loaded weights for {agent_id}")
        else:
            print(f"Warning: No weights found for {agent_id}")
        agents[agent_id] = agent
    
    return agents


def evaluate_episode(agents, env, render=False, debug_roles=False):
    """
    Evaluate a single episode.
    
    Returns:
        dict: Episode statistics
    """
    try:
        obs, info = env.reset()
        print(f"Environment reset. Agents: {env.agents}")
        print(f"Initial obs keys: {list(obs.keys()) if obs else 'None'}")
        
        # Handle case where env.agents is empty but we have observations
        if not env.agents and obs:
            agent_list = list(obs.keys())
            print(f"Using agents from observations: {agent_list}")
        else:
            agent_list = env.agents
        
        terminated = {agent: False for agent in agent_list}
        truncated = {agent: False for agent in agent_list}
        
        episode_rewards = {agent: 0.0 for agent in agent_list}
        episode_steps = 0
        role_counts = {agent: {ATTACK: 0, DEFEND: 0, INTERCEPT: 0} for agent in agent_list}
        
        max_steps = 1000  # Safety limit
        
        # Continue while we have observations and not all agents are done
        step_count = 0
        while obs and step_count < max_steps:
            # Get actions from trained policy for our agents
            actions = {}
            
            for agent_id in agent_list:
                if agent_id not in obs:
                    continue
                    
                try:
                    if agent_id in ['agent_0', 'agent_1', 'agent_2']:
                        # Use trained policy
                        agent_obs = obs[agent_id]
                        agent = agents[agent_id]
                        
                        with torch.no_grad():
                            state = torch.FloatTensor(agent_obs)
                            action_probs = torch.softmax(agent(state), dim=-1)
                            dist = Categorical(action_probs)
                            action = dist.sample().item()
                        
                        actions[agent_id] = action
                        
                        # Track roles for debugging
                        if debug_roles:
                            role_id = extract_role_from_obs(agent_obs)
                            role_counts[agent_id][role_id] += 1
                            
                    else:
                        # Use random policy for opponents
                        actions[agent_id] = env.action_space[agent_id].sample()
                except Exception as e:
                    print(f"Warning: Error getting action for {agent_id}: {e}")
                    # Use random action as fallback
                    actions[agent_id] = env.action_space[agent_id].sample()
            
            if not actions:
                print("No actions generated, ending episode")
                break
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate rewards
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            
            episode_steps += 1
            step_count += 1
            
            # Debug info for first few steps
            if episode_steps <= 3:
                print(f"Step {episode_steps}: rewards={rewards}, terminated={terminated}, truncated={truncated}")
            
            # Check if episode should end
            if obs and all(terminated.get(agent, False) or truncated.get(agent, False) for agent in obs.keys()):
                print("All agents terminated or truncated")
                break
            
            # Render if requested
            if render and episode_steps % 10 == 0:  # Render every 10 steps to avoid slowdown
                try:
                    env.render()
                except Exception:
                    pass  # Ignore rendering errors
        
        print(f"Episode ended after {episode_steps} steps")
        
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
        import traceback
        traceback.print_exc()
        # Return default values on error
        return {
            'episode_rewards': {agent: 0.0 for agent in ['agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']},
            'episode_steps': 0,
            'final_scores': {'blue': 0, 'red': 0},
            'blue_wins': False,
            'role_counts': None
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate hierarchical PPO policy for 3v3 PyQuaticus')
    parser.add_argument('--checkpoint', help='Path to PyTorch checkpoint file (.pt)', type=str, required=True)
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
    config_dict['sim_speedup_factor'] = 8  # Same as training
    config_dict['max_score'] = 5  # Same as training
    config_dict['max_time'] = 300  # Same as training
    config_dict['tagging_cooldown'] = 45  # Same as training
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
    
    # Create environment for evaluation
    env = env_creator_with_roles(env_config)
    
    # Get observation and action spaces
    try:
        if hasattr(env, 'observation_spaces') and env.observation_spaces is not None:
            obs_space = env.observation_spaces["agent_0"]
            act_space = env.action_spaces["agent_0"]
        elif hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
            obs_space = env.observation_space
            act_space = env.action_space
        else:
            raise Exception("Could not determine observation/action spaces")
        
        # Handle Dict spaces
        if hasattr(obs_space, 'spaces'):
            obs_space = obs_space.spaces["agent_0"]
        if hasattr(act_space, 'spaces'):
            act_space = act_space.spaces["agent_0"]
        
        obs_dim = obs_space.shape[0]
        action_dim = act_space.n
        print(f"Observation space: {obs_space}")
        print(f"Action space: {act_space}")
    except Exception as e:
        print(f"Error getting spaces: {e}")
        env.close()
        sys.exit(1)
    
    # Load trained agents
    try:
        agents = load_checkpoint(args.checkpoint, obs_dim, action_dim)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        env.close()
        sys.exit(1)
    
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Running {args.episodes} episodes...")
    print("=" * 50)
    
    # Evaluation metrics
    total_blue_wins = 0
    total_red_wins = 0
    episode_rewards_list = []
    episode_steps_list = []
    all_role_counts = {agent: {ATTACK: 0, DEFEND: 0, INTERCEPT: 0} for agent in ['agent_0', 'agent_1', 'agent_2']}
    
    try:
        # Run evaluation episodes
        for episode in range(args.episodes):
            episode_stats = evaluate_episode(
                agents, env, render=args.render, debug_roles=args.debug_roles
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
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            env.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
