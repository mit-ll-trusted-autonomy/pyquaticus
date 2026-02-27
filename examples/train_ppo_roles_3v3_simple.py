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
Simple training script for 3v3 PyQuaticus without Ray RLlib dependencies.
This uses a basic PPO implementation that works on Windows without Ray.
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import pyquaticus.utils.rewards as rew
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus.hierarchical.role_wrapper import wrap_env_with_roles
from pyquaticus.hierarchical.roles import ATTACK


class SimplePolicy(nn.Module):
    """Simple neural network policy for PPO."""
    
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


class PPOAgent:
    """Simple PPO agent implementation."""
    
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.policy = SimplePolicy(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.memory = []
        
    def select_action(self, state, agent_id=None):
        # Handle both single agent and multi-agent observations
        if isinstance(state, dict):
            # If it's a dict, extract the observation for this agent
            if agent_id and agent_id in state:
                obs_array = state[agent_id]
            else:
                # Fallback to first agent's observation
                obs_array = list(state.values())[0]
        else:
            obs_array = state
        
        state_tensor = torch.FloatTensor(obs_array)
        with torch.no_grad():
            action_probs = torch.softmax(self.policy(state_tensor), dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def update(self):
        if len(self.memory) < 10:  # Need some experience to update
            return
            
        # Convert memory to tensors - handle multi-agent structure
        states = []
        actions = []
        rewards = []
        old_log_probs = []
        
        for exp in self.memory:
            # Handle both single agent and multi-agent observations
            if isinstance(exp['state'], dict):
                # If it's a dict, extract the observation for this agent
                # We need to identify which agent this belongs to
                agent_id = None
                for key in exp['state']:
                    if key.startswith('agent_'):
                        agent_id = key
                        break
                
                if agent_id and agent_id in exp['state']:
                    state = exp['state'][agent_id]
                else:
                    # Fallback to first agent's observation
                    state = list(exp['state'].values())[0]
            else:
                state = exp['state']
            
            states.append(state)
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            old_log_probs.append(exp['log_prob'])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Compute discounted rewards
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(rewards.numpy()):
            running_reward = reward + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # PPO epochs
            # Get current action probabilities
            action_probs = torch.softmax(self.policy(states), dim=-1)
            dist = Categorical(action_probs)
            curr_log_probs = dist.log_prob(actions)
            
            # Compute ratio
            ratio = torch.exp(curr_log_probs - old_log_probs)
            
            # Compute surrogate loss
            surr1 = ratio * discounted_rewards
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * discounted_rewards
            loss = -torch.min(surr1, surr2).mean()
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.memory = []


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


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical PPO policy for 3v3 PyQuaticus")
    parser.add_argument("--render", help="Enable rendering", action="store_true")
    parser.add_argument("--iterations", help="Number of training iterations", type=int, default=10000)
    parser.add_argument("--save-dir", help="Directory to save checkpoints", type=str, default="./hierarchical_checkpoints")
    parser.add_argument("--episodes", help="Episodes per iteration", type=int, default=10)
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    
    RENDER_MODE = "human" if args.render else None
    
    # Environment configuration
    config_dict = config_dict_std.copy()
    config_dict["sim_speedup_factor"] = 8
    config_dict["max_score"] = 5
    config_dict["max_time"] = 300
    config_dict["tagging_cooldown"] = 45
    config_dict["tag_on_oob"] = True
    config_dict["team_size"] = 3
    
    # Create reward configuration
    try:
        role_reward_func = create_role_reward_config()
        print("Successfully created role reward function")
    except Exception as e:
        print(f"Error creating role reward function: {e}")
        role_reward_func = rew.caps_and_grabs
    
    reward_config = {
        "agent_0": role_reward_func,
        "agent_1": role_reward_func,
        "agent_2": role_reward_func,
        "agent_3": None,  # Random opponents
        "agent_4": None,
        "agent_5": None,
    }
    
    # Create environment
    env_config = {
        "config_dict": config_dict,
        "render_mode": RENDER_MODE,
        "reward_config": reward_config,
        "team_size": 3,
    }
    
    try:
        env = env_creator_with_roles(env_config)
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        sys.exit(1)
    
    # Get observation and action spaces
    try:
        # Debug: print environment attributes
        print(f"Environment type: {type(env)}")
        print(f"Environment attributes: {[attr for attr in dir(env) if 'space' in attr.lower()]}")
        
        # Try different methods to access spaces
        if hasattr(env, 'observation_spaces') and env.observation_spaces is not None:
            print("Using observation_spaces")
            obs_space = env.observation_spaces["agent_0"]
            act_space = env.action_spaces["agent_0"]
        elif hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
            print("Using observation_space")
            obs_space = env.observation_space
            act_space = env.action_space
        else:
            # Try to get spaces from a sample observation
            print("Trying to get spaces from sample observation")
            obs, info = env.reset()
            print(f"Sample obs keys: {list(obs.keys()) if obs else 'None'}")
            if obs and "agent_0" in obs:
                sample_obs = obs["agent_0"]
                obs_dim = len(sample_obs) if hasattr(sample_obs, '__len__') else sample_obs.shape[0] if hasattr(sample_obs, 'shape') else None
                # Get action space from environment
                if hasattr(env, 'action_spaces') and env.action_spaces:
                    act_space = env.action_spaces["agent_0"]
                elif hasattr(env, 'action_space'):
                    act_space = env.action_space
                else:
                    # Default action space
                    act_space = type('ActionSpace', (), {'n': 17})()
                print(f"Derived obs_dim: {obs_dim}, action_dim: {act_space.n}")
                obs_dim = obs_dim
                action_dim = act_space.n
            else:
                raise Exception("Could not determine observation/action spaces")
        
        if 'obs_space' in locals() and 'act_space' in locals():
            # Handle Dict spaces
            if hasattr(obs_space, 'spaces'):
                # It's a Dict space, get the space for agent_0
                obs_space = obs_space.spaces["agent_0"]
            
            if hasattr(act_space, 'spaces'):
                # It's a Dict space, get the space for agent_0
                act_space = act_space.spaces["agent_0"]
            
            obs_dim = obs_space.shape[0]
            action_dim = act_space.n
            print(f"Observation space: {obs_space}")
            print(f"Action space: {act_space}")
    except Exception as e:
        print(f"Error getting spaces: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        sys.exit(1)
    
    # Create agents
    agents = {}
    for agent_id in ["agent_0", "agent_1", "agent_2"]:
        agents[agent_id] = PPOAgent(obs_dim, action_dim)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {args.iterations} iterations...")
    best_reward = float("-inf")
    
    try:
        for iteration in range(args.iterations):
            start_time = time.time()
            episode_rewards = []
            
            for episode in range(args.episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # Get actions from our agents
                    actions = {}
                    log_probs = {}
                    
                    for agent_id in ["agent_0", "agent_1", "agent_2"]:
                        if agent_id in obs:
                            action, log_prob = agents[agent_id].select_action(obs, agent_id)
                            actions[agent_id] = action
                            log_probs[agent_id] = log_prob
                    
                    # Random actions for opponents
                    for agent_id in ["agent_3", "agent_4", "agent_5"]:
                        if agent_id in obs:
                            actions[agent_id] = act_space.sample()
                    
                    # Step environment
                    next_obs, rewards, terminated, truncated, info = env.step(actions)
                    done = all(terminated.values()) or all(truncated.values())
                    
                    # Store experience for our agents
                    for agent_id in ["agent_0", "agent_1", "agent_2"]:
                        if agent_id in rewards:
                            # Store the full observation dict for context
                            agents[agent_id].memory.append({
                                'state': obs,  # Store full observation dict
                                'action': actions[agent_id],
                                'reward': rewards[agent_id],
                                'log_prob': log_probs[agent_id]
                            })
                            episode_reward += rewards[agent_id]
                    
                    obs = next_obs
                
                episode_rewards.append(episode_reward)
            
            # Update agents
            for agent_id in ["agent_0", "agent_1", "agent_2"]:
                agents[agent_id].update()
            
            # Calculate metrics
            mean_reward = np.mean(episode_rewards)
            if mean_reward > best_reward:
                best_reward = mean_reward
            
            end_time = time.time()
            iteration_time = end_time - start_time
            
            # Progress reporting
            if iteration % 10 == 0:
                print(f"Iter {iteration:4d}: Reward={mean_reward:7.2f} (Best: {best_reward:7.2f}) Time={iteration_time:5.2f}s")
            
            # Save checkpoint
            if iteration % 100 == 0 and iteration > 0:
                checkpoint_path = f"{args.save_dir}/iter_{iteration}.pt"
                torch.save({agent_id: agent.policy.state_dict() for agent_id, agent in agents.items()}, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        # Final checkpoint
        final_checkpoint = f"{args.save_dir}/final.pt"
        torch.save({agent_id: agent.policy.state_dict() for agent_id, agent in agents.items()}, final_checkpoint)
        print(f"Final checkpoint saved: {final_checkpoint}")
        print(f"Best reward achieved: {best_reward:.2f}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupt_checkpoint = f"{args.save_dir}/interrupted.pt"
        torch.save({agent_id: agent.policy.state_dict() for agent_id, agent in agents.items()}, interrupt_checkpoint)
        print(f"Interrupt checkpoint saved: {interrupt_checkpoint}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
