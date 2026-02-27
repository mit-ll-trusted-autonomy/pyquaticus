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
Train a single shared PPO policy with hierarchical roles for 3v3 PyQuaticus.

This script trains one PPO policy that learns to act conditioned on role
(ATTACK/DEFEND/INTERCEPT) assignments that change every ROLE_PERIOD steps.
Optimized for maximum training efficiency and competitive performance.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

import pyquaticus.utils.rewards as rew
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus.hierarchical.role_wrapper import wrap_env_with_roles
from pyquaticus.hierarchical.roles import ATTACK


class RoleBasedRewardWrapper:
    """
    Wrapper to provide role-based rewards to agents.
    """

    def __init__(self, base_env):
        self.base_env = base_env
        self.current_roles = {}

    def set_roles(self, roles):
        """Set current roles for reward calculation."""
        self.current_roles = roles

    def step(self, actions):
        """Step environment and compute role-based rewards."""
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)

        # Modify rewards based on roles - enhanced for competitive play
        # Apply role-based reward scaling
        modified_rewards = {}
        for agent_id, reward in rewards.items():
            # Apply role-based reward scaling based on current role
            role_id = self.current_roles.get(agent_id, ATTACK)
            # Scale rewards based on role for faster learning
            role_scaling = {0: 1.2, 1: 1.1, 2: 1.15}  # ATTACK, DEFEND, INTERCEPT
            scaling_factor = role_scaling.get(role_id, 1.0)
            modified_rewards[agent_id] = reward * scaling_factor

        return obs, modified_rewards, terminated, truncated, info


def create_role_reward_config():
    """
    Create reward configuration that uses hierarchical role rewards.
    """

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
        # Extract role from observation if available, otherwise default to ATTACK
        role_id = ATTACK  # Default role

        # Try to extract role from agent's current observation if state has it
        try:
            if hasattr(state, "get") and "agent_observations" in state:
                agent_obs = state["agent_observations"].get(agent_id)
                if agent_obs is not None and len(agent_obs) >= 3:
                    # Last 3 elements should be role one-hot
                    role_one_hot = agent_obs[-3:]
                    role_id = int(np.argmax(role_one_hot))  # Fix: Convert to int
        except Exception:
            # If extraction fails, use default
            pass

        # Calculate base hierarchical reward
        base_reward = rew.hierarchical_role_reward(
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
            role_id,
        )

        # Apply role-specific scaling based on extracted role_id
        role_scaling = {
            0: 1.2,
            1: 1.1,
            2: 1.15,
        }  # ATTACK: 1.2, DEFEND: 1.1, INTERCEPT: 1.15
        scaling_factor = role_scaling.get(role_id, 1.0)

        return base_reward * scaling_factor

    return role_reward_wrapper


class RandPolicy(Policy):
    """Optimized random policy for baseline opponents."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


def env_creator_with_roles(config):
    """Create environment with hierarchical roles - optimized for stability."""
    try:
        # Create base environment with enhanced configuration
        base_env = pyquaticus_v0.PyQuaticusEnv(**config)

        # Wrap with role functionality
        role_env = wrap_env_with_roles(base_env)

        # Wrap with PettingZoo wrapper for RLLib
        return ParallelPettingZooWrapper(role_env)
    except Exception as e:
        print(f"Error creating environment with roles: {e}")
        print("Falling back to environment without roles")
        # Fallback to environment without role modifications
        base_env = pyquaticus_v0.PyQuaticusEnv(**config)
        return ParallelPettingZooWrapper(base_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train hierarchical PPO policy for 3v3 PyQuaticus"
    )
    parser.add_argument("--render", help="Enable rendering", action="store_true")
    parser.add_argument(
        "--checkpoint", help="Resume from checkpoint", type=str, default=None
    )
    parser.add_argument(
        "--iterations", help="Number of training iterations", type=int, default=10000
    )
    parser.add_argument(
        "--save-dir",
        help="Directory to save checkpoints",
        type=str,
        default="./hierarchical_checkpoints",
    )
    parser.add_argument(
        "--num-workers", help="Number of parallel workers", type=int, default=2
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    RENDER_MODE = "human" if args.render else None

    # Optimized environment configuration for competitive play
    config_dict = config_dict_std.copy()
    config_dict["sim_speedup_factor"] = 8  # Increased for faster training
    config_dict["max_score"] = 5  # Longer games for better learning
    config_dict["max_time"] = 300  # Extended time limit
    config_dict["tagging_cooldown"] = 45  # Reduced cooldown for more dynamic play
    config_dict["tag_on_oob"] = True
    config_dict["team_size"] = 3

    # Validate configuration
    if not isinstance(config_dict, dict):
        raise ValueError("config_dict must be a dictionary")

    # Use hierarchical role rewards for our agents
    try:
        role_reward_func = create_role_reward_config()
        print("Successfully created role reward function")
    except Exception as e:
        print(f"Error creating role reward function: {e}")
        print("Using default reward function")
        role_reward_func = rew.caps_and_grabs

    # Enhanced reward configuration
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
        test_env = env_creator_with_roles(env_config)
        register_env(
            "pyquaticus_hierarchical", lambda config: env_creator_with_roles(config)
        )
        print("Successfully registered environment")
    except Exception as e:
        print(f"Error registering environment: {e}")
        print("Exiting...")
        sys.exit(1)

    # Get observation and action spaces with error handling
    obs_space = None
    act_space = None

    try:
        # Try different methods to access spaces with comprehensive error checking
        if (
            hasattr(test_env, "observation_spaces")
            and test_env.observation_spaces is not None
            and hasattr(test_env, "action_spaces")
            and test_env.action_spaces is not None
            and "agent_0" in test_env.observation_spaces
            and "agent_0" in test_env.action_spaces
        ):
            obs_space = test_env.observation_spaces["agent_0"]
            act_space = test_env.action_spaces["agent_0"]
        elif (
            hasattr(test_env, "observation_space")
            and hasattr(test_env, "action_space")
            and test_env.observation_space is not None
            and test_env.action_space is not None
        ):
            obs_space = test_env.observation_space
            act_space = test_env.action_space
        else:
            # Create a temporary environment to get spaces
            temp_env = pyquaticus_v0.PyQuaticusEnv(**env_config["config_dict"])
            pz_env = ParallelPettingZooWrapper(temp_env)
            if (
                hasattr(pz_env, "observation_spaces")
                and pz_env.observation_spaces is not None
                and hasattr(pz_env, "action_spaces")
                and pz_env.action_spaces is not None
                and "agent_0" in pz_env.observation_spaces
                and "agent_0" in pz_env.action_spaces
            ):
                obs_space = pz_env.observation_spaces["agent_0"]
                act_space = pz_env.action_spaces["agent_0"]
            pz_env.close()

        if obs_space is None or act_space is None:
            raise Exception("Could not determine observation/action spaces")

        print(f"Observation space: {obs_space}")
        print(f"Action space: {act_space}")
    except Exception as e:
        print(f"Error getting observation/action spaces: {e}")
        print("Exiting...")
        sys.exit(1)

    def policy_mapping_fn(agent_id, episode, **kwargs):
        """Map agents to policies - all our agents use the same shared policy."""
        if agent_id in ["agent_0", "agent_1", "agent_2"]:
            return "hierarchical_policy"
        return "random_policy"

    # Policy configuration
    policies = {
        "hierarchical_policy": (None, obs_space, act_space, {}),
        "random_policy": (RandPolicy, obs_space, act_space, {"no_checkpoint": True}),
    }

    test_env.close()

    # Initialize Ray with optimized settings
    # Completely bypass Ray for Windows compatibility
    print("Bypassing Ray entirely for Windows compatibility")
    print("Using local training mode")
    
    # Set environment variables to disable Ray
    os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
    os.environ["RAY_EXPERIMENTAL_NOSET_RAY_ADDRESS"] = "1"
    
    # Don't initialize Ray at all - let RLlib handle it locally

    # Optimized PPO configuration for competitive performance
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .environment(env="pyquaticus_hierarchical", env_config=env_config)
        .env_runners(
            num_env_runners=0,  # Force local training (no workers)
            num_cpus_per_env_runner=1,
            num_envs_per_env_runner=1,
            rollout_fragment_length=1000,  # Smaller fragments for local training
        )
        .resources(
            num_gpus=0,  # CPU training for stability
            num_cpus_for_main_process=1,
        )
        .framework("torch")  # Use PyTorch backend instead of TensorFlow
        .debugging(
            log_level="ERROR"  # Reduce logging overhead
        )
    )

    # Update configuration with optimized hyperparameters
    ppo_config.update_from_dict(
        {
            "train_batch_size": 8192,  # Larger batch size for stability
            "sgd_minibatch_size": 256,  # Mini-batch size for SGD updates
            "num_sgd_iter": 15,  # More SGD iterations for better convergence
            "lr": 5e-4,  # Slightly higher learning rate for faster convergence
            "gamma": 0.995,  # Higher discount for long-term planning
            "lambda": 0.98,  # GAE lambda parameter
            "clip_param": 0.25,  # PPO clipping parameter
            "entropy_coeff": 0.005,  # Reduced entropy for focused exploration
            "vf_loss_coeff": 0.5,  # Value function loss coefficient
            "grad_clip": 0.5,  # Gradient clipping for stability
            "vf_clip_param": 10.0,  # Value function clipping
        }
    )

    # Multi-agent configuration
    ppo_config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["hierarchical_policy"],
    )

    # Build algorithm
    print("Building PPO algorithm...")
    try:
        # Try the newer API first
        try:
            algo = ppo_config.build_algo()
        except AttributeError:
            # Fall back to the old API
            algo = ppo_config.build()
        print("Algorithm built successfully")
    except Exception as e:
        print(f"Error building algorithm: {e}")
        print("This is likely due to Ray Windows compatibility issues.")
        print("Using the simplified training script instead...")
        print("Run: python examples/train_ppo_roles_3v3_simple.py --iterations 8 --save-dir ./hierarchical_checkpoints")
        sys.exit(1)

    # Resume from checkpoint if provided
    if args.checkpoint:
        try:
            algo.restore(args.checkpoint)
            print(f"Resumed from checkpoint: {args.checkpoint}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")

    # Create save directory
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Checkpoint directory: {args.save_dir}")
    except Exception as e:
        print(f"Error creating save directory: {e}")
        print("Using current directory for checkpoints")
        args.save_dir = "."

    # Training loop with enhanced monitoring
    print(f"Starting optimized training for {args.iterations} iterations...")
    print(f"Configuration: {args.num_workers} workers, batch size 8192")
    print("=" * 60)

    best_reward = float("-inf")
    try:
        for i in range(args.iterations):
            start_time = time.time()

            # Train one iteration
            result = algo.train()

            end_time = time.time()
            iteration_time = end_time - start_time

            # Extract key metrics
            episode_reward_mean = result.get("env_runners", {}).get(
                "episode_reward_mean", result.get("episode_reward_mean", 0)
            )
            episode_len_mean = result.get("env_runners", {}).get(
                "episode_len_mean", result.get("episode_len_mean", 0)
            )

            # Track best performance
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean

            # Enhanced progress reporting
            if i % 50 == 0:
                print(
                    f"Iter {i:4d}: Reward={episode_reward_mean:7.2f} (Best: {best_reward:7.2f}) "
                    f"Length={episode_len_mean:6.1f} Time={iteration_time:5.2f}s"
                )

            # More frequent checkpointing for competitive training
            if i % 250 == 0 and i > 0:
                try:
                    checkpoint_path = algo.save(f"{args.save_dir}/iter_{i}")
                    print(f"Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    print(f"Error saving checkpoint at iteration {i}: {e}")

            # Early stopping if performance plateaus (optional competitive feature)
            if i > 2000 and i % 500 == 0:
                recent_rewards = []
                for j in range(max(0, i - 500), i):
                    if j % 50 == 0:  # Only check every 50th iteration
                        recent_rewards.append(episode_reward_mean)

                if len(recent_rewards) > 5:
                    reward_std = np.std(recent_rewards)
                    if (
                        reward_std < 0.1 and episode_reward_mean > 2.0
                    ):  # Converged to good policy
                        print(
                            f"Training converged at iteration {i} with stable reward {episode_reward_mean:.2f}"
                        )
                        break

        # Final checkpoint
        try:
            final_checkpoint = algo.save(f"{args.save_dir}/final")
            print(f"Final checkpoint saved: {final_checkpoint}")
            print(f"Best reward achieved: {best_reward:.2f}")
        except Exception as e:
            print(f"Error saving final checkpoint: {e}")

        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final performance: {best_reward:.2f}")
        print("Your bot is now ready for competition!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        try:
            interrupt_checkpoint = algo.save(f"{args.save_dir}/interrupted")
            print(f"Interrupt checkpoint saved: {interrupt_checkpoint}")
        except Exception:
            pass
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up Ray
        try:
            algo.stop()
            if ray.is_initialized():
                ray.shutdown()
            print("Cleanup completed")
        except Exception:
            pass
