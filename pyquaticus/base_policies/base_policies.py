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

from collections import OrderedDict
from ray.rllib.policy.policy import PolicySpec, Policy
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.utils.obs_utils import ObsNormalizer
import numpy as np
import pyquaticus.envs.pyquaticus as pyq
from pyquaticus.envs.pyquaticus import Team


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
        
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass


class NoOp(Policy):
    """
    No-op policy - stays in place

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

        return [16], [], {}

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass
    
def AttackGen(agentid, team, mode, team_size):

    class AttackPolicy(Policy):
        """
        Creates an attacker policy
        """

        policy = None
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseAttacker(agentid, team, mode=mode)
            self.action_dict = OrderedDict([(p, 16) for p in range(2*team_size)])
            self.normalizer = register_state_elements(team_size)

        def compute_actions(self,
                            obs_batch,
                            state_batches,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs):
            
            # Iterate over all observations in obs_batch
            for obs in obs_batch[0]:

                # Unnormalize and unpack numpy arrays to create a new dictionary
                new_obs_dict = OrderedDict()
                for key, value in self.normalizer.unnormalized(obs).items():
                    if isinstance(value, np.ndarray) and value.size == 1:
                        new_obs_dict[key] = value.item()  # Unpack single-value numpy arrays
                    else:
                        new_obs_dict[key] = value  # Keep original value

                # Compute action and add it to the action dictionary
                self.action_dict[agentid] = self.policy.compute_action({agentid : new_obs_dict})

            return [self.action_dict[agentid]], [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass
        
    return AttackPolicy

def DefendGen(agentid, team, mode, team_size):
    class DefendPolicy(Policy):
        """
        Creates a defender policy
        """
        policy = None
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseDefender(agentid, team, mode=mode)
            self.action_dict = OrderedDict([(p, 16) for p in range(2*team_size)])
            self.normalizer = register_state_elements(team_size)

        def compute_actions(self,
                            obs_batch,
                            state_batches,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs):
            # Iterate over all observations in obs_batch
            for obs in obs_batch[0]:

                # Unnormalize and unpack numpy arrays to create a new dictionary
                new_obs_dict = OrderedDict()
                for key, value in self.normalizer.unnormalized(obs).items():
                    if isinstance(value, np.ndarray) and value.size == 1:
                        new_obs_dict[key] = value.item()  # Unpack single-value numpy arrays
                    else:
                        new_obs_dict[key] = value  # Keep original value

                # Compute action and add it to the action dictionary
                self.action_dict[agentid] = self.policy.compute_action({agentid : new_obs_dict})

            return [self.action_dict[agentid]], [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass
    return DefendPolicy

def CombinedGen(agentid, team, mode, team_size):
    class CombinedPolicy(Policy):
        """
        Creates a combined (attacker and defender) policy
        """
        policy = None
        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = Heuristic_CTF_Agent(agentid, team, mode=mode)
            self.action_dict = OrderedDict([(p, 16) for p in range(2*team_size)])
            self.normalizer = register_state_elements(team_size)

        def compute_actions(self,
                            obs_batch,
                            state_batches,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs): # Iterate over all observations in obs_batch
            for obs in obs_batch[0]:

                # Unnormalize and unpack numpy arrays to create a new dictionary
                new_obs_dict = OrderedDict()
                for key, value in self.normalizer.unnormalized(obs).items():
                    if isinstance(value, np.ndarray) and value.size == 1:
                        new_obs_dict[key] = value.item()  # Unpack single-value numpy arrays
                    else:
                        new_obs_dict[key] = value  # Keep original value

                # Compute action and add it to the action dictionary
                self.action_dict[agentid] = self.policy.compute_action({agentid : new_obs_dict})

            return [self.action_dict[agentid]], [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass
    return CombinedPolicy

# Function to create the normalizer for observations
def register_state_elements(team_size):
    """Initializes the normalizer."""
    agent_obs_normalizer = ObsNormalizer(True)
    max_bearing = [180]
    max_dist = [np.linalg.norm(pyq.config_dict_std["world_size"]) + 10]  # add a ten meter buffer
    min_dist = [0.0]
    max_bool, min_bool = [1.0], [0.0]
    max_speed, min_speed = [pyq.config_dict_std["max_speed"]], [0.0]
    agent_obs_normalizer.register("retrieve_flag_bearing", max_bearing)
    agent_obs_normalizer.register("retrieve_flag_distance", max_dist, min_dist)
    agent_obs_normalizer.register("protect_flag_bearing", max_bearing)
    agent_obs_normalizer.register("protect_flag_distance", max_dist, min_dist)
    agent_obs_normalizer.register("wall_0_bearing", max_bearing)
    agent_obs_normalizer.register("wall_0_distance", max_dist, min_dist)
    agent_obs_normalizer.register("wall_1_bearing", max_bearing)
    agent_obs_normalizer.register("wall_1_distance", max_dist, min_dist)
    agent_obs_normalizer.register("wall_2_bearing", max_bearing)
    agent_obs_normalizer.register("wall_2_distance", max_dist, min_dist)
    agent_obs_normalizer.register("wall_3_bearing", max_bearing)
    agent_obs_normalizer.register("wall_3_distance", max_dist, min_dist)
    agent_obs_normalizer.register("speed", max_speed, min_speed)
    agent_obs_normalizer.register("has_flag", max_bool, min_bool)
    agent_obs_normalizer.register("on_side", max_bool, min_bool)
    agent_obs_normalizer.register(
        "tagging_cooldown", [pyq.config_dict_std["tagging_cooldown"]], [0.0]
    )

    num_on_team = team_size

    for i in range(num_on_team - 1):
        teammate_name = f"teammate_{i}"
        agent_obs_normalizer.register((teammate_name, "bearing"), max_bearing)
        agent_obs_normalizer.register(
            (teammate_name, "distance"), max_dist, min_dist
        )
        agent_obs_normalizer.register(
            (teammate_name, "relative_heading"), max_bearing
        )
        agent_obs_normalizer.register(
            (teammate_name, "speed"), max_speed, min_speed
        )
        agent_obs_normalizer.register(
            (teammate_name, "has_flag"), max_bool, min_bool
        )
        agent_obs_normalizer.register(
            (teammate_name, "on_side"), max_bool, min_bool
        )
        agent_obs_normalizer.register(
            (teammate_name, "tagging_cooldown"), [pyq.config_dict_std["tagging_cooldown"]], [0.0]
        )

    for i in range(num_on_team):
        opponent_name = f"opponent_{i}"
        agent_obs_normalizer.register((opponent_name, "bearing"), max_bearing)
        agent_obs_normalizer.register(
            (opponent_name, "distance"), max_dist, min_dist
        )
        agent_obs_normalizer.register(
            (opponent_name, "relative_heading"), max_bearing
        )
        agent_obs_normalizer.register(
            (opponent_name, "speed"), max_speed, min_speed
        )
        agent_obs_normalizer.register(
            (opponent_name, "has_flag"), max_bool, min_bool
        )
        agent_obs_normalizer.register(
            (opponent_name, "on_side"), max_bool, min_bool
        )
        agent_obs_normalizer.register(
            (opponent_name, "tagging_cooldown"), [pyq.config_dict_std["tagging_cooldown"]], [0.0]
        )

    return agent_obs_normalizer
