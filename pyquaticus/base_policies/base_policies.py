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


def AttackGen(agentid, team, teammate_ids, opponent_ids, obs_normalizer, state_normalizer, mode, continuous, team_size):

    class AttackPolicy(Policy):
        """
        Creates an attacker policy
        """

        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseAttacker(agentid, team, teammate_ids, opponent_ids, obs_normalizer, state_normalizer, mode=mode, continuous=continuous)
            self.action_dict = {}

        def compute_actions(self,
                            obs_batch,
                            state_batches,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs):

            if info_batch is None:
                raise Warning("Warning: Base policy requires info as well as obs")

            # Iterate over all observations in obs_batch
            for i in range(len(obs_batch)):

                # Compute action and add it to the action dictionary
                self.action_dict[agentid] = self.policy.compute_action(obs_batch[i], info_batch[i])

            return [self.action_dict[agentid]], [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass

    return AttackPolicy


def DefendGen(agentid, team, teammate_ids, opponent_ids, obs_normalizer, state_normalizer, mode, continuous, team_size):
    class DefendPolicy(Policy):
        """
        Creates a defender policy
        """

        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = BaseDefender(agentid, team, teammate_ids, opponent_ids, obs_normalizer, state_normalizer, mode=mode, continuous=continuous)
            self.action_dict = {}

        def compute_actions(self,
                            obs_batch,
                            state_batches,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs):

            if info_batch is None:
                raise Warning("Warning: Base policy requires info as well as obs")

            # Iterate over all observations in obs_batch
            for i in range(len(obs_batch)):

                # Compute action and add it to the action dictionary
                self.action_dict[agentid] = self.policy.compute_action(obs_batch[i], info_batch[i])

            return [self.action_dict[agentid]], [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass
    return DefendPolicy


def CombinedGen(agentid, team, teammate_ids, opponent_ids, obs_normalizer, state_normalizer, mode, continuous, team_size):
    class CombinedPolicy(Policy):
        """
        Creates a combined (attacker and defender) policy
        """

        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)
            self.policy = Heuristic_CTF_Agent(agentid, team, teammate_ids, opponent_ids, obs_normalizer, state_normalizer, mode=mode, continuous=continuous)
            self.action_dict = {}

        def compute_actions(self,
                            obs_batch,
                            state_batches,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            info_batch=None,
                            episodes=None,
                            **kwargs):  # Iterate over all observations in obs_batch
            if info_batch is None:
                raise Warning("Warning: Base policy requires info as well as obs")

            # Iterate over all observations in obs_batch
            for i in range(len(obs_batch)):

                # Compute action and add it to the action dictionary
                self.action_dict[agentid] = self.policy.compute_action(obs_batch[i], info_batch[i])

            return [self.action_dict[agentid]], [], {}

        def get_weights(self):
            return {}

        def learn_on_batch(self, samples):
            return {}

        def set_weights(self, weights):
            pass
    return CombinedPolicy
