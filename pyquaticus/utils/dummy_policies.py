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
from enum import Enum

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from ray.rllib.policy import Policy


class NoWeightsPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


class NoOpPolicy(NoWeightsPolicy):
    """Custom policy that takes a no-op action."""

    def __init__(self, observation_space, action_space, config):
        NoWeightsPolicy.__init__(self, observation_space, action_space, config)
        # assuming that zeros are a no-op
        dtype = None
        if isinstance(action_space, Box):
            dtype = np.float32
        elif isinstance(action_space, Discrete):
            dtype = np.int32
        else:
            assert isinstance(
                action_space, MultiDiscrete
            ), "Expecting a Box, Discrete, or MultiDiscrete"
            dtype = np.int32
        self.no_op_action = np.zeros(action_space.shape, dtype=dtype)

    def compute_action(self, obs):
        return self.no_op_action

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        return [self.no_op_action for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats


def get_patrol_policy(
    env, team, switch_prob=0.0, x_frac=0.8, y_low_frac=0.0, y_high_frac=1.0
):
    """
    Args:
        env: ctf_v0 environment
        team: which team (red or blue) is using this policy
        switch_prob: probability of switching direction
        x_frac: x location to go to as fraction of world size
        y_low_frac: low y location as a fraction of world size
        y_high_frac: high y location as a fraction of world size.

    Notes: default x_frac/y_low_frac/y_high_frac are for a patrolling red agent
    """
    assert (
        env.config_dict["action_space"] == "vel"
    ), "PatrolPolicy expects velocity control"

    class PatrolPolicy(NoWeightsPolicy):
        """
        This is a hacked-together Policy class to make
        a defending red agent patrol up and down in front of the flag.

        It is currently hardcoded for the specific environment.
        """

        class Mode(Enum):
            TO_X = 1
            UP = 2
            DOWN = 3

        # patrol location
        ideal_x = x_frac * env.world_size[0]
        y_lower = y_low_frac * env.world_size[1]
        y_upper = y_high_frac * env.world_size[1]

        def __init__(self, observation_space, action_space, config):
            NoWeightsPolicy.__init__(self, observation_space, action_space, config)
            self.state = self.Mode.TO_X
            assert len(env.agents_of_team[team]) == 1, "Expecting one agent"
            self.agent_idx = env.agents_of_team[team][0]

        def identify_mode(self, obs_dict):
            """
            Determine which mode the agent is in.

            Args:
                obs_dict: the unnormalized observation dictionary

            Returns
            -------
                mode: which direction should the agent head
            """
            pos = obs_dict[("agent_position", self.agent_idx)]
            vel = obs_dict[("agent_velocity", self.agent_idx)]
            if abs(pos[0] - self.ideal_x) >= 1:
                return self.Mode.TO_X

            mode = None
            if pos[1] >= self.y_upper:
                mode = self.Mode.DOWN
            elif pos[1] <= self.y_lower:
                mode = self.Mode.UP
            elif vel[1] < 0:
                mode = self.Mode.DOWN
            else:
                mode = self.Mode.UP

            if np.random.binomial(1, switch_prob):
                mode = self.Mode.UP if self.state == self.Mode.DOWN else self.state.DOWN

            return mode

        def compute_action(self, obs):
            """
            Compute an action for the given position.

            Args:
                obs: raw observations of red teams-- expecting a flattened tuple
                     where the tuple contains the regular observations and
                     the automata state

            Returns
            -------
                action: a red velocity action
            """
            # Handle both a tuple and a flattened tuple (one array)
            if isinstance(obs, tuple):
                assert (
                    len(obs) == 2
                ), "Expecting regular observations and automata state"
                regular_obs, _ = obs
            else:
                regular_obs_len = env.observation_space[0].shape[0]
                assert (
                    obs.shape[0] == regular_obs_len + env.observation_space[1].shape[0]
                )
                regular_obs = obs[:regular_obs_len]
            obs_dict = env.normalizer[team].unnormalized(regular_obs)
            pos_idx = ("agent_position", self.agent_idx)
            pos = obs_dict[pos_idx]
            mode = self.identify_mode(obs_dict)

            if mode == self.Mode.TO_X:
                if pos[0] < self.ideal_x:
                    action = [1.0, 0.0]  # east
                else:
                    action = [-1.0, 0.0]  # west
            elif mode == self.Mode.UP:
                action = [0.0, 1.0]  # north
            else:
                assert mode == self.Mode.DOWN
                action = [0.0, -1.0]  # south

            return np.asarray(action, dtype=np.float32)

        def compute_actions(
            self,
            obs_batch,
            state_batches,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            **kwargs,
        ):
            return [self.compute_action(obs) for obs in obs_batch], [], {}

    return PatrolPolicy


def get_tagger_policy(
    env, team, enemy_team, x_frac=0.8, y_frac=0.5, scrimmage_offset=0.0
):
    """
    Args:
        env: ctf_v0 environment
        team: which team (red or blue) is using this policy
        enemy_team: which team (red or blue) is the opposing team
        x_frac: x location to go to as fraction of world size (To change mid point)
        y_frac: y location as a fraction of world size (To change mid point)
        scrimmage_offset: Modify the line that blue has to cross to make red agent chase.

    Notes: default x_frac/y_frac/scrimmage_offset are for a tagging ,red agent
    """
    assert (
        env.config_dict["action_space"] == "vel"
    ), "TaggerPolicy expects velocity control"

    class TaggerPolicy(NoWeightsPolicy):
        """
        This is a hacked-together Policy class to make
        a defending red agent try to tag the blue agent in his own half of the field or
        based on scrimmage offset.

        It is currently hardcoded for the specific environment.
        """

        class Mode(Enum):
            TO_OPP = 1
            TO_MID = 2

        # Give more room to blue agents before being chased
        scrimmage = env.scrimmage * (1 + scrimmage_offset)
        # x_mid is set closer to flag to leave room for opponent to attack
        x_mid = x_frac * env.world_size[0]
        y_mid = y_frac * env.world_size[1]

        def __init__(self, observation_space, action_space, config):
            NoWeightsPolicy.__init__(self, observation_space, action_space, config)
            # Start by going to MID first
            self.state = self.Mode.TO_MID
            assert len(env.agents_of_team[team]) == 1, "Expecting one agent"
            self.agent_idx = env.agents_of_team[team][0]
            # Enemy agents can be multiple
            self.enemy_agent_idx = env.agents_of_team[enemy_team]

        def identify_mode(self, obs_dict):
            """
            Determine which mode the agent is in.

            Args:
                obs_dict: the unnormalized observation dictionary

            Returns
            -------
                mode: which direction should the agent head
            """
            # Mode depends on enemy location
            for idx in self.enemy_agent_idx:
                pos = obs_dict[("agent_position", idx)]

                # Enemy is beyond modified scrimmage line
                if pos[0] > self.scrimmage:
                    return self.Mode.TO_OPP

            return self.Mode.TO_MID

        def compute_action(self, obs):
            """
            Compute an action for the given position. This function uses observations
            of both teams.

            Args:
                obs: Dictionary of raw observations for teams -- containing the regular
                     observations and the automata state

            Returns
            -------
                action: a red velocity action
            """
            # handle both a tuple and a flattened tuple (one array)
            if isinstance(obs, tuple):
                assert (
                    len(obs) == 2
                ), "Expecting regular observations and automata state"
                regular_obs, _ = obs
            else:
                regular_obs_len = env.observation_space[0].shape[0]
                assert (
                    obs.shape[0] == regular_obs_len + env.observation_space[1].shape[0]
                )
                regular_obs = obs[:regular_obs_len]

            # Tagging team's obs
            obs_dict = env.normalizer[team].unnormalized(regular_obs)
            pos_idx = ("agent_position", self.agent_idx)
            pos = obs_dict[pos_idx]

            # Enemy team obs
            en_ag_pos = {}
            for idx in self.enemy_agent_idx:
                pos_en_idx = ("agent_dist", self.agent_idx, idx)
                en_ag_pos[(pos_idx[0], idx)] = obs_dict[pos_idx] + obs_dict[pos_en_idx]

            mode = self.identify_mode(en_ag_pos)

            # Find closest enemy agent
            if mode == self.Mode.TO_MID:
                mid = [self.x_mid, self.y_mid]
                action = (mid - pos) / env.get_distance_between_2_points(mid, pos)
            elif mode == self.Mode.TO_OPP:
                closest_agent = min(
                    en_ag_pos.values(),
                    key=lambda x: env.get_distance_between_2_points(pos, x),
                )
                action = (closest_agent - pos) / env.get_distance_between_2_points(
                    closest_agent, pos
                )

            return np.asarray(action, dtype=np.float32)

        def compute_actions(
            self,
            obs_batch,
            state_batches,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            **kwargs,
        ):
            return [self.compute_action(obs) for obs in obs_batch], [], {}

    return TaggerPolicy
