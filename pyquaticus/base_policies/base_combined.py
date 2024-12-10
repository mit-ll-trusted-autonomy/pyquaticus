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

import numpy as np

import pyquaticus.base_policies.base_attack as attack_policy
import pyquaticus.base_policies.base_defend as defend_policy
from pyquaticus.base_policies.base import BaseAgentPolicy
from pyquaticus.envs.pyquaticus import config_dict_std, Team
from pyquaticus.utils.obs_utils import ObsNormalizer

from typing import Union

modes = {"easy", "medium", "hard", "nothing"}


class Heuristic_CTF_Agent(BaseAgentPolicy):
    """Combined attack and defense policy accounting for enemy positions before deciding to attack/defend."""

    def __init__(
        self,
        agent_id: int,
        team: Team,
        teammate_ids: Union[list[int], int, None],
        opponent_ids: Union[list[int], int, None],
        obs_normalizer: ObsNormalizer,
        state_normalizer: ObsNormalizer,
        mode="easy",
        continuous: bool = False,
        flag_keepout=10.0,
        catch_radius=config_dict_std["catch_radius"],
        using_pyquaticus=True,
        defensiveness=20.0,
    ):
        super().__init__(agent_id, team, teammate_ids, opponent_ids, obs_normalizer, state_normalizer)

        if mode not in modes:
            raise ValueError(f"mode {mode} not a valid mode out of {modes}")

        if mode not in modes:
            raise ValueError(f"Invalid mode {mode}, valid modes are {modes}")
        self.mode = mode
        self.flag_keepout = flag_keepout
        self.defensiveness = defensiveness
        self.using_pyquaticus = using_pyquaticus
        self.id = agent_id
        self.continuous = continuous
        self.base_attacker = attack_policy.BaseAttacker(
            self.id,
            team,
            teammate_ids,
            opponent_ids,
            obs_normalizer,
            state_normalizer,
            mode,
            continuous,
            using_pyquaticus,
        )
        self.base_defender = defend_policy.BaseDefender(
            self.id,
            team,
            teammate_ids,
            opponent_ids,
            obs_normalizer,
            state_normalizer,
            mode,
            continuous,
            flag_keepout,
            catch_radius,
            using_pyquaticus,
        )
        self.scrimmage = None

        # My state
        self.my_pos = None
        self.my_hdg = None
        self.has_flag = False
        self.on_sides = False

        # State of my team
        self.friendly_team_pos = None
        self.friendly_team_density = None
        self.my_team_has_flag = False

        # State of the opp team
        self.opp_team_pos = None
        self.opp_team_density = None
        self.opp_team_has_flag = False

    def set_mode(self, mode="easy"):
        """
        Determine which mode the agent is in:
        'easy' = Easy Attacker
        'medium' = Medium Attacker
        'hard' = Hard Attacker.
        """
        if mode not in modes:
            raise ValueError(f"Invalid mode {mode}")
        self.mode = mode

    def compute_action(self, obs, info):
        """
        **THIS FUNCTION REQUIRES UNNORMALIZED GLOBAL STATE**.

        Upates the state of the agent using the observation `obs` and then computes
        an action to take based on the new state.

        Args:
            obs: Dictionary of raw observations for teams -- containing the regular
                    observations

        Returns
        -------
            action: A discrete action index representing the agents choice.

        """
        # Update the state based on this observation
        self.update_state(obs, info)

        if self.mode == "nothing":
            if self.continuous:
                return (0, 0)
            else:
                return -1

        if self.mode == "easy":
            # Opp is close - needs to defend:
            if self.is_close_to_flag() and False in self.opp_team_tag:
                return self.base_defender.compute_action(obs, info)

            # Opp on defensive - needs to attack
            else:
                return self.base_attacker.compute_action(obs, info)

        else:
            # If I have the flag, just bring it back to base
            if self.has_flag:
                return self.base_attacker.compute_action(obs, info)

            elif self.opp_team_has_flag:
                return self.base_defender.compute_action(obs, info)

            # Opp is close - go on defensive
            elif self.is_close_to_flag() and (False in self.opp_team_tag):
                return self.base_defender.compute_action(obs, info)

            # Opp on defensive - needs to attack
            elif self.is_far_from_flag():
                return self.base_attacker.compute_action(obs, info)

            else:
                if self.mode == "hard":
                    return self.base_attacker.compute_action(obs, info)
                else:
                    return self.random_defense_action(self.opp_team_pos)

    def update_state(self, obs, info):
        """
        Method to convert the observation space into one more relative to the
        agent.

        Args:
            obs: The observation from the gym

        """
        super().update_state(obs, info)

        # Initialize the scrimmage line as the mid point between the two flags
        if self.scrimmage is None:
            self.scrimmage = self.opp_flag_loc[0] + self.my_flag_loc[0] / 2

        self.my_team_density, self.opp_team_density = self.get_team_density(
            self.my_team_pos, self.opp_team_pos
        )

    def random_defense_action(self, enem_positions):
        """
        Randomly compute an action that steers the agent to it's own side of the field and sometimes
        towards its flag.
        """
        if self.scrimmage is None:
            raise RuntimeWarning(
                "Must call update_state() before trying to get an action."
            )

        if np.random.random() < 0.25:
            span_len = self.scrimmage
            goal_vec = [np.random.random() * span_len, 0]
        else:
            near_enemy_dist = 1000
            for en in enem_positions:
                temp_enem_dist = en[0]
                if temp_enem_dist < near_enemy_dist:
                    near_enemy_dist = temp_enem_dist
                    nearest_enemy = en
            if np.random.random() < 0.5:
                goal_vec = self.bearing_to_vec(nearest_enemy[1])
            else:
                own_flag_dist = self.my_flag_distance
                if own_flag_dist > self.flag_keepout + 2.0:
                    goal_vec = self.bearing_to_vec(self.my_flag_bearing)
                else:
                    span_len = self.scrimmage - self.defensiveness
                    goal_vec = [np.random.random() * span_len, 0]

        if not self.on_sides:
            direction = goal_vec + self.get_avoid_vect(
                self.opp_team_pos, avoid_threshold=15
            )

        # TODO: Fix this
        # Some big speed hard-coded so that every agent drives at max speed
        desired_speed = 50

        try:
            heading_error = self.angle180(self.vec_to_heading(direction))

            if self.continuous:
                if np.isnan(heading_error):
                    heading_error = 0

                if np.abs(heading_error) < 5:
                    heading_error = 0

                return (desired_speed, heading_error)

            else:
                if self.mode != "hard":
                    if 1 >= heading_error >= -1:
                        return 12
                    elif heading_error < -1:
                        return 14
                    elif heading_error > 1:
                        return 10
                else:
                    # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
                    if 1 >= heading_error >= -1:
                        return 4
                    elif heading_error < -1:
                        return 6
                    elif heading_error > 1:
                        return 2
        except Exception:
            # Drive straights
            if self.continuous:
                return (desired_speed, 0)
            else:
                return 4

    def get_team_density(self, friendly_positions, enemy_positions):
        """This function returns the center of mass and varience of all the agents in the team."""
        home_x = []
        home_y = []
        away_x = []
        away_y = []

        # TO DO: THIS NEEDS TO BE BY X and Y seperately
        for agp in friendly_positions:
            ag = self.rb_to_rect(agp)
            home_x.append(ag[0])
            home_y.append(ag[1])
        for agp in enemy_positions:
            ag = self.rb_to_rect(agp)
            away_x.append(ag[0])
            away_y.append(ag[1])

        home_mean = np.array([np.mean(home_x), np.mean(home_y)])
        home_std = np.mean(np.array([np.std(home_x), np.std(home_y)]))

        away_mean = np.array([np.mean(away_x), np.mean(away_y)])
        away_std = np.mean(np.array([np.std(away_x), np.std(away_y)]))

        return [home_mean, home_std], [away_mean, away_std]

    def is_close_to_flag(self, threshold=30):
        """
        Checks how close the opposing teams position is relative to the flag.

        Args:
            threshold: The threshold distance to use for comparison

        Returns
        -------
            True if the center of mass of the opposing team is within
            ``threshold`` units of the flag, False otherwise
        """
        if self.opp_team_density is None:
            raise RuntimeWarning(
                "Must call update_state() before trying to get an action."
            )

        dist_to_flag = self.get_distance_between_2_points(
            self.opp_team_density[0], self.my_flag_loc
        )
        return dist_to_flag < threshold

    def is_far_from_flag(self, threshold=50):
        """
        Checks how far the opposing teams position is relative to the flag.

        Args:
            threshold: The threshold distance to use for comparison

        Returns
        -------
            True if the center of mass of the opposing team is further than
            ``threshold`` units of the flag, False otherwise
        """

        if self.opp_team_density is None:
            raise RuntimeWarning(
                "Must call update_state() before trying to get an action."
            )

        dist_to_flag = self.get_distance_between_2_points(
            self.opp_team_density[0], self.my_flag_loc
        )
        return dist_to_flag > threshold
