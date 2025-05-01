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

from typing import Union

import numpy as np

import pyquaticus.base_policies.base_attack as attack_policy
import pyquaticus.base_policies.base_defend as defend_policy
from pyquaticus.base_policies.base import BaseAgentPolicy
from pyquaticus.base_policies.utils import (
    dist_rel_bearing_to_local_rect,
    get_avoid_vect,
    global_rect_to_abs_bearing,
    local_rect_to_rel_bearing,
    rel_bearing_to_local_unit_rect,
)
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import angle180, closest_point_on_line, dist

MODES = {"easy", "medium", "hard", "nothing"}


class Heuristic_CTF_Agent(BaseAgentPolicy):
    """Combined attack and defense policy accounting for enemy positions before deciding to attack/defend."""

    def __init__(
        self,
        agent_id: str,
        team: Team,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        continuous: bool = False,
        mode="easy",
        defensiveness=20.0,
    ):
        super().__init__(agent_id, team, env)
        self.state_normalizer = env.global_state_normalizer
        self.walls = env._walls[team.value]
        self.max_speed = env.players[self.id].get_max_speed()
        self.set_mode(mode)
        self.defensiveness = defensiveness
        self.continuous = continuous
        self.flag_keepout = getattr(env, "flag_keepout_radius", 3)
        self.base_attacker = attack_policy.BaseAttacker(
            self.id,
            team,
            env,
            continuous,
            mode,
        )
        self.base_defender = defend_policy.BaseDefender(
            self.id,
            team,
            env,
            continuous,
            mode,
        )
        self.scrimmage = None

    def set_mode(self, mode: str):
        """Sets difficulty mode."""
        if mode not in MODES:
            raise ValueError(f"mode {mode} not in set of valid modes: {MODES}")
        self.mode = mode

    def compute_action(self, obs, info):
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and heading error.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """
        # Update the state based on this observation
        self.update_state(obs, info)

        if self.mode == "nothing":
            return self.action_from_vector(None, 0)

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
            near_enemy_dist = np.inf
            nearest_enemy = None
            for en in enem_positions:
                temp_enem_dist = en[0]
                if temp_enem_dist < near_enemy_dist:
                    near_enemy_dist = temp_enem_dist
                    nearest_enemy = en
            assert nearest_enemy is not None
            if np.random.random() < 0.5:
                goal_vec = rel_bearing_to_local_unit_rect(nearest_enemy[1])
            else:
                own_flag_dist = self.my_flag_distance
                if own_flag_dist > self.flag_keepout + 2.0:
                    goal_vec = rel_bearing_to_local_unit_rect(self.my_flag_bearing)
                else:
                    span_len = self.scrimmage - self.defensiveness
                    goal_vec = [np.random.random() * span_len, 0]

        if not self.on_sides:
            goal_vec = goal_vec + get_avoid_vect(self.opp_team_pos, avoid_threshold=15)

        if self.mode == "hard":
            return self.action_from_vector(goal_vec, 1)
        else:
            return self.action_from_vector(goal_vec, 0.5)

    def get_team_density(self, friendly_positions, enemy_positions):
        """This function returns the center of mass and varience of all the agents in the team."""
        home_x = []
        home_y = []
        away_x = []
        away_y = []

        # TO DO: THIS NEEDS TO BE BY X and Y seperately
        for agp in friendly_positions:
            ag = dist_rel_bearing_to_local_rect(*agp)
            home_x.append(ag[0])
            home_y.append(ag[1])
        for agp in enemy_positions:
            ag = dist_rel_bearing_to_local_rect(*agp)
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

        dist_to_flag = dist(self.opp_team_density[0], np.array(self.my_flag_loc))
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

        dist_to_flag = dist(self.opp_team_density[0], np.array(self.my_flag_loc))
        return dist_to_flag > threshold

    def update_state(self, obs, info: dict[str, dict]) -> None:
        """
        Method to convert the gym obs and info into data more relative to the
        agent.

        Note: all rectangular positions are in the ego agent's local coordinate frame.
        Note: all bearings are relative, measured in degrees clockwise from the ego agent's heading.

        Args:
            obs: observation from gym
            info: info from gym
        """

        global_state = info[self.id]["global_state"]
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)
        # print(global_state)

        # Copy this agents state from the global state
        self.speed = global_state[(self.id, "speed")]
        self.on_sides = global_state[(self.id, "on_side")]
        self.has_flag = global_state[(self.id, "has_flag")]
        self.tagging_cooldown = global_state[(self.id, "tagging_cooldown")]
        self.is_tagged = global_state[(self.id, "is_tagged")]

        # Calculate the rectangular coordinates for the flags location relative to the agent.
        team_str = self.team.name.lower().split("_")[0]
        opp_str = "red" if team_str == "blue" else "blue"

        self.my_flag_distance = dist(
            global_state[(self.id, "pos")], global_state[team_str + "_flag_pos"]
        )
        self.my_flag_bearing = angle180(
            global_rect_to_abs_bearing(
                global_state[team_str + "_flag_pos"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.my_flag_loc = dist_rel_bearing_to_local_rect(
            self.my_flag_distance, self.my_flag_bearing
        )
        self.opp_flag_distance = dist(
            global_state[(self.id, "pos")], global_state[opp_str + "_flag_pos"]
        )
        self.opp_flag_bearing = angle180(
            global_rect_to_abs_bearing(
                global_state[opp_str + "_flag_pos"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.opp_flag_loc = dist_rel_bearing_to_local_rect(
            self.opp_flag_distance, self.opp_flag_bearing
        )

        self.home_distance = dist(
            global_state[(self.id, "pos")], global_state[team_str + "_flag_home"]
        )
        self.home_bearing = angle180(
            global_rect_to_abs_bearing(
                global_state[team_str + "_flag_home"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.home_loc = dist_rel_bearing_to_local_rect(
            self.home_distance, self.home_bearing
        )

        # Copy the polar positions of each agent, separated by team and get their tag status
        self.opp_team_pos = []
        self.opp_team_pos_dict = {}  # for labeling by agent_id
        self.my_team_pos = []
        self.opp_team_tag = []
        self.my_team_tag = []
        self.opp_team_has_flag = False
        self.my_team_has_flag = False
        for id in self.teammate_ids:
            if id != self.id:
                distance = dist(
                    global_state[(self.id, "pos")], global_state[(id, "pos")]
                )
                bearing = angle180(
                    global_rect_to_abs_bearing(
                        global_state[(id, "pos")] - global_state[(self.id, "pos")]
                    )
                    - global_state[(self.id, "heading")]
                )
                self.my_team_pos.append(np.array((distance, bearing)))
                self.my_team_has_flag = (
                    self.my_team_has_flag or global_state[(id, "has_flag")]
                )
                self.my_team_tag.append(global_state[(id, "is_tagged")])
        for id in self.opponent_ids:
            distance = dist(global_state[(self.id, "pos")], global_state[(id, "pos")])
            bearing = angle180(
                global_rect_to_abs_bearing(
                    global_state[(id, "pos")] - global_state[(self.id, "pos")]
                )
                - global_state[(self.id, "heading")]
            )
            self.opp_team_pos.append(np.array((distance, bearing)))
            self.opp_team_has_flag = (
                self.opp_team_has_flag or global_state[(id, "has_flag")]
            )
            self.opp_team_tag.append(global_state[(id, "is_tagged")])
            self.opp_team_pos_dict[id] = np.array((distance, bearing))

        self.wall_distances = []
        self.wall_bearings = []
        for wall in self.walls:
            closest = closest_point_on_line(
                wall[0], wall[1], global_state[(self.id, "pos")]
            )
            self.wall_distances.append(dist(closest, global_state[(self.id, "pos")]))
            bearing = angle180(
                global_rect_to_abs_bearing(closest - global_state[(self.id, "pos")])
                - global_state[(self.id, "heading")]
            )
            self.wall_bearings.append(bearing)

        # Initialize the scrimmage line as the mid point between the two flags
        if self.scrimmage is None:
            self.scrimmage = self.opp_flag_loc[0] + self.my_flag_loc[0] / 2

        self.my_team_density, self.opp_team_density = self.get_team_density(
            self.my_team_pos, self.opp_team_pos
        )

    def action_from_vector(self, vector, desired_speed_normalized):
        if desired_speed_normalized == 0:
            if self.continuous:
                return (0, 0)
            else:
                return -1
        rel_bearing = local_rect_to_rel_bearing(vector)
        if self.continuous:
            return (desired_speed_normalized * self.max_speed, rel_bearing)
        elif desired_speed_normalized == 0.5:
            if 1 >= rel_bearing >= -1:
                return 12
            elif rel_bearing < -1:
                return 14
            elif rel_bearing > 1:
                return 10
        elif desired_speed_normalized == 1:
            if 1 >= rel_bearing >= -1:
                return 4
            elif rel_bearing < -1:
                return 6
            elif rel_bearing > 1:
                return 2
