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

from pyquaticus.base_policies.base import BaseAgentPolicy
from pyquaticus.envs.pyquaticus import config_dict_std, Team

modes = {"nothing", "easy", "medium", "hard"}


class BaseDefenderContinuous(BaseAgentPolicy):
    """This is a Policy class that contains logic for defending the flag."""

    def __init__(
        self,
        agent_id: int,
        team: Team,
        mode: str = "easy",
        flag_keepout=10.0,
        catch_radius=config_dict_std["catch_radius"],
        using_pyquaticus=True,
    ):
        super().__init__(agent_id, team)

        if mode not in modes:
            raise ValueError(f"mode {mode} not in set of valid modes {modes}")
        self.mode = mode
        self.flag_keepout = flag_keepout
        self.catch_radius = catch_radius
        self.using_pyquaticus = using_pyquaticus
        self.goal = "PM"

    def set_mode(self, mode: str):
        """
        Determine which mode the agent is in:
        'easy' = Easy Attacker
        'medium' = Medium Attacker
        'hard' = Hard Attacker.
        """
        if mode not in modes:
            raise ValueError(f"mode {mode} not in set of valid modes {modes}")
        self.mode = mode

    def get_distance_between_2_points(self, start: np.array, end: np.array) -> float:
        """
        Convenience method for returning distance between two points.

        Args:
            start: Starting position to measure from
            end: Point to measure to
        Returns:
            The distance between `start` and `end`
        """
        return np.linalg.norm(np.asarray(start) - np.asarray(end))

    def compute_action(self, obs):
        """
        **THIS FUNCTION REQUIRES UNNORMALIZED OBSERVATIONS**.

        Compute an action for the given position. This function uses observations
        of both teams.

        Args:
            obs: Unnormalized observation from the gym

        Returns
        -------
            desired_speed: m/s
            heading_error: deg
        """
        desired_speed = 50

        my_obs = self.update_state(obs)

        if self.mode == "easy":

            # NOTE: This logic causes the agent to drive towards a captured flag, but it won't ever attempt to tag
            # the agent carrying it

            ag_vect = [0, 0]
            my_flag_vec = self.bearing_to_vec(self.my_flag_bearing)

            # If far away from the flag, move towards it
            if self.my_flag_distance > (self.flag_keepout + self.catch_radius + 1.0):
                ag_vect = my_flag_vec

            # If too close to the flag, move away
            else:
                ag_vect = np.multiply(-1.0, my_flag_vec)

            # Convert the vector to a heading, and then pick the best discrete action to perform
            try:
                heading_error = self.angle180(self.vec_to_heading(ag_vect))

                if np.isnan(heading_error):
                    heading_error = 0

                if np.abs(heading_error) < 5:
                    heading_error = 0

            except:
                # If there is an error converting the vector to a heading, just go straight
                heading_error = 0

        elif self.mode == "nothing":

            desired_speed = 0
            heading_error = 0

        elif self.mode == "medium":

            my_flag_vec = self.bearing_to_vec(self.my_flag_bearing)

            # If the blue team has the flag, chase them
            if self.opp_team_has_flag:
                ag_vect = my_flag_vec

            # If the blue team doesn't have the flag, guard it
            else:
                flag_dist = self.my_flag_distance

                if flag_dist > (self.flag_keepout + self.catch_radius + 1.0):
                    ag_vect = my_flag_vec

                else:
                    ag_vect = np.multiply(-1.0, my_flag_vec)

            # Convert the vector to a heading, and then pick the best discrete action to perform
            try:
                heading_error = self.angle180(self.vec_to_heading(ag_vect))

                if np.isnan(heading_error):
                    heading_error = 0

                if np.abs(heading_error) < 5:
                    heading_error = 0

            except:
                # If there is an error converting the vector to a heading, just go straight
                heading_error = 0

        elif self.mode == "hard":

            # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
            wall_pos = []

            if my_obs["wall_0_distance"] < 7 and (-90 < my_obs["wall_0_bearing"] < 90):
                wall_0_unit_vec = self.rb_to_rect(
                    (my_obs["wall_0_distance"], my_obs["wall_0_bearing"])
                )
                wall_pos.append(
                    (
                        my_obs["wall_0_distance"] * wall_0_unit_vec[0],
                        my_obs["wall_0_distance"] * wall_0_unit_vec[1],
                    )
                )

            elif my_obs["wall_2_distance"] < 7 and (
                -90 < my_obs["wall_2_bearing"] < 90
            ):
                wall_2_unit_vec = self.rb_to_rect(
                    (my_obs["wall_2_distance"], my_obs["wall_2_bearing"])
                )
                wall_pos.append(
                    (
                        my_obs["wall_2_distance"] * wall_2_unit_vec[0],
                        my_obs["wall_2_distance"] * wall_2_unit_vec[1],
                    )
                )
            if my_obs["wall_1_distance"] < 7 and (-90 < my_obs["wall_1_bearing"] < 90):
                wall_1_unit_vec = self.rb_to_rect(
                    (my_obs["wall_1_distance"], my_obs["wall_1_bearing"])
                )
                wall_pos.append(
                    (
                        my_obs["wall_1_distance"] * wall_1_unit_vec[0],
                        my_obs["wall_1_distance"] * wall_1_unit_vec[1],
                    )
                )
            elif my_obs["wall_3_distance"] < 7 and (
                -90 < my_obs["wall_3_bearing"] < 90
            ):
                wall_3_unit_vec = self.rb_to_rect(
                    (my_obs["wall_3_distance"], my_obs["wall_3_bearing"])
                )
                wall_pos.append(
                    (
                        my_obs["wall_3_distance"] * wall_3_unit_vec[0],
                        my_obs["wall_3_distance"] * wall_3_unit_vec[1],
                    )
                )

            ag_vect = [0, 0]

            defense_perim = 5 * self.flag_keepout

            # Get nearest untagged enemy:
            # NOTE: Is this sufficient for larger (gps) environments?
            min_enemy_distance = 1000.00 
            enemy_dis_dict = {}
            closest_enemy = None
            for enem, pos in self.opp_team_pos_dict.items():
                enemy_dis_dict[enem] = pos[0]
                if pos[0] < min_enemy_distance and not my_obs[(enem, "is_tagged")]:
                    min_enemy_distance = pos[0]
                    closest_enemy = enem
                    enemy_loc = self.rb_to_rect(pos)

            if closest_enemy == None:
                min_enemy_distance = min(enemy_dis_dict.values())
                closest_enemy = min(enemy_dis_dict, key=enemy_dis_dict.get)
                enemy_loc = self.rb_to_rect(self.opp_team_pos_dict[closest_enemy])

            if not self.opp_team_has_flag:
                defend_pt = self.closest_point_on_line(
                    self.my_flag_loc, enemy_loc, [0, 0]
                )
                defend_pt_flag_dist = self.get_distance_between_2_points(
                    defend_pt, self.my_flag_loc
                )
                unit_def_flag = self.unit_vect_between_points(
                    defend_pt, self.my_flag_loc
                )
                enemy_dist_2_flag = self.get_distance_between_2_points(
                    enemy_loc, self.my_flag_loc
                )

                if (
                    enemy_dist_2_flag > defense_perim
                    or my_obs[(closest_enemy, "is_tagged")]
                ):
                    if (
                        defend_pt_flag_dist > defense_perim
                        or my_obs[(closest_enemy, "is_tagged")]
                    ):
                        guide_pt = [
                            self.my_flag_loc[0] + (unit_def_flag[0] * defense_perim),
                            self.my_flag_loc[1] + (unit_def_flag[1] * defense_perim),
                        ]
                    else:
                        guide_pt = defend_pt
                else:
                    guide_pt = enemy_loc

                ag_vect = guide_pt
            else:
                ag_vect = self.bearing_to_vec(self.my_flag_bearing)

            if wall_pos is not None:
                ag_vect = ag_vect + self.get_avoid_vect(wall_pos)

            # Convert the vector to a heading
            try:
                heading_error = self.angle180(self.vec_to_heading(ag_vect))

                if np.isnan(heading_error):
                    heading_error = 0

                if np.abs(heading_error) < 5:
                    heading_error = 0

            # If there is an error converting the vector to a heading, just go straight
            except:
                heading_error = 0

        return desired_speed, heading_error
