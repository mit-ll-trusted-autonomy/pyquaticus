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

modes = {"easy", "medium", "hard", "competition_nothing", "competition_easy"}


class BaseDefender(BaseAgentPolicy):
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
        self.goal = []
        self.competition_easy_1 = [135, 115]
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

    def compute_action(self, obs):
        """
        **THIS FUNCTION REQUIRES UNNORMALIZED OBSERVATIONS**.

        Compute an action for the given position. This function uses observations
        of both teams.

        Args:
            obs: Unnormalized observation from the gym

        Returns
        -------
            action: The action index describing which speed/heading combo to use (assumes
            discrete action values from `ctf-gym.envs.pyquaticus.ACTION_MAP`)
        """
        my_obs = self.update_state(obs)

        if self.mode == "easy":
            ag_vect = [0, 0]
            my_flag_vec = self.bearing_to_vec(self.my_flag_bearing)
            
           
            # If far away from the flag, move towards it
            if self.my_flag_distance > (
                self.flag_keepout + self.catch_radius + 1.0
            ):
                ag_vect = my_flag_vec

            # If too close to the flag, move away
            else:
                ag_vect = np.multiply(-1.0, my_flag_vec)

            
            act_index = 12
            act_heading = self.angle180(self.vec_to_heading(ag_vect))

            if 1 >= act_heading >= -1:
                act_index = 12
            elif act_heading < -1:
                act_index = 14
            elif act_heading > 1:
                act_index = 10
        
        elif self.mode=="competition_nothing":
            #If tagged return to untag
            if self.is_tagged:
                ag_vect = self.bearing_to_vec(my_obs["own_home_bearing"])

            act_index = -1
        elif self.mode=="competition_easy":
            coords = [self.my_flag_loc]
            #If tagged return to untagS
            ag_vect = -1
            if self.is_tagged:
                ag_vect = self.bearing_to_vec(my_obs["own_home_bearing"])
                self.competition_easy_1 = [135, 115]
            else:
                if self.team == Team.RED_TEAM:
                    if self.competition_easy_1[1] <= my_obs["wall_1_distance"] <= self.competition_easy_1[0]: #If near halfline start protective behavior
                        self.competition_easy_1 = [115, 95]
                        if self.goal == []:
                            self.goal = "wall_2"
                        if my_obs[self.goal + "_distance"] < 15:
                            if self.goal =="wall_0":
                                ag_vect = self.bearing_to_vec(my_obs["wall_3_bearing"])
                                self.goal = "wall_2"
                            else:
                                ag_vect = self.bearing_to_vec(my_obs["wall_1_bearing"])
                                self.goal = "wall_0"
                        else:
                            ag_vect = self.bearing_to_vec(my_obs[self.goal+"_bearing"])

                    else:
                        ag_vect = self.bearing_to_vec(my_obs["wall_1_bearing"])
                else:# Blue Team Wall Order is Different
                    if self.competition_easy_1[1] <= my_obs["wall_3_distance"] <= self.competition_easy_1[0]: #If near halfline start protective behavior
                        self.competition_easy_1 = [90, 55]
                        if self.goal == []:
                            self.goal = "wall_2"
                        if my_obs[self.goal + "_distance"] < 15:
                            if self.goal =="wall_0":
                                ag_vect = self.bearing_to_vec(my_obs["wall_1_bearing"])
                                self.goal = "wall_2"
                            else:
                                ag_vect = self.bearing_to_vec(my_obs["wall_3_bearing"])
                                self.goal = "wall_0"
                        else:
                            ag_vect = self.bearing_to_vec(my_obs[self.goal+"_bearing"])

                    else:
                        ag_vect = self.bearing_to_vec(my_obs["wall_3_bearing"])

            # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
            try:
                act_heading = self.angle180(self.vec_to_heading(ag_vect))
                if 1 >= act_heading >= -1:
                    act_index = 4
                elif act_heading < -1:
                    act_index = 6
                elif act_heading > 1:
                    act_index = 2
            except:
                act_index = -1

        elif self.mode == "medium":
            my_flag_vec = self.bearing_to_vec(self.my_flag_bearing)
    
            # If the blue team doesn't have the flag, guard it
            if self.opp_team_has_flag:
                # If the blue team has the flag, chase them
                ag_vect = my_flag_vec

            else:
                flag_dist = self.my_flag_distance

                if flag_dist > (self.flag_keepout + self.catch_radius + 1.0):
                    ag_vect = my_flag_vec

                else:
                    ag_vect = np.multiply(-1.0, my_flag_vec)

            act_index = 12
            act_heading = self.angle180(self.vec_to_heading(ag_vect))

            if 1 >= act_heading >= -1:
                act_index = 12
            elif act_heading < -1:
                act_index = 14
            elif act_heading > 1:
                act_index = 10

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
            # Get nearest enemy:
            min_enemy_distance = 1000.00
            for enem in self.opp_team_pos:
                if enem[0] < min_enemy_distance:
                    min_enemy_distance = enem[0]
                    enemy_loc = self.rb_to_rect(enem)

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

                if enemy_dist_2_flag > defense_perim:
                    if defend_pt_flag_dist > defense_perim:
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

            act_index = 16

            # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
            try:
                act_heading = self.angle180(self.vec_to_heading(ag_vect))
                if 1 >= act_heading >= -1:
                    act_index = 4
                elif act_heading < -1:
                    act_index = 6
                elif act_heading > 1:
                    act_index = 2
            except:
                act_index = 4

        return act_index
