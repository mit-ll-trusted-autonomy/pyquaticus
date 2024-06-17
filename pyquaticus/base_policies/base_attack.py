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

modes = {"nothing", "easy", "medium", "hard", "competition_easy", "competition_medium"}
"""
Difficulty modes for the policy, each one has different behavior. 
'easy' = Easy Attacker - Go straight to goal
'medium' = Medium Attacker - Go to goal and avoid others
'hard' = Hard Attacker - Not Implemented, but plan to goal smartly
"""


class BaseAttacker(BaseAgentPolicy):
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(
        self, agent_id: int, team=Team.RED_TEAM, mode="easy", using_pyquaticus=True
    ):
        super().__init__(agent_id, team)
        if mode not in modes:
            raise AttributeError(f"Invalid mode {mode}")
        self.mode = mode

        if team not in Team:
            raise AttributeError(f"Invalid team {team}")

        self.using_pyquaticus = using_pyquaticus
        self.competition_easy = [15, 6, 50, 35]
        self.goal = 'SC'

    def set_mode(self, mode: str):
        """Sets difficulty mode."""
        if mode not in modes:
            raise ValueError(f"Invalid mode {mode}")
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
            # If I or someone on my team has the flag, go back home
            if self.has_flag or self.my_team_has_flag:
                goal_vect = self.bearing_to_vec(my_obs["own_home_bearing"])
            # Otherwise go get the opponents flag
            else:
                goal_vect = self.bearing_to_vec(self.opp_flag_bearing)

            # Convert the vector to a heading, and then pick the best discrete action to perform
            try:
                act_heading = self.angle180(self.vec_to_heading(goal_vect))

                if 1 >= act_heading >= -1:
                    act_index = 12
                elif act_heading < -1:
                    act_index = 14
                elif act_heading > 1:
                    act_index = 10
                else:
                    # Should only happen if the act_heading is somehow NAN
                    act_index = 12
            except:
                # If there is an error converting the vector to a heading, just go straight
                act_index = 12

        elif self.mode=="nothing":
            act_index = -1

        elif self.mode=="competition_easy":
            if self.team == Team.RED_TEAM:
                estimated_position = [my_obs["wall_1_distance"], my_obs["wall_0_distance"]]   
            else:
                estimated_position = [my_obs["wall_3_distance"], my_obs["wall_2_distance"]]
            value = self.goal

            if self.team == Team.BLUE_TEAM:
                if 'P' in self.goal:
                    value = 'S' + value[1:]
                elif 'S' in self.goal:
                    value = 'P' + value[1:]
                if 'X' not in self.goal and self.goal not in ['SC', 'CC', 'PC']:
                    value += 'X'
                elif self.goal not in ['SC', 'CC', 'PC']:
                    value = value[:-1]
            if my_obs["is_tagged"]:
                self.goal = 'SC'

            if -2.5 <= self.get_distance_between_2_points(estimated_position, config_dict_std["aquaticus_field_points"][value]) <= 2.5:
                
                if self.goal == 'SC':
                    self.goal = 'CFX'
                elif self.goal == 'CFX':
                    self.goal = 'PC'
                elif self.goal == 'PC':
                    self.goal = 'CF'
                elif self.goal == 'CF':
                    self.goal = 'SC'
            if self.goal == 'CF' and -6 <= self.get_distance_between_2_points(estimated_position, config_dict_std["aquaticus_field_points"][value]) <= 6:
                self.goal = 'SC'
            return self.goal
        elif self.mode == "medium":
            
           
            # If I or someone on my team has the flag, return to my side.
            if self.has_flag or self.my_team_has_flag:
                # Weighted to follow goal more than avoiding others
                goal_vect = np.multiply(
                    2.00, self.bearing_to_vec(my_obs["own_home_bearing"])
                )
                avoid_vect = self.get_avoid_vect(self.opp_team_pos)
                my_action = goal_vect + avoid_vect

            # Otherwise, go get the other teams flag
            else:
                goal_vect = np.multiply(
                    2.00, self.bearing_to_vec(self.opp_flag_bearing)
                )
                avoid_vect = self.get_avoid_vect(self.opp_team_pos)
                my_action = goal_vect + avoid_vect

            # Convert the heading to a discrete action to follow
            try:
                act_heading = self.angle180(self.vec_to_heading(my_action))

                if 1 >= act_heading >= -1:
                    act_index = 12
                elif act_heading < -1:
                    act_index = 14
                elif act_heading > 1:
                    act_index = 10
                else:
                    # Should only happen if the act_heading is somehow NAN
                    act_index = 12
            except:
                act_index = 12

        elif self.mode == "competition_medium":
            # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
            if my_obs["wall_0_distance"] < 10 and (-90 < my_obs["wall_0_bearing"] < 90):
                wall_0_unit_vec = self.rb_to_rect(
                    (my_obs["wall_0_distance"], my_obs["wall_0_bearing"])
                )
                self.opp_team_pos.append(
                    (
                        my_obs["wall_0_distance"] * wall_0_unit_vec[0],
                        my_obs["wall_0_distance"] * wall_0_unit_vec[1],
                    )
                )
            elif my_obs["wall_2_distance"] < 10 and (
                -90 < my_obs["wall_2_bearing"] < 90
            ):
                wall_2_unit_vec = self.rb_to_rect(
                    (my_obs["wall_2_distance"], my_obs["wall_2_bearing"])
                )
                self.opp_team_pos.append(
                    (
                        my_obs["wall_2_distance"] * wall_2_unit_vec[0],
                        my_obs["wall_2_distance"] * wall_2_unit_vec[1],
                    )
                )
            if my_obs["wall_1_distance"] < 10 and (-90 < my_obs["wall_1_bearing"] < 90):
                wall_1_unit_vec = self.rb_to_rect(
                    (my_obs["wall_1_distance"], my_obs["wall_1_bearing"])
                )
                self.opp_team_pos.append(
                    (
                        my_obs["wall_1_distance"] * wall_1_unit_vec[0],
                        my_obs["wall_1_distance"] * wall_1_unit_vec[1],
                    )
                )
            elif my_obs["wall_3_distance"] < 10 and (
                -90 < my_obs["wall_3_bearing"] < 90
            ):
                wall_3_unit_vec = self.rb_to_rect(
                    (my_obs["wall_3_distance"], my_obs["wall_3_bearing"])
                )
                self.opp_team_pos.append(
                    (
                        my_obs["wall_3_distance"] * wall_3_unit_vec[0],
                        my_obs["wall_3_distance"] * wall_3_unit_vec[1],
                    )
                )

            # Increase the avoidance threshold to start avoiding when farther away
            avoid_thresh = 60.0
            # If I have the flag, go back to my side
            if self.has_flag :
                goal_vect = np.multiply(
                    1.25, self.bearing_to_vec(my_obs["own_home_bearing"])
                )
                avoid_vect = self.get_avoid_vect(
                    self.opp_team_pos, avoid_threshold=avoid_thresh
                )
                my_action = goal_vect + (avoid_vect)

            # Otherwise go get the flag
            else:
                goal_vect = self.bearing_to_vec(self.opp_flag_bearing)
                avoid_vect = self.get_avoid_vect(
                    self.opp_team_pos, avoid_threshold=avoid_thresh
                )
                if ((not np.any(goal_vect + (avoid_vect))) or (np.allclose(np.abs(np.abs(goal_vect) - np.abs(avoid_vect)), np.zeros(np.array(goal_vect).shape), atol = 1e-01, rtol=1e-02))):
                    # Special case where a player is closely in line with the goal
                    # vector such that the calculated avoid vector nearly negates the
                    # action (the player is in a spot that causes the agent to just go
                    # straight into them). In this case just start going towards the top
                    # or bottom boundary, whichever is farthest.

                    top_dist = my_obs["wall_0_distance"]
                    bottom_dist = my_obs["wall_2_distance"]

                    # Some bias towards teh bottom boundary to force it to stick with a
                    # direction.
                    if top_dist > 1.25 * bottom_dist:
                        my_action = self.rb_to_rect((top_dist, my_obs["wall_0_bearing"]))
                    else:
                        my_action = self.rb_to_rect((bottom_dist, my_obs["wall_2_bearing"]))
                else:
                    my_action = np.multiply(1.25, goal_vect) + avoid_vect

            # Try to convert the heading to a discrete action
            try:
                act_heading = self.angle180(self.vec_to_heading(my_action))
                # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
                if 1 >= act_heading >= -1:
                    act_index = 4
                elif act_heading < -1:
                    act_index = 6
                elif act_heading > 1:
                    act_index = 2
                else:
                    # Should only happen if the act_heading is somehow NAN
                    act_index = 4
            except:
                act_index = 4

        elif self.mode == "hard":
            # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
            if my_obs["wall_0_distance"] < 10 and (-90 < my_obs["wall_0_bearing"] < 90):
                wall_0_unit_vec = self.rb_to_rect(
                    (my_obs["wall_0_distance"], my_obs["wall_0_bearing"])
                )
                self.opp_team_pos.append(
                    (
                        my_obs["wall_0_distance"] * wall_0_unit_vec[0],
                        my_obs["wall_0_distance"] * wall_0_unit_vec[1],
                    )
                )
            elif my_obs["wall_2_distance"] < 10 and (
                -90 < my_obs["wall_2_bearing"] < 90
            ):
                wall_2_unit_vec = self.rb_to_rect(
                    (my_obs["wall_2_distance"], my_obs["wall_2_bearing"])
                )
                self.opp_team_pos.append(
                    (
                        my_obs["wall_2_distance"] * wall_2_unit_vec[0],
                        my_obs["wall_2_distance"] * wall_2_unit_vec[1],
                    )
                )
            if my_obs["wall_1_distance"] < 10 and (-90 < my_obs["wall_1_bearing"] < 90):
                wall_1_unit_vec = self.rb_to_rect(
                    (my_obs["wall_1_distance"], my_obs["wall_1_bearing"])
                )
                self.opp_team_pos.append(
                    (
                        my_obs["wall_1_distance"] * wall_1_unit_vec[0],
                        my_obs["wall_1_distance"] * wall_1_unit_vec[1],
                    )
                )
            elif my_obs["wall_3_distance"] < 10 and (
                -90 < my_obs["wall_3_bearing"] < 90
            ):
                wall_3_unit_vec = self.rb_to_rect(
                    (my_obs["wall_3_distance"], my_obs["wall_3_bearing"])
                )
                self.opp_team_pos.append(
                    (
                        my_obs["wall_3_distance"] * wall_3_unit_vec[0],
                        my_obs["wall_3_distance"] * wall_3_unit_vec[1],
                    )
                )

            # Increase the avoidance threshold to start avoiding when farther away
            avoid_thresh = 30.0
            # If I or someone on my team has the flag, go back to my side
            if self.has_flag or self.my_team_has_flag:
                goal_vect = np.multiply(
                    1.25, self.bearing_to_vec(my_obs["own_home_bearing"])
                )
                avoid_vect = self.get_avoid_vect(
                    self.opp_team_pos, avoid_threshold=avoid_thresh
                )
                my_action = goal_vect + (avoid_vect)

            # Otherwise go get the flag
            else:
                goal_vect = self.bearing_to_vec(self.opp_flag_bearing)
                avoid_vect = self.get_avoid_vect(
                    self.opp_team_pos, avoid_threshold=avoid_thresh
                )
                if ((not np.any(goal_vect + (avoid_vect))) or (np.allclose(np.abs(np.abs(goal_vect) - np.abs(avoid_vect)), np.zeros(np.array(goal_vect).shape), atol = 1e-01, rtol=1e-02))):
                    # Special case where a player is closely in line with the goal
                    # vector such that the calculated avoid vector nearly negates the
                    # action (the player is in a spot that causes the agent to just go
                    # straight into them). In this case just start going towards the top
                    # or bottom boundary, whichever is farthest.

                    top_dist = my_obs["wall_0_distance"]
                    bottom_dist = my_obs["wall_2_distance"]

                    # Some bias towards teh bottom boundary to force it to stick with a
                    # direction.
                    if top_dist > 1.25 * bottom_dist:
                        my_action = self.rb_to_rect((top_dist, my_obs["wall_0_bearing"]))
                    else:
                        my_action = self.rb_to_rect((bottom_dist, my_obs["wall_2_bearing"]))
                else:
                    my_action = np.multiply(1.25, goal_vect) + avoid_vect

            # Try to convert the heading to a discrete action
            try:
                act_heading = self.angle180(self.vec_to_heading(my_action))
                # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
                if 1 >= act_heading >= -1:
                    act_index = 4
                elif act_heading < -1:
                    act_index = 6
                elif act_heading > 1:
                    act_index = 2
                else:
                    # Should only happen if the act_heading is somehow NAN
                    act_index = 4
            except:
                act_index = 4

        return act_index
