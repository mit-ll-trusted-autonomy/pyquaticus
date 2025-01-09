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
from pyquaticus.envs.pyquaticus import config_dict_std, Team, PyQuaticusEnv
from pyquaticus.utils.obs_utils import ObsNormalizer

from typing import Union

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
        self,
        agent_id: int,
        team: Team,
        env: PyQuaticusEnv,
        mode: str = "easy",
        using_pyquaticus: bool = True,
    ):
        super().__init__(agent_id, team, env)
        if mode not in modes:
            raise AttributeError(f"Invalid mode {mode}")
        self.mode = mode

        if team not in Team:
            raise AttributeError(f"Invalid team {team}")

        self.continuous = env.action_type == "continuous"

        self.using_pyquaticus = using_pyquaticus
        self.competition_easy = [15, 6, 50, 35]
        self.goal = "SC"

    def set_mode(self, mode: str):
        """Sets difficulty mode."""
        if mode not in modes:
            raise ValueError(f"Invalid mode {mode}")
        self.mode = mode

    def compute_action(self, obs, info):
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
        self.update_state(obs, info)

        global_state = info["global_state"]

        # Unnormalize state, if necessary
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        if self.mode == "easy":

            desired_speed = self.max_speed / 2

            # If I or someone on my team has the flag, go back home
            if self.has_flag or self.my_team_has_flag:
                goal_vect = self.bearing_to_vec(self.own_home_bearing)

            # Otherwise go get the opponents flag
            else:
                goal_vect = self.bearing_to_vec(self.opp_flag_bearing)

            # Convert the vector to a heading, and then pick the best discrete action to perform
            try:
                heading_error = self.angle180(self.vec_to_heading(goal_vect))

                if self.continuous:
                    if np.isnan(heading_error):
                        heading_error = 0

                    if np.abs(heading_error) < 5:
                        heading_error = 0

                    return (desired_speed, heading_error)

                else:
                    if 1 >= heading_error >= -1:
                        return 12
                    elif heading_error < -1:
                        return 14
                    elif heading_error > 1:
                        return 10
                    else:
                        # Should only happen if the act_heading is somehow NAN
                        return 12
            except Exception:
                # If there is an error converting the vector to a heading, just go straight
                if self.continuous:
                    return (desired_speed, 0)
                else:
                    return 12

        elif self.mode == "nothing":
            if self.continuous:
                return (0, 0)
            else:
                return -1

        elif self.mode == "competition_easy":
            if self.team == Team.RED_TEAM:
                estimated_position = np.asarray(
                    [
                        global_state["wall_1_distance"],
                        global_state["wall_0_distance"],
                    ]
                )
            else:
                estimated_position = np.asarray(
                    [
                        global_state["wall_3_distance"],
                        global_state["wall_2_distance"],
                    ]
                )
            value = self.goal

            if self.team == Team.BLUE_TEAM:
                if "P" in self.goal:
                    value = "S" + value[1:]
                elif "S" in self.goal:
                    value = "P" + value[1:]
                if "X" not in self.goal and self.goal not in ["SC", "CC", "PC"]:
                    value += "X"
                elif self.goal not in ["SC", "CC", "PC"]:
                    value = value[:-1]
            if global_state["is_tagged"]:
                self.goal = "SC"

            if (
                -2.5
                <= self.get_distance_between_2_points(
                    estimated_position, config_dict_std["aquaticus_field_points"][value]
                )
                <= 2.5
            ):

                if self.goal == "SC":
                    self.goal = "CFX"
                elif self.goal == "CFX":
                    self.goal = "PC"
                elif self.goal == "PC":
                    self.goal = "CF"
                elif self.goal == "CF":
                    self.goal = "SC"
            if (
                self.goal == "CF"
                and -6
                <= self.get_distance_between_2_points(
                    estimated_position, config_dict_std["aquaticus_field_points"][value]
                )
                <= 6
            ):
                self.goal = "SC"
            return self.goal
        
        elif self.mode == "medium":

            desired_speed = self.max_speed / 2

            # If I or someone on my team has the flag, return to my side.
            if self.has_flag or self.my_team_has_flag:

                # Weighted to follow goal more than avoiding others
                goal_vect = np.multiply(
                    2.00, self.bearing_to_vec(self.own_home_bearing)
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
                heading_error = self.angle180(self.vec_to_heading(my_action))

                if self.continuous:
                    if np.isnan(heading_error):
                        heading_error = 0

                    if np.abs(heading_error) < 5:
                        heading_error = 0

                    return (desired_speed, heading_error)

                else:
                    if 1 >= heading_error >= -1:
                        return 12
                    elif heading_error < -1:
                        return 14
                    elif heading_error > 1:
                        return 10
                    else:
                        # Should only happen if the act_heading is somehow NAN
                        return 12
            except Exception:
                # If there is an error converting the vector to a heading, just go straight
                if self.continuous:
                    return (desired_speed, 0)
                else:
                    return 12

        elif self.mode == "competition_medium":

            desired_speed = self.max_speed
            
            # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
            if global_state["wall_0_distance"] < 10 and (
                -90 < global_state["wall_0_bearing"] < 90
            ):
                wall_0_unit_vec = self.rb_to_rect(
                    np.array((global_state["wall_0_distance"], global_state["wall_0_bearing"]))
                )
                self.opp_team_pos.append(
                    (
                        global_state["wall_0_distance"] * wall_0_unit_vec[0],
                        global_state["wall_0_distance"] * wall_0_unit_vec[1],
                    )
                )
            elif global_state["wall_2_distance"] < 10 and (
                -90 < global_state["wall_2_bearing"] < 90
            ):
                wall_2_unit_vec = self.rb_to_rect(
                    np.array((global_state["wall_2_distance"], global_state["wall_2_bearing"]))
                )
                self.opp_team_pos.append(
                    (
                        global_state["wall_2_distance"] * wall_2_unit_vec[0],
                        global_state["wall_2_distance"] * wall_2_unit_vec[1],
                    )
                )
            if global_state["wall_1_distance"] < 10 and (
                -90 < global_state["wall_1_bearing"] < 90
            ):
                wall_1_unit_vec = self.rb_to_rect(
                    np.array((global_state["wall_1_distance"], global_state["wall_1_bearing"]))
                )
                self.opp_team_pos.append(
                    (
                        global_state["wall_1_distance"] * wall_1_unit_vec[0],
                        global_state["wall_1_distance"] * wall_1_unit_vec[1],
                    )
                )
            elif global_state["wall_3_distance"] < 10 and (
                -90 < global_state["wall_3_bearing"] < 90
            ):
                wall_3_unit_vec = self.rb_to_rect(
                    np.array((global_state["wall_3_distance"], global_state["wall_3_bearing"]))
                )
                self.opp_team_pos.append(
                    (
                        global_state["wall_3_distance"] * wall_3_unit_vec[0],
                        global_state["wall_3_distance"] * wall_3_unit_vec[1],
                    )
                )

            # Increase the avoidance threshold to start avoiding when farther away
            avoid_thresh = 60.0
            # If I have the flag, go back to my side
            if self.has_flag:
                goal_vect = np.multiply(
                    1.25, self.bearing_to_vec(global_state["own_home_bearing"])
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
                if (not np.any(goal_vect + (avoid_vect))) or (
                    np.allclose(
                        np.abs(np.abs(goal_vect) - np.abs(avoid_vect)),
                        np.zeros(np.array(goal_vect).shape),
                        atol=1e-01,
                        rtol=1e-02,
                    )
                ):
                    # Special case where a player is closely in line with the goal
                    # vector such that the calculated avoid vector nearly negates the
                    # action (the player is in a spot that causes the agent to just go
                    # straight into them). In this case just start going towards the top
                    # or bottom boundary, whichever is farthest.

                    top_dist = global_state["wall_0_distance"]
                    bottom_dist = global_state["wall_2_distance"]

                    # Some bias towards teh bottom boundary to force it to stick with a
                    # direction.
                    if top_dist > 1.25 * bottom_dist:
                        my_action = self.rb_to_rect(
                            np.array((top_dist, global_state["wall_0_bearing"]))
                        )
                    else:
                        my_action = self.rb_to_rect(
                            np.array((bottom_dist, global_state["wall_2_bearing"]))
                        )
                else:
                    my_action = np.multiply(1.25, goal_vect) + avoid_vect

            # Try to convert the heading to a discrete action
            try:
                heading_error = self.angle180(self.vec_to_heading(my_action))
                # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
                if 1 >= heading_error >= -1:
                    return 4
                elif heading_error < -1:
                    return 6
                elif heading_error > 1:
                    return 2
                else:
                    # Should only happen if the act_heading is somehow NAN
                    return 4
            except Exception:
                return 4

        elif self.mode == "hard":

            desired_speed = self.max_speed

            # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
            if global_state["wall_0_distance"] < 10 and (
                -90 < global_state["wall_0_bearing"] < 90
            ):
                wall_0_unit_vec = self.rb_to_rect(
                    np.array((global_state["wall_0_distance"], global_state["wall_0_bearing"]))
                )
                self.opp_team_pos.append(
                    (
                        global_state["wall_0_distance"] * wall_0_unit_vec[0],
                        global_state["wall_0_distance"] * wall_0_unit_vec[1],
                    )
                )
            elif global_state["wall_2_distance"] < 10 and (
                -90 < global_state["wall_2_bearing"] < 90
            ):
                wall_2_unit_vec = self.rb_to_rect(
                    np.array((global_state["wall_2_distance"], global_state["wall_2_bearing"]))
                )
                self.opp_team_pos.append(
                    (
                        global_state["wall_2_distance"] * wall_2_unit_vec[0],
                        global_state["wall_2_distance"] * wall_2_unit_vec[1],
                    )
                )
            if global_state["wall_1_distance"] < 10 and (
                -90 < global_state["wall_1_bearing"] < 90
            ):
                wall_1_unit_vec = self.rb_to_rect(
                    np.array((global_state["wall_1_distance"], global_state["wall_1_bearing"]))
                )
                self.opp_team_pos.append(
                    (
                        global_state["wall_1_distance"] * wall_1_unit_vec[0],
                        global_state["wall_1_distance"] * wall_1_unit_vec[1],
                    )
                )
            elif global_state["wall_3_distance"] < 10 and (
                -90 < global_state["wall_3_bearing"] < 90
            ):
                wall_3_unit_vec = self.rb_to_rect(
                    np.array((global_state["wall_3_distance"], global_state["wall_3_bearing"]))
                )
                self.opp_team_pos.append(
                    (
                        global_state["wall_3_distance"] * wall_3_unit_vec[0],
                        global_state["wall_3_distance"] * wall_3_unit_vec[1],
                    )
                )

            # Increase the avoidance threshold to start avoiding when farther away
            avoid_thresh = 30.0
            # If I or someone on my team has the flag, go back to my side
            if self.has_flag or self.my_team_has_flag:
                goal_vect = np.multiply(
                    1.25, self.bearing_to_vec(global_state["own_home_bearing"])
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
                if (not np.any(goal_vect + (avoid_vect))) or (
                    np.allclose(
                        np.abs(np.abs(goal_vect) - np.abs(avoid_vect)),
                        np.zeros(np.array(goal_vect).shape),
                        atol=1e-01,
                        rtol=1e-02,
                    )
                ):
                    # Special case where a player is closely in line with the goal
                    # vector such that the calculated avoid vector nearly negates the
                    # action (the player is in a spot that causes the agent to just go
                    # straight into them). In this case just start going towards the top
                    # or bottom boundary, whichever is farthest.

                    top_dist = global_state["wall_0_distance"]
                    bottom_dist = global_state["wall_2_distance"]

                    # Some bias towards teh bottom boundary to force it to stick with a
                    # direction.
                    if top_dist > 1.25 * bottom_dist:
                        my_action = self.rb_to_rect(
                            np.array((top_dist, global_state["wall_0_bearing"]))
                        )
                    else:
                        my_action = self.rb_to_rect(
                            np.array((bottom_dist, global_state["wall_2_bearing"]))
                        )
                else:
                    my_action = np.multiply(1.25, goal_vect) + avoid_vect

            # Try to convert the heading to a discrete action
            try:
                heading_error = self.angle180(self.vec_to_heading(my_action))
                # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
                # TODO: Look at this
                if self.continuous:
                    if np.isnan(heading_error):
                        heading_error = 0

                    if np.abs(heading_error) < 5:
                        heading_error = 0

                    return (desired_speed, heading_error)

                else:
                    if 1 >= heading_error >= -1:
                        return 4
                    elif heading_error < -1:
                        return 6
                    elif heading_error > 1:
                        return 2
                    else:
                        # Should only happen if the act_heading is somehow NAN
                        return 4
            except Exception:
                # If there is an error converting the vector to a heading, just go straight
                if self.continuous:
                    return (desired_speed, 0)
                else:
                    return 4

        else:
            if self.continuous:
                return (0, 0)
            else:
                return -1
