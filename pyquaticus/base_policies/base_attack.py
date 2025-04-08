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
from pyquaticus.envs.pyquaticus import Team, PyQuaticusEnv
from pyquaticus.utils.utils import mag_bearing_to

modes = {"nothing", "easy", "medium", "hard", "competition_easy", "competition_medium"}


class BaseAttacker(BaseAgentPolicy):
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(
        self,
        agent_id: str,
        team: Team,
        env: PyQuaticusEnv,
        continuous: bool = False,
        mode: str = "easy",
    ):
        super().__init__(agent_id, team, env)

        if mode not in modes:
            raise AttributeError(f"Invalid mode {mode}")
        self.mode = mode

        if team not in Team:
            raise AttributeError(f"Invalid team {team}")

        self.continuous = continuous
        self.goal = "SC"

        if not env.gps_env:
            self.aquaticus_field_points = env.aquaticus_field_points

    def set_mode(self, mode: str):
        """Sets difficulty mode."""
        if mode not in modes:
            raise ValueError(f"Invalid mode {mode}")
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
        self.update_state(obs, info)y

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
                heading_error = self.angle180(self.vec_to_heading(goal_vect) - self.heading)

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
                        # Should only happen if the heading error is somehow NAN
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
                        self.wall_distances[1],
                        self.wall_distances[0],
                    ]
                )
            else:
                estimated_position = np.asarray(
                    [
                        self.wall_distances[3],
                        self.wall_distances[2],
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
            if self.is_tagged:
                self.goal = "SC"

            if (
                -2.5
                <= self.get_distance_between_2_points(
                    estimated_position, self.aquaticus_field_points[value]
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
                    estimated_position, self.aquaticus_field_points[value]
                )
                <= 6
            ):
                self.goal = "SC"

            if self.continuous:

                # Make point system the same on both blue and red side
                if self.team == Team.BLUE_TEAM:
                    if "P" in self.goal:
                        self.goal = "S" + self.goal[1:]
                    elif "S" in self.goal:
                        self.goal = "P" + self.goal[1:]
                    if "X" not in self.goal and self.goal not in ["SC", "CC", "PC"]:
                        self.goal += "X"
                    elif self.goal not in ["SC", "CC", "PC"]:
                        self.goal = self.goal[:-1]

                _, heading = mag_bearing_to(
                    self.pos,
                    self.aquaticus_field_points[self.goal],
                    self.heading,
                )
                if (
                    self.get_distance_between_2_points(
                        self.pos,
                        self.aquaticus_field_points[self.goal],
                    )
                    <= 0.3
                ):
                    speed = 0.0
                else:
                    speed = self.max_speed

                return speed, heading
            else:
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
                heading_error = self.angle180(self.vec_to_heading(my_action) - self.heading)

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

            # If I'm close to a wall and driving towards it, add the closest point to the wall as an obstacle to avoid
            if self.wall_distances[0] < 10 and (-90 < self.angle180(self.wall_bearings[0] - self.heading) < 90):
                wall_0_unit_vec = self.rb_to_rect(
                    np.array(
                        (
                            self.wall_distances[0],
                            self.wall_bearings[0],
                        )
                    )
                )
                self.opp_team_pos.append(
                    (
                        self.wall_distances[0] * wall_0_unit_vec[0],
                        self.wall_distances[0] * wall_0_unit_vec[1],
                    )
                )
            elif self.wall_distances[2] < 10 and (-90 < self.angle180(self.wall_bearings[2] - self.heading) < 90):
                wall_2_unit_vec = self.rb_to_rect(
                    np.array(
                        (
                            self.wall_distances[2],
                            self.wall_bearings[2],
                        )
                    )
                )
                self.opp_team_pos.append(
                    (
                        self.wall_distances[2] * wall_2_unit_vec[0],
                        self.wall_distances[2] * wall_2_unit_vec[1],
                    )
                )
            if self.wall_distances[1] < 10 and (-90 < self.angle180(self.wall_bearings[1] - self.heading) < 90):
                wall_1_unit_vec = self.rb_to_rect(
                    np.array(
                        (
                            self.wall_distances[1],
                            self.wall_bearings[1],
                        )
                    )
                )
                self.opp_team_pos.append(
                    (
                        self.wall_distances[1] * wall_1_unit_vec[0],
                        self.wall_distances[1] * wall_1_unit_vec[1],
                    )
                )
            elif self.wall_distances[3] < 10 and (-90 < self.angle180(self.wall_bearings[3] - self.heading) < 90):
                wall_3_unit_vec = self.rb_to_rect(
                    np.array(
                        (
                            self.wall_distances[3],
                            self.wall_bearings[3],
                        )
                    )
                )
                self.opp_team_pos.append(
                    (
                        self.wall_distances[3] * wall_3_unit_vec[0],
                        self.wall_distances[3] * wall_3_unit_vec[1],
                    )
                )

            # Increase the avoidance threshold to start avoiding when farther away
            avoid_thresh = 60.0

            # If I have the flag, go back to my side
            if self.has_flag:
                goal_vect = np.multiply(
                    1.25, self.bearing_to_vec(self.own_home_bearing)
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

                    top_dist = self.wall_distances[0]
                    bottom_dist = self.wall_distances[2]

                    # Some bias towards teh bottom boundary to force it to stick with a
                    # direction.
                    if top_dist > 1.25 * bottom_dist:
                        my_action = self.rb_to_rect(
                            np.array((top_dist, self.wall_bearings[0]))
                        )
                    else:
                        my_action = self.rb_to_rect(
                            np.array((bottom_dist, self.wall_bearings[2]))
                        )
                else:
                    my_action = np.multiply(1.25, goal_vect) + avoid_vect

            # Try to convert the heading to a discrete action
            try:
                heading_error = self.angle180(self.vec_to_heading(my_action) - self.heading)
                # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
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

        elif self.mode == "hard":

            desired_speed = self.max_speed

            # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
            if self.wall_distances[0] < 10 and (-90 < self.angle180(self.wall_bearings[0] - self.heading) < 90):
                wall_0_unit_vec = self.rb_to_rect(
                    np.array(
                        (
                            self.wall_distances[0],
                            self.wall_bearings[0],
                        )
                    )
                )
                self.opp_team_pos.append(
                    (
                        self.wall_distances[0] * wall_0_unit_vec[0],
                        self.wall_distances[0] * wall_0_unit_vec[1],
                    )
                )
            elif self.wall_distances[2] < 10 and (-90 < self.angle180(self.wall_bearings[2] - self.heading) < 90):
                wall_2_unit_vec = self.rb_to_rect(
                    np.array(
                        (
                            self.wall_distances[2],
                            self.wall_bearings[2],
                        )
                    )
                )
                self.opp_team_pos.append(
                    (
                        self.wall_distances[2] * wall_2_unit_vec[0],
                        self.wall_distances[2] * wall_2_unit_vec[1],
                    )
                )
            if self.wall_distances[1] < 10 and (-90 < self.angle180(self.wall_bearings[1] - self.heading) < 90):
                wall_1_unit_vec = self.rb_to_rect(
                    np.array(
                        (
                            self.wall_distances[1],
                            self.wall_bearings[1],
                        )
                    )
                )
                self.opp_team_pos.append(
                    (
                        self.wall_distances[1] * wall_1_unit_vec[0],
                        self.wall_distances[1] * wall_1_unit_vec[1],
                    )
                )
            elif self.wall_distances[3] < 10 and (-90 < self.angle180(self.wall_bearings[3] - self.heading) < 90):
                wall_3_unit_vec = self.rb_to_rect(
                    np.array(
                        (
                            self.wall_distances[3],
                            self.wall_bearings[3],
                        )
                    )
                )
                self.opp_team_pos.append(
                    (
                        self.wall_distances[3] * wall_3_unit_vec[0],
                        self.wall_distances[3] * wall_3_unit_vec[1],
                    )
                )

            # Increase the avoidance threshold to start avoiding when farther away
            avoid_thresh = 30.0
            # If I or someone on my team has the flag, go back to my side
            if self.has_flag or self.my_team_has_flag:
                goal_vect = np.multiply(
                    1.25, self.bearing_to_vec(self.own_home_bearing)
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

                    top_dist = self.wall_distances[0]
                    bottom_dist = self.wall_distances[2]

                    # Some bias towards teh bottom boundary to force it to stick with a
                    # direction.
                    if top_dist > 1.25 * bottom_dist:
                        my_action = self.rb_to_rect(
                            np.array((top_dist, self.wall_bearings[0]))
                        )
                    else:
                        my_action = self.rb_to_rect(
                            np.array((bottom_dist, self.wall_bearings[2]))
                        )
                else:
                    my_action = np.multiply(1.25, goal_vect) + avoid_vect

            # Try to convert the heading to a discrete action
            try:
                heading_error = self.angle180(self.vec_to_heading(my_action) - self.heading)
                # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
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
