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

from pyquaticus.base_policies.base import BaseAgentPolicy
from pyquaticus.base_policies.utils import (
    bearing_to_vec,
    dist_rel_bearing_to_local_rect,
    get_avoid_vect,
    global_rect_to_abs_bearing,
    local_rect_to_rel_bearing,
    rb_to_rect,
    rel_bearing_to_local_unit_rect,
    vec_to_heading,
)
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import angle180, closest_point_on_line, dist

MODES = {"nothing", "easy", "medium", "hard", "competition_easy", "competition_medium"}


class BaseAttacker(BaseAgentPolicy):
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(
        self,
        agent_id: str,
        team: Team,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        continuous: bool = False,
        mode: str = "easy",
    ):
        super().__init__(agent_id, team, env)

        self.set_mode(mode)

        if team not in Team:
            raise AttributeError(f"Invalid team {team}")

        self.continuous = continuous
        self.goal = "SC"

        self.state_normalizer = env.global_state_normalizer
        self.walls = env._walls[team.value]
        self.max_speed = env.players[self.id].get_max_speed()

        if isinstance(env, PyQuaticusMoosBridge) or not env.gps_env:
            self.aquaticus_field_points = env.aquaticus_field_points

    def set_mode(self, mode: str):
        """Sets difficulty mode."""
        if mode not in MODES:
            raise ValueError(f"mode {mode} not in set of valid modes: {MODES}")
        self.mode = mode

    def compute_action(self, obs, info: dict[str, dict]):
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
        self.update_state(obs, info)

        unnorm_obs = info[self.id].get("unnorm_obs", None)
        if unnorm_obs is None:
            unnorm_obs = obs[self.id]

        if self.mode == "easy":

            # If I or someone on my team has the flag, go back home
            if self.has_flag or self.my_team_has_flag:
                return self.action_from_vector(self.home_loc, 0.5)

            # Otherwise go get the opponents flag
            else:
                return self.action_from_vector(self.opp_flag_loc, 0.5)

        elif self.mode == "nothing":
            return self.action_from_vector(None, 0)

        elif self.mode == "competition_easy":

            assert self.aquaticus_field_points is not None

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
                <= dist(estimated_position, self.aquaticus_field_points[value])
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
                <= dist(estimated_position, self.aquaticus_field_points[value])
                <= 6
            ):
                self.goal = "SC"

            return self.goal

        elif self.mode == "medium":

            # If I or someone on my team has the flag, return to my side.
            if self.has_flag or self.my_team_has_flag:

                # Weighted to follow goal more than avoiding others
                goal_vect = 2 * rel_bearing_to_local_unit_rect(self.home_bearing)
                avoid_vect = get_avoid_vect(self.opp_team_pos)
                my_action = goal_vect + avoid_vect

            # Otherwise, go get the other teams flag
            else:
                goal_vect = 2 * rel_bearing_to_local_unit_rect(self.opp_flag_bearing)
                avoid_vect = get_avoid_vect(self.opp_team_pos)
                my_action = goal_vect + avoid_vect

            return self.action_from_vector(my_action, 0.5)

        elif self.mode == "competition_medium":

            # If I'm close to a wall and driving towards it, add the closest point to the wall as an obstacle to avoid
            if self.wall_distances[0] < 10 and (-90 < self.wall_bearings[0] < 90):
                wall_0_unit_vec = rb_to_rect(
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
            elif self.wall_distances[2] < 10 and (-90 < self.wall_bearings[2] < 90):
                wall_2_unit_vec = rb_to_rect(
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
            if self.wall_distances[1] < 10 and (-90 < self.wall_bearings[1] < 90):
                wall_1_unit_vec = rb_to_rect(
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
            elif self.wall_distances[3] < 10 and (-90 < self.wall_bearings[3] < 90):
                wall_3_unit_vec = rb_to_rect(
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
                    1.25, bearing_to_vec(unnorm_obs["own_home_bearing"])
                )
                avoid_vect = get_avoid_vect(
                    self.opp_team_pos, avoid_threshold=avoid_thresh
                )
                my_action = goal_vect + (avoid_vect)

            # Otherwise go get the flag
            else:
                goal_vect = bearing_to_vec(self.opp_flag_bearing)
                avoid_vect = get_avoid_vect(
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

                    # Some bias towards the bottom boundary to force it to stick with a
                    # direction.
                    if top_dist > 1.25 * bottom_dist:
                        my_action = rb_to_rect(
                            np.array((top_dist, self.wall_bearings[0]))
                        )
                    else:
                        my_action = rb_to_rect(
                            np.array((bottom_dist, self.wall_bearings[2]))
                        )
                else:
                    my_action = np.multiply(1.25, goal_vect) + avoid_vect

            return self.action_from_vector(my_action, 1)

        elif self.mode == "hard":

            # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
            if self.wall_distances[0] < 10 and (-90 < self.wall_bearings[0] < 90):
                self.opp_team_pos.append(
                    (
                        self.wall_distances[0],
                        self.wall_bearings[0],
                    )
                )
            elif self.wall_distances[2] < 10 and (-90 < self.wall_bearings[2] < 90):
                self.opp_team_pos.append(
                    (
                        self.wall_distances[2],
                        self.wall_bearings[2],
                    )
                )
            if self.wall_distances[1] < 10 and (-90 < self.wall_bearings[1] < 90):
                self.opp_team_pos.append(
                    (
                        self.wall_distances[1],
                        self.wall_bearings[1],
                    )
                )
            elif self.wall_distances[3] < 10 and (-90 < self.wall_bearings[3] < 90):
                self.opp_team_pos.append(
                    (
                        self.wall_distances[3],
                        self.wall_bearings[3],
                    )
                )

            # Increase the avoidance threshold to start avoiding when farther away
            avoid_thresh = 30.0

            # If I or someone on my team has the flag, go back to my side
            if self.has_flag or self.my_team_has_flag:
                goal_vect = 1.25 * rel_bearing_to_local_unit_rect(self.home_bearing)
                avoid_vect = get_avoid_vect(
                    self.opp_team_pos, avoid_threshold=avoid_thresh
                )
                my_action = goal_vect + (avoid_vect)

            # Otherwise go get the flag
            else:
                goal_vect = rel_bearing_to_local_unit_rect(self.opp_flag_bearing)
                avoid_vect = get_avoid_vect(
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

                    # Some bias towards the bottom boundary to force it to stick with a
                    # direction.
                    if top_dist > 1.25 * bottom_dist:
                        my_action = dist_rel_bearing_to_local_rect(
                            top_dist, self.wall_bearings[0]
                        )
                    else:
                        my_action = dist_rel_bearing_to_local_rect(
                            bottom_dist, self.wall_bearings[2]
                        )
                else:
                    my_action = np.multiply(1.25, goal_vect) + avoid_vect

            return self.action_from_vector(my_action, 1)

        else:
            return self.action_from_vector(None, 0)

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

        my_pos = global_state[(self.id, "pos")]
        my_heading = global_state[(self.id, "heading")]

        # Copy this agents state from the global state
        self.speed = global_state[(self.id, "speed")]
        self.on_sides = global_state[(self.id, "on_side")]
        self.has_flag = global_state[(self.id, "has_flag")]
        self.tagging_cooldown = global_state[(self.id, "tagging_cooldown")]
        self.is_tagged = global_state[(self.id, "is_tagged")]

        # Calculate the rectangular coordinates for the flags location relative to the agent.
        team_str = self.team.name.lower().split("_")[0]
        opp_str = "red" if team_str == "blue" else "blue"

        self.my_flag_distance = dist(my_pos, global_state[team_str + "_flag_pos"])
        self.my_flag_bearing = angle180(
            global_rect_to_abs_bearing(global_state[team_str + "_flag_pos"] - my_pos)
            - my_heading
        )
        self.my_flag_loc = dist_rel_bearing_to_local_rect(
            self.my_flag_distance, self.my_flag_bearing
        )
        self.opp_flag_distance = dist(my_pos, global_state[opp_str + "_flag_pos"])
        self.opp_flag_bearing = angle180(
            global_rect_to_abs_bearing(global_state[opp_str + "_flag_pos"] - my_pos)
            - my_heading
        )
        self.opp_flag_loc = dist_rel_bearing_to_local_rect(
            self.opp_flag_distance, self.opp_flag_bearing
        )

        self.home_distance = dist(my_pos, global_state[team_str + "_flag_home"])
        self.home_bearing = angle180(
            global_rect_to_abs_bearing(global_state[team_str + "_flag_home"] - my_pos)
            - my_heading
        )
        self.home_loc = dist_rel_bearing_to_local_rect(
            self.home_distance, self.home_bearing
        )

        # Copy the polar positions of each agent, separated by team and get their tag status
        """hello"""
        self.opp_team_pos = []
        self.opp_team_pos_dict = {}  # for labeling by agent_id
        self.my_team_pos = []
        self.opp_team_tag = []
        self.my_team_tag = []
        self.opp_team_has_flag = False
        self.my_team_has_flag = False
        for id in self.teammate_ids:
            if id != self.id:
                distance = dist(my_pos, global_state[(id, "pos")])
                bearing = angle180(
                    global_rect_to_abs_bearing(global_state[(id, "pos")] - my_pos)
                    - my_heading
                )
                self.my_team_pos.append(np.array((distance, bearing)))
                self.my_team_has_flag = (
                    self.my_team_has_flag or global_state[(id, "has_flag")]
                )
                self.my_team_tag.append(global_state[(id, "is_tagged")])
        for id in self.opponent_ids:
            distance = dist(my_pos, global_state[(id, "pos")])
            bearing = angle180(
                global_rect_to_abs_bearing(global_state[(id, "pos")] - my_pos)
                - my_heading
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
            closest = closest_point_on_line(wall[0], wall[1], my_pos)
            self.wall_distances.append(dist(closest, my_pos))
            bearing = angle180(
                global_rect_to_abs_bearing(closest - my_pos) - my_heading
            )
            self.wall_bearings.append(bearing)
