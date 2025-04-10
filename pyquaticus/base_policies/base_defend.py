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
from pyquaticus.utils.utils import mag_bearing_to

from typing import Union

modes = {"nothing", "easy", "medium", "hard", "competition_easy", "competition_medium"}


class BaseDefender(BaseAgentPolicy):
    """This is a Policy class that contains logic for defending the flag."""

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
            raise ValueError(f"mode {mode} not in set of valid modes {modes}")
        self.mode = mode

        self.continuous = continuous
        self.flag_keepout = env.flag_keepout_radius
        self.catch_radius = env.catch_radius
        self.goal = "PM"

        if not env.gps_env:
            self.aquaticus_field_points = env.aquaticus_field_points

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
        self.update_state(obs, info)

        global_state = info[self.id]["global_state"]

        # Unnormalize state, if necessary
        if not isinstance(global_state, dict):
            if global_state.dtype == np.object_:
                global_state = global_state[0]
            else:
                if len(global_state.shape) == 1:
                    global_state = self.state_normalizer.unnormalized(global_state)
                else:
                    global_state = self.state_normalizer.unnormalized(global_state[0])

        if self.mode == "easy":

            desired_speed = self.max_speed / 2

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
                heading_error = self.angle180(self.vec_to_heading(ag_vect) - self.heading)

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
                self.get_distance_between_2_points(
                    estimated_position, self.aquaticus_field_points[value]
                )
                <= 2.5
            ):
                if self.goal == "SM":
                    self.goal = "PM"
                else:
                    self.goal = "SM"

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

        elif self.mode == "competition_medium":

            desired_speed = self.max_speed

            my_flag_vec = self.bearing_to_vec(self.my_flag_bearing)
            # Check if opponents are on teams side
            min_enemy_distance = 1000.00
            enemy_dis_dict = {}
            closest_enemy = None
            enemy_loc = None
            for enem, pos in self.opp_team_pos_dict.items():
                enemy_dis_dict[enem] = pos[0]
                if (
                    pos[0] < min_enemy_distance
                    and not global_state[(enem, "is_tagged")]
                    and global_state[(enem, "on_side")] == 0
                ):
                    min_enemy_distance = pos[0]
                    closest_enemy = enem
                    enemy_loc = self.rb_to_rect(pos)
            # If the blue team doesn't have the flag, guard it
            if self.opp_team_has_flag:
                # If the blue team has the flag, chase them
                ag_vect = my_flag_vec
            elif closest_enemy is not None:
                ag_vect = enemy_loc
            else:
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
                point = "CH" if self.team == Team.RED_TEAM else "CHX"
                if (
                    self.get_distance_between_2_points(
                        estimated_position,
                        self.aquaticus_field_points[point],
                    )
                    <= 2.5
                ):
                    if self.continuous:
                        return (0, 0)
                    else:
                        return -1
                else:
                    if self.continuous:
                        goal = "CH"
                        # Make point system the same on both blue and red side
                        if self.team == Team.BLUE_TEAM:
                            if "P" in goal:
                                goal = "S" + goal[1:]
                            elif "S" in goal:
                                goal = "P" + goal[1:]
                            if "X" not in goal and goal not in ["SC", "CC", "PC"]:
                                goal += "X"
                            elif self.goal not in ["SC", "CC", "PC"]:
                                goal = goal[:-1]

                        _, heading = mag_bearing_to(
                            self.pos,
                            self.aquaticus_field_points[goal],
                            self.heading,
                        )
                        if (
                            -0.3
                            <= self.get_distance_between_2_points(
                                self.pos,
                                self.aquaticus_field_points[goal],
                            )
                            <= 0.3
                        ):
                            speed = 0.0
                        else:
                            speed = self.max_speed

                        return speed, heading
                    else:
                        return "CH"

            # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
            # Convert the vector to a heading, and then pick the best discrete action to perform
            try:
                heading_error = self.angle180(self.vec_to_heading(ag_vect) - self.heading)

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

        elif self.mode == "medium":

            desired_speed = self.max_speed / 2

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

                    # Convert the vector to a heading, and then pick the best discrete action to perform
            try:
                heading_error = self.angle180(self.vec_to_heading(ag_vect) - self.heading)

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

        elif self.mode == "hard":

            desired_speed = self.max_speed

            # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
            wall_pos = []
            if self.wall_distances[0] < 7 and (-90 < self.angle180(self.wall_bearings[0] - self.heading) < 90):
                wall_0_unit_vec = self.rb_to_rect(
                    np.array((self.wall_distances[0], self.wall_bearings[0]))
                )
                wall_pos.append(
                    (
                        self.wall_distances[0] * wall_0_unit_vec[0],
                        self.wall_distances[0] * wall_0_unit_vec[1],
                    )
                )
            elif self.wall_distances[2] < 7 and (-90 < self.angle180(self.wall_bearings[2] - self.heading) < 90):
                wall_2_unit_vec = self.rb_to_rect(
                    np.array((self.wall_distances[2], self.wall_bearings[2]))
                )
                wall_pos.append(
                    (
                        self.wall_distances[2] * wall_2_unit_vec[0],
                        self.wall_distances[2] * wall_2_unit_vec[1],
                    )
                )
            if self.wall_distances[1] < 7 and (-90 < self.angle180(self.wall_bearings[1] - self.heading) < 90):
                wall_1_unit_vec = self.rb_to_rect(
                    np.array((self.wall_distances[1], self.wall_bearings[1]))
                )
                wall_pos.append(
                    (
                        self.wall_distances[1] * wall_1_unit_vec[0],
                        self.wall_distances[1] * wall_1_unit_vec[1],
                    )
                )
            elif self.wall_distances[3] < 7 and (-90 < self.angle180(self.wall_bearings[3] - self.heading) < 90):
                wall_3_unit_vec = self.rb_to_rect(
                    np.array((self.wall_distances[3], self.wall_bearings[3]))
                )
                wall_pos.append(
                    (
                        self.wall_distances[3] * wall_3_unit_vec[0],
                        self.wall_distances[3] * wall_3_unit_vec[1],
                    )
                )

            ag_vect = [0, 0]
            defense_perim = 5 * self.flag_keepout
            # Get nearest untagged enemy:
            min_enemy_distance = 1000.00
            enemy_dis_dict = {}
            closest_enemy = None
            enemy_loc = np.asarray((0, 0))
            for enem, pos in self.opp_team_pos_dict.items():
                enemy_dis_dict[enem] = pos[0]
                if (
                    pos[0] < min_enemy_distance
                    and not global_state[(enem, "is_tagged")]
                ):
                    min_enemy_distance = pos[0]
                    closest_enemy = enem
                    enemy_loc = self.rb_to_rect(pos)

            if closest_enemy is None:
                min_enemy_distance = min(enemy_dis_dict.values())
                closest_enemy = min(enemy_dis_dict, key=enemy_dis_dict.__getitem__)
                enemy_loc = self.rb_to_rect(self.opp_team_pos_dict[closest_enemy])

            if not self.opp_team_has_flag:
                # If flag is closest point on line, then unit_def_flag is NaN,
                # which causes agent to stop (act_index is never updated from 16)
                # Is this the desired behavior?
                # print(self.opp_team_pos_dict[closest_enemy])
                # print(self.my_flag_loc)
                # print(enemy_loc)
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
                    or global_state[(closest_enemy, "is_tagged")]
                ):
                    if (
                        defend_pt_flag_dist > defense_perim
                        or global_state[(closest_enemy, "is_tagged")]
                    ):
                        guide_pt = [
                            self.my_flag_loc[0] + (unit_def_flag[0] * defense_perim),
                            self.my_flag_loc[1] + (unit_def_flag[1] * defense_perim),
                        ]
                        # print(guide_pt)
                    else:
                        guide_pt = defend_pt
                else:
                    guide_pt = enemy_loc

                ag_vect = guide_pt
            else:
                ag_vect = self.bearing_to_vec(self.my_flag_bearing)

            if wall_pos is not None:
                # print(wall_pos)
                ag_vect = ag_vect + self.get_avoid_vect(wall_pos)
                #print(self.get_avoid_vect(wall_pos))

            # Modified to use fastest speed and make big turns use a slower speed to increase turning radius
            # Convert the vector to a heading, and then pick the best discrete action to perform
            try:
                heading_error = self.angle180(-1 * self.vec_to_heading(ag_vect) + 90 - self.heading)

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
