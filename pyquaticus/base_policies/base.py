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

from pyquaticus.envs.pyquaticus import Team, PyQuaticusEnv
#from pyquaticus.moos.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.utils.utils import closest_point_on_line, mag_bearing_to

from typing import Any, Union


class BaseAgentPolicy:
    """
    Class containing utility routines for agents to calculate useful info using
    the observation space.
    """

    def __init__(
        self,
        agent_id: str,
        team: Team,
        env: PyQuaticusEnv,#Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        suppress_numpy_warnings=True,
    ):
        self.id = agent_id
        self.idx = env.agents.index(agent_id)
        self.obs_normalizer = env.agent_obs_normalizer
        self.state_normalizer = env.global_state_normalizer

        self.walls = env._walls

        if isinstance(team, str):
            if team == "red":
                team = Team.RED_TEAM
            elif team == "blue":
                team = Team.BLUE_TEAM
            else:
                raise ValueError(f"Got unknown team: {team}")
        self.team = team

        # Make sure own id is not in teammate_ids
        agents_per_team = env.team_size
        if team == Team.BLUE_TEAM:
            self.teammate_idxs = [i for i in range(agents_per_team) if i != self.idx]
            self.teammate_ids = env.agent_ids_of_team[Team.BLUE_TEAM]
            self.opponent_idxs = [i for i in range(agents_per_team, 2 * agents_per_team)]
            self.opponent_ids = env.agent_ids_of_team[Team.RED_TEAM]
        else:
            self.teammate_idxs = [
                i for i in range(agents_per_team, 2 * agents_per_team) if i != self.idx
            ]
            self.teammate_ids = env.agent_ids_of_team[Team.RED_TEAM]
            self.opponent_idxs = [i for i in range(agents_per_team)]
            self.opponent_ids = env.agent_ids_of_team[Team.BLUE_TEAM]

        self.max_speed = env.players[self.id].get_max_speed()

        self.speed = 0.0
        self.has_flag = False
        self.on_sides = False
        self.is_tagged = False

        if suppress_numpy_warnings:
            np.seterr(all="ignore")

    def compute_action(self, obs, info) -> Any:
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
        pass

    def update_state(self, obs, info) -> None:
        """
        Method to convert the gym obs and info into data more relative to the
        agent.

        Args:
            obs: observation from gym
            info: info from gym
        """

        global_state = info[self.id]["global_state"]

        # Unnormalize state, if necessary
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)

        self.opp_team_pos = []
        self.opp_team_pos_dict = {}  # for labeling by agent_id
        self.opp_team_rect_pos = []
        self.my_team_rect_pos = []
        self.my_team_pos = []
        self.opp_team_tag = []
        self.my_team_tag = []
        self.opp_team_has_flag = False
        self.my_team_has_flag = False

        # Copy this agents state from the observation
        idx = self.idx
        self.pos = np.asarray(
            global_state[(self.id, "pos")],
        )
        self.heading = global_state[(self.id, "heading")]
        self.speed = global_state[(self.id, "speed")]
        self.on_sides = global_state[(self.id, "on_side")]
        self.has_flag = global_state[(self.id, "has_flag")]
        self.tagging_cooldown = global_state[(self.id, "tagging_cooldown")]
        self.is_tagged = global_state[(self.id, "is_tagged")]

        # Calculate the rectangular coordinates for the flags location relative to the agent.
        if self.team == Team.BLUE_TEAM:
            self.my_flag_loc = np.asarray(global_state["blue_flag_home"]) - self.pos
            self.my_flag_distance = np.linalg.norm(self.my_flag_loc)
            self.my_flag_bearing = (
                self.angle180(
                    -1.0 * self.vec_to_heading(self.my_flag_loc) + 90
                )  # convert to maritime standard for heading
            )
            self.own_home_bearing = self.my_flag_bearing
            self.opp_flag_loc = np.asarray(global_state["red_flag_home"]) - self.pos
            self.opp_flag_distance = np.linalg.norm(self.opp_flag_loc)
            self.opp_flag_bearing = (
                self.angle180(
                    -1.0 * self.vec_to_heading(self.opp_flag_loc) + 90
                )  # convert to maritime standard for heading
            )
        else:
            self.my_flag_loc = np.asarray(global_state["red_flag_home"]) - self.pos
            self.my_flag_distance = np.linalg.norm(self.my_flag_loc)
            self.my_flag_bearing = (
                self.angle180(
                    -1.0 * self.vec_to_heading(self.my_flag_loc) + 90
                )  # convert to maritime standard for heading
            )
            self.own_home_bearing = self.my_flag_bearing
            self.opp_flag_loc = np.asarray(global_state["blue_flag_home"]) - self.pos
            self.opp_flag_distance = np.linalg.norm(self.opp_flag_loc)
            self.opp_flag_bearing = (
                self.angle180(
                    -1.0 * self.vec_to_heading(self.opp_flag_loc) + 90
                )  # convert to maritime standard for heading
            )

        # Copy the polar (distance and absolute bearing) positions of each agent, separated by team and get their tag status
        # Update flag positions if picked up
        # Get rectangular, absolute positions of each agent as well

        for k in global_state.keys():
            if type(k) is tuple:
                if k[0] in self.opponent_ids:
                    rect_pos = np.asarray(global_state[(k[0], "pos")])
                    polar_pos = np.asarray(
                        (
                            np.linalg.norm(rect_pos - self.pos),
                            self.angle180(
                                -1.0 * self.vec_to_heading(rect_pos - self.pos) + 90
                            )
                        )
                    )
                    self.opp_team_rect_pos.append(rect_pos)
                    self.opp_team_pos.append(polar_pos)
                    self.opp_team_pos_dict[k[0]] = polar_pos
                    self.opp_team_has_flag = (
                        self.opp_team_has_flag or global_state[(k[0], "has_flag")]
                    )

                    # update own flag position if flag has been picked up
                    if global_state[(k[0], "has_flag")]:
                        self.my_flag_distance = polar_pos[0]
                        self.my_flag_bearing = polar_pos[1]
                        self.my_flag_loc = rect_pos - self.pos

                    self.opp_team_tag.append(global_state[(k[0], "is_tagged")])

                elif k[0] in self.teammate_ids:
                    rect_pos = np.asarray(global_state[(k[0], "pos")])
                    polar_pos = np.asarray(
                        (
                            np.linalg.norm(rect_pos - self.pos),
                            self.angle180(
                                -1.0 * self.vec_to_heading(rect_pos - self.pos) + 90
                            )
                        )
                    )
                    self.my_team_rect_pos.append(rect_pos)
                    self.my_team_pos.append(polar_pos)
                    self.my_team_has_flag = (
                        self.my_team_has_flag or global_state[(k[0], "has_flag")]
                    )
                    # update opponent flag position if flag has been picked up by teammate
                    if global_state[(k[0], "has_flag")]:
                        self.opp_flag_distance = polar_pos[0]
                        self.opp_flag_bearing = polar_pos[1]
                        self.opp_flag_loc = rect_pos - self.pos
                    self.my_team_tag.append(global_state[(k[0], "is_tagged")])

            # update opponent flag position if flag has been picked up by agent
            if self.has_flag:
                self.opp_flag_distance = 0.0
                self.opp_flag_bearing = 0.0
                self.opp_flag_loc = np.asarray((0.0, 0.0))

            # Get wall distances and absolute bearings
            self.wall_distances = []
            self.wall_bearings = []
            for i in range(4):
                wall = self.walls[int(self.team)][i]
                wall_closest_point = closest_point_on_line(wall[0], wall[1], self.pos)
                dist, bearing = mag_bearing_to(
                    self.pos, wall_closest_point
                )
                self.wall_distances.append(dist)
                self.wall_bearings.append(bearing)

    def angle180(self, deg):
        """Rotates an angle to be between -180 and +180 degrees."""
        while deg > 180:
            deg -= 360
        while deg < -180:
            deg += 360
        return deg

    def vec_to_heading(self, vec):
        """Converts a vector to a magnitude and heading (deg)."""
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        return self.angle180(angle)

    def bearing_to_vec(self, heading):
        return [np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))]

    def rb_to_rect(self, point: np.ndarray, maritime: bool = True) -> np.ndarray:
        """Returns the rectangular coordinates of polar point `point`."""
        dist = point[0]
        bearing = point[1]
        if maritime:
            bearing = 90 - bearing
        unit_vec = self.bearing_to_vec(bearing)
        return np.array([dist * unit_vec[0], dist * unit_vec[1]], dtype=np.float64)

    def get_avoid_vect(self, avoid_pos, avoid_threshold=10.0):
        """
        This function finds the vector most pointing away to all enemy agents.

        Args:
            agent: An agents position
            avoid_pos: All other agent (polar) positions we potentially need to avoid.
            avoid_threshold: The threshold that, when the agent is closer than this range,
                is attempted to be avoided.

        Returns
        -------
            np.array vector that points away from as many agents as possible
        """
        avoid_vects = []
        need_avoid = False
        for avoid_ag in avoid_pos:
            if avoid_ag[0] < avoid_threshold:
                coeff = np.divide(avoid_threshold, avoid_ag[0])
                ag_vect = self.bearing_to_vec(avoid_ag[1])
                avoid_vects.append([coeff * ag_vect[0], coeff * ag_vect[1]])
                need_avoid = True
        av_x = 0.0
        av_y = 0.0
        if need_avoid:
            for vects in avoid_vects:
                av_x += vects[0]
                av_y += vects[1]
            norm = np.linalg.norm(np.array([av_x, av_y]))
            final_avoid_unit_vect = np.array(
                [-1.0 * np.divide(av_x, norm), -1.0 * np.divide(av_y, norm)]
            )
        else:
            final_avoid_unit_vect = np.array([0, 0])

        return final_avoid_unit_vect

    def unit_vect_between_points(self, start: np.ndarray, end: np.ndarray):
        """Calculates the unit vector between two rectangular points."""
        norm = self.get_distance_between_2_points(start, end)
        vect = np.asarray((end[0] - start[0], end[1] - start[1]))
        unit_vect = np.divide(vect, norm)
        return unit_vect

    def gaussian_unit_vect_between_points(self, start: np.ndarray, end: np.ndarray):
        """Calculates the gaussian unit vector between points."""
        noisy_end_x = np.random.normal(end[0], 5)
        noisy_end_y = np.random.normal(end[1], 5)
        new_end = np.array([noisy_end_x, noisy_end_y])
        norm = self.get_distance_between_2_points(start, new_end)
        vect = np.asarray((new_end[0] - start[0], new_end[1] - start[1]))
        unit_vect = np.divide(vect, norm)
        return unit_vect

    def action_from_vector(self, vector):
        """Returns the action from a vector."""
        return vector

    def distance_between(self, a: tuple[float, float], b: tuple[float, float]):
        """Calculate the distance between two polar coordinates."""
        ra = a[0]
        rb = b[0]
        ta = a[1]
        tb = b[1]
        r_ab_sq = (ra * ra) + (rb * rb)
        dist_sq = r_ab_sq - 2 * ra * rb * np.cos(np.deg2rad(tb - ta))
        return np.sqrt(dist_sq)

    def get_distance_between_2_points(self, start: np.ndarray, end: np.ndarray):
        """Calculates the distance between two rectagular points."""
        return np.linalg.norm(start - end)

    def closest_point_on_line(self, A, B, P):
        """
        Calculates the closest point to point `P` on a line between `A` and `B`.

        Args:
            A: One point on the line (x, y)
            B: Other point on the line (x, y)
            P: The goal point, the point that we want to find the closest point
                on the line of AB
        Returns:
            The point (x, y) on the line AB closest to point P
        """
        A = np.array(A)
        B = np.array(B)
        P = np.array(P)

        v_AB = B - A
        v_AP = P - A
        len_AB = np.linalg.norm(v_AB)
        unit_AB = v_AB / len_AB
        v_AB_AP = np.dot(v_AP, v_AB)
        proj_dist = np.divide(v_AB_AP, len_AB)

        if proj_dist <= 0.0:
            return A
        elif proj_dist >= len_AB:
            return B
        else:
            return np.asarray(
                [A[0] + (unit_AB[0] * proj_dist), A[1] + (unit_AB[1] * proj_dist)]
            )
