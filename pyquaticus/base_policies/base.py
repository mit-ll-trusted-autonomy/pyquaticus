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

from typing import Any, Union

import numpy as np

from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge

from pyquaticus.utils.utils import angle180, dist


class BaseAgentPolicy:
    """
    Class containing utility routines for agents to calculate useful info using
    the observation space.
    """

    def __init__(
        self,
        agent_id: str,
        team: Team,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
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
            # self.teammate_idxs = [i for i in range(agents_per_team) if i != self.idx]
            self.teammate_ids = env.agent_ids_of_team[Team.BLUE_TEAM]
            # self.opponent_idxs = [
            #     i for i in range(agents_per_team, 2 * agents_per_team)
            # ]
            self.opponent_ids = env.agent_ids_of_team[Team.RED_TEAM]
        else:
            # self.teammate_idxs = [
            #     i for i in range(agents_per_team, 2 * agents_per_team) if i != self.idx
            # ]
            self.teammate_ids = env.agent_ids_of_team[Team.RED_TEAM]
            # self.opponent_idxs = [i for i in range(agents_per_team)]
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
        unnorm_obs = info[self.id].get("unnorm_obs", None)
        if unnorm_obs is None:
            unnorm_obs = obs[self.id]

        global_state = info[self.id]["global_state"]
        if not isinstance(global_state, dict):
            global_state = self.state_normalizer.unnormalized(global_state)
        print(global_state)

        self.opp_team_pos = []
        self.opp_team_pos_dict = {}  # for labeling by agent_id
        self.my_team_pos = []
        self.opp_team_tag = []
        self.my_team_tag = []
        self.opp_team_has_flag = False
        self.my_team_has_flag = False
        opp_team_ids = set()
        my_team_ids = set()

        # Copy this agents state from the global state
        self.speed = global_state[(self.id, "speed")]
        self.on_sides = global_state[(self.id, "on_side")]
        self.has_flag = global_state[(self.id, "has_flag")]
        self.tagging_cooldown = global_state[(self.id, "tagging_cooldown")]
        self.is_tagged = global_state[(self.id, "is_tagged")]

        # Calculate the rectangular coordinates for the flags location relative to the agent.
        team_str = self.team.name.lower().split("_")[0]
        self.my_flag_distance = unnorm_obs["own_home_distance"]
        self.my_flag_bearing = unnorm_obs["own_home_bearing"]
        self.my_flag_loc = (
            unnorm_obs["own_home_distance"]
            * np.cos(np.deg2rad(unnorm_obs["own_home_bearing"])),
            unnorm_obs["own_home_distance"]
            * np.sin(np.deg2rad(unnorm_obs["own_home_bearing"])),
        )
        self.new_my_flag_distance = dist(
            global_state[(self.id, "pos")], global_state[team_str + "_flag_pos"]
        )
        self.new_my_flag_bearing = angle180(
            self.global_rect_to_abs_bearing(
                global_state[team_str + "_flag_pos"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.new_my_flag_loc = self.dist_rel_bearing_to_local_rect(self.new_my_flag_distance, self.new_my_flag_bearing)

        opp_str = "red" if team_str == "blue" else "blue"
        self.opp_flag_distance = unnorm_obs["opponent_home_distance"]
        self.opp_flag_bearing = unnorm_obs["opponent_home_bearing"]
        self.opp_flag_loc = (
            unnorm_obs["opponent_home_distance"]
            * np.cos(np.deg2rad(unnorm_obs["opponent_home_bearing"])),
            unnorm_obs["opponent_home_distance"]
            * np.sin(np.deg2rad(unnorm_obs["opponent_home_bearing"])),
        )
        self.new_opp_flag_distance = dist(
            global_state[(self.id, "pos")], global_state[opp_str + "_flag_pos"]
        )
        self.new_opp_flag_bearing = angle180(
            self.global_rect_to_abs_bearing(
                global_state[opp_str + "_flag_pos"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.new_opp_flag_loc = self.dist_rel_bearing_to_local_rect(self.new_opp_flag_distance, self.new_opp_flag_bearing)

        self.home = (
            unnorm_obs["own_home_distance"]
            * np.cos(np.deg2rad(unnorm_obs["own_home_bearing"])),
            unnorm_obs["own_home_distance"]
            * np.sin(np.deg2rad(unnorm_obs["own_home_bearing"])),
        )
        self.new_home_distance = dist(
            global_state[(self.id, "pos")], global_state[team_str + "_flag_home"]
        )
        self.new_home_bearing = angle180(
            self.global_rect_to_abs_bearing(
                global_state[team_str + "_flag_home"] - global_state[(self.id, "pos")]
            )
            - global_state[(self.id, "heading")]
        )
        self.new_home_loc = self.dist_rel_bearing_to_local_rect(self.new_home_distance, self.new_home_bearing)

        # Copy the polar positions of each agent, separated by team and get their tag status
        # Update flag positions if picked up
        for k in unnorm_obs:
            if type(k) is tuple:
                if k[0].find("opponent_") != -1 and k[0] not in opp_team_ids:
                    opp_team_ids.add(k[0])
                    self.opp_team_pos.append(
                        (unnorm_obs[(k[0], "distance")], unnorm_obs[(k[0], "bearing")])
                    )
                    self.opp_team_pos_dict[k[0]] = (
                        unnorm_obs[(k[0], "distance")],
                        unnorm_obs[(k[0], "bearing")],
                    )
                    self.opp_team_has_flag = (
                        self.opp_team_has_flag or unnorm_obs[k[0], "has_flag"]
                    )
                    # update own flag position if flag has been picked up
                    if unnorm_obs[k[0], "has_flag"]:
                        self.my_flag_distance = unnorm_obs[(k[0], "distance")]
                        self.my_flag_bearing = unnorm_obs[(k[0], "bearing")]
                        self.my_flag_loc = (
                            unnorm_obs[(k[0], "distance")]
                            * np.cos(np.deg2rad(unnorm_obs[(k[0], "bearing")])),
                            unnorm_obs[(k[0], "distance")]
                            * np.sin(np.deg2rad(unnorm_obs[(k[0], "bearing")])),
                        )
                    self.opp_team_tag.append(unnorm_obs[(k[0], "is_tagged")])
                elif k[0].find("teammate_") != -1 and k[0] not in my_team_ids:
                    my_team_ids.add(k[0])
                    self.my_team_pos.append(
                        (unnorm_obs[(k[0], "distance")], unnorm_obs[(k[0], "bearing")])
                    )
                    self.my_team_has_flag = (
                        self.my_team_has_flag or unnorm_obs[(k[0], "has_flag")]
                    )
                    # update opponent flag position if flag has been picked up by teammate
                    if unnorm_obs[k[0], "has_flag"]:
                        self.opp_flag_distance = unnorm_obs[(k[0], "distance")]
                        self.opp_flag_bearing = unnorm_obs[(k[0], "bearing")]
                        self.opp_flag_loc = (
                            unnorm_obs[(k[0], "distance")]
                            * np.cos(np.deg2rad(unnorm_obs[(k[0], "bearing")])),
                            unnorm_obs[(k[0], "distance")]
                            * np.sin(np.deg2rad(unnorm_obs[(k[0], "bearing")])),
                        )
                    self.my_team_tag.append(unnorm_obs[(k[0], "is_tagged")])
        if self.id == "agent_0":
            print(self.my_team_pos)

        # update opponent flag position if flag has been picked up by agent
        if self.has_flag:
            self.opp_flag_distance = 0.0
            self.opp_flag_bearing = 0.0
            self.opp_flag_loc = (0.0, 0.0)

        # Get wall distances and bearings
        self.wall_distances = []
        self.wall_bearings = []
        for i in range(4):
            self.wall_distances.append(unnorm_obs[f"wall_{i}_distance"])
            self.wall_bearings.append(unnorm_obs[f"wall_{i}_bearing"])

    def vec_to_heading(self, vec):
        """Converts a vector to a magnitude and heading (deg)."""
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        return angle180(angle)

    def bearing_to_vec(self, heading):
        return [np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))]

    def rb_to_rect(self, point: np.ndarray) -> np.ndarray:
        """Returns the rectangular coordinates of polar point `point`."""
        dist = point[0]
        bearing = point[1]
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

    @staticmethod
    def unit_vect_between_points(start: np.ndarray, end: np.ndarray):
        """Calculates the unit vector between two rectangular points."""
        norm = np.linalg.norm(start - end)
        vect = end - start
        unit_vect = np.divide(vect, norm)
        return (end - start) / np.linalg.norm(end - start)

    def action_from_vector(self, vector, desired_speed):
        """Returns the action from a vector."""
        return vector

    @staticmethod
    def global_rect_to_abs_bearing(vec):
        return 90 - np.degrees(np.arctan2(vec[1], vec[0]))
    
    @staticmethod
    def dist_rel_bearing_to_local_rect(dist, rel_bearing):
        return dist * BaseAgentPolicy.rel_bearing_to_local_unit_rect(rel_bearing)

    @staticmethod
    def rel_bearing_to_local_unit_rect(rel_bearing):
        rad = np.deg2rad(rel_bearing)
        return np.array((np.sin(rad), np.cos(rad)))
