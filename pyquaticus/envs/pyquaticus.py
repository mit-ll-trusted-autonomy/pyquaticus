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

import colorsys
import contextily as cx
import copy
import cv2
import itertools
import math
import mercantile as mt
import numpy as np
import os
import pathlib
import pickle
import pygame
import random
import warnings
import subprocess

from abc import ABC
from collections import defaultdict, OrderedDict
from contextily.tile import _sm2ll
from datetime import datetime
from geographiclib.geodesic import Geodesic
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from math import ceil, floor
from pettingzoo import ParallelEnv
from pygame import draw, SRCALPHA, surfarray
from pygame.math import Vector2
from pygame.transform import rotozoom
from pyquaticus.config import (
    ACTION_MAP,
    config_dict_std,
    EQUATORIAL_RADIUS,
    LINE_INTERSECT_TOL,
    lidar_detection_classes,
    LIDAR_DETECTION_CLASS_MAP,
    MAX_SPEED,
    POLAR_RADIUS,
)
from pyquaticus.structs import (
    CircleObstacle,
    Flag,
    PolygonObstacle,
    RenderingPlayer,
    Team,
)
from pyquaticus.utils.obs_utils import ObsNormalizer
from pyquaticus.utils.pid import PID
from pyquaticus.utils.utils import (
    angle180,
    clip,
    closest_point_on_line,
    vector_to,
    detect_collision,
    get_rot_angle,
    get_screen_res,
    heading_angle_conversion,
    mag_bearing_to,
    mag_heading_to_vec,
    rc_intersection,
    reflect_vector,
    rot2d,
    vec_to_mag_heading,
    intersect_line_rectangle,
)
from scipy.ndimage import label
from shapely import intersection, LineString, Point, Polygon
from typing import Optional, Union

from pyquaticus.envs.dynamics.heron import heron_move_agents
from pyquaticus.envs.dynamics.large_usv import large_usv_move_agents
from pyquaticus.envs.dynamics.drone import drone_move_agents
from pyquaticus.envs.dynamics.single_integrator import si_move_agents


class PyQuaticusEnvBase(ParallelEnv, ABC):
    """
    ### Description.

    This class contains the base behavior for the main class PyQuaticusEnv below.
    The functionality of this class is shared between both the main Pyquaticus
    entry point (PyQuaticusEnv) and the PyQuaticusMoosBridge class that allows
    deploying policies on a MOOS-IvP backend.

    The exposed functionality includes the following:
    1. converting from discrete actions to a desired speed/heading command
    2. converting from raw states in Player objects to a normalized observation space

    ### Action Space
    A discrete action space with all combinations of
    max speed, half speed; and
    45 degree heading intervals

    ### Observation Space

        Per Agent (supplied in a dictionary from agent-id to a Box):
            Opponent home flag relative bearing (clockwise degrees)
            Opponent home flag distance (meters)
            Own home flag relative bearing (clockwise degrees)
            Own home flag distance (meters)
            Wall 0 relative bearing (clockwise degrees)
            Wall 0 distance (meters)
            Wall 1 relative bearing (clockwise degrees)
            Wall 1 distance (meters)
            Wall 2 relative bearing (clockwise degrees)
            Wall 2 distance (meters)
            Wall 3 relative bearing (clockwise degrees)
            Wall 3 distance (meters)
            Scrimmage line bearing (clockwise degrees)
            Scrimmage line distance (meters)
            Own speed (meters per second)
            Own flag status (boolean)
            On side (boolean)
            Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            Is tagged (boolean)
            For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
                Bearing from you (clockwise degrees)
                Distance (meters)
                Heading of other agent relative to the vector to you (clockwise degrees)
                Speed (meters per second)
                Has flag status (boolean)
                On their side status (boolean)
                Tagging cooldown (seconds)
                Is tagged (boolean)
        Note 1 : the angles are 0 when the agent is pointed directly at the object
                 and increase in the clockwise direction
        Note 2 : the wall distances can be negative when the agent is out of bounds
        Note 3 : the boolean args Tag/Flag status are -1 false and +1 true
        Note 4: the values are normalized by default
    """

    def _to_speed_heading(self, action_dict):
        """
        Processes the raw discrete actions.

        Returns:
            dict from agent id -> (speed, relative heading)
            Note: we use relative heading here so that it can be used directly
                  to the heading error in the PID controller
        """
        processed_action_dict = OrderedDict()
        for player in self.players.values():
            if player.id in action_dict:
                default_action = True
                try:
                    action_dict[player.id] / 2
                except:
                    default_action = False
                if default_action:
                    speed, heading = self._discrete_action_to_speed_relheading(
                        action_dict[player.id]
                    )
                else:
                    # Make point system the same on both blue and red side
                    if player.team == Team.BLUE_TEAM:
                        if "P" in action_dict[player.id]:
                            action_dict[player.id] = "S" + action_dict[player.id][1:]
                        elif "S" in action_dict[player.id]:
                            action_dict[player.id] = "P" + action_dict[player.id][1:]
                        if "X" not in action_dict[player.id] and action_dict[
                            player.id
                        ] not in ["SC", "CC", "PC"]:
                            action_dict[player.id] += "X"
                        elif action_dict[player.id] not in ["SC", "CC", "PC"]:
                            action_dict[player.id] = action_dict[player.id][:-1]

                    _, heading = mag_bearing_to(
                        player.pos,
                        self.config_dict["aquaticus_field_points"][
                            action_dict[player.id]
                        ],
                        player.heading,
                    )
                    if (
                        -0.3
                        <= self.get_distance_between_2_points(
                            player.pos,
                            self.config_dict["aquaticus_field_points"][
                                action_dict[player.id]
                            ],
                        )
                        <= 0.3
                    ):  #
                        speed = 0.0
                    else:
                        speed = self.max_speed
            else:
                # if no action provided, stop moving
                speed, heading = 0.0, player.heading
            processed_action_dict[player.id] = np.array(
                [speed, heading], dtype=np.float32
            )
        return processed_action_dict

    def _discrete_action_to_speed_relheading(self, action):
        return self.action_map[action]

    def _relheading_to_global_heading(self, player_heading, relheading):
        return angle180((player_heading + relheading) % 360)

    def _register_state_elements(self, num_on_team, num_obstacles):
        """Initializes the normalizers."""
        agent_obs_normalizer = ObsNormalizer(False)
        global_state_normalizer = ObsNormalizer(False)

        ### Agent Observation Normalizer ###
        if self.lidar_obs:
            max_bearing = [180]
            max_dist_scrimmage = [self.env_diag]
            max_dist_lidar = self.num_lidar_rays * [self.lidar_range]
            min_dist = [0.0]
            max_bool, min_bool = [1.0], [0.0]
            max_speed, min_speed = [MAX_SPEED], [0.0]
            max_score, min_score = [self.max_score], [0.0]

            agent_obs_normalizer.register("scrimmage_line_bearing", max_bearing)
            agent_obs_normalizer.register(
                "scrimmage_line_distance", max_dist_scrimmage, min_dist
            )
            agent_obs_normalizer.register("speed", max_speed, min_speed)
            agent_obs_normalizer.register("has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("team_has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("opponent_has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("on_side", max_bool, min_bool)
            agent_obs_normalizer.register(
                "tagging_cooldown", [self.tagging_cooldown], [0.0]
            )
            agent_obs_normalizer.register("is_tagged", max_bool, min_bool)
            agent_obs_normalizer.register("team_score", max_score, min_score)
            agent_obs_normalizer.register("opponent_score", max_score, min_score)
            agent_obs_normalizer.register("ray_distances", max_dist_lidar)
            agent_obs_normalizer.register(
                "ray_labels", self.num_lidar_rays * [len(LIDAR_DETECTION_CLASS_MAP) - 1]
            )
        else:
            max_bearing = [180]
            max_dist = [self.env_diag]
            min_dist = [0.0]
            max_bool, min_bool = [1.0], [0.0]
            max_speed, min_speed = [MAX_SPEED], [0.0]
            max_score, min_score = [self.max_score], [0.0]

            agent_obs_normalizer.register("opponent_home_bearing", max_bearing)
            agent_obs_normalizer.register("opponent_home_distance", max_dist, min_dist)
            agent_obs_normalizer.register("own_home_bearing", max_bearing)
            agent_obs_normalizer.register("own_home_distance", max_dist, min_dist)
            agent_obs_normalizer.register("wall_0_bearing", max_bearing)
            agent_obs_normalizer.register("wall_0_distance", max_dist, min_dist)
            agent_obs_normalizer.register("wall_1_bearing", max_bearing)
            agent_obs_normalizer.register("wall_1_distance", max_dist, min_dist)
            agent_obs_normalizer.register("wall_2_bearing", max_bearing)
            agent_obs_normalizer.register("wall_2_distance", max_dist, min_dist)
            agent_obs_normalizer.register("wall_3_bearing", max_bearing)
            agent_obs_normalizer.register("wall_3_distance", max_dist, min_dist)
            agent_obs_normalizer.register("scrimmage_line_bearing", max_bearing)
            agent_obs_normalizer.register("scrimmage_line_distance", max_dist, min_dist)
            agent_obs_normalizer.register("speed", max_speed, min_speed)
            agent_obs_normalizer.register("has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("on_side", max_bool, min_bool)
            agent_obs_normalizer.register(
                "tagging_cooldown", [self.tagging_cooldown], [0.0]
            )
            agent_obs_normalizer.register("is_tagged", max_bool, min_bool)
            agent_obs_normalizer.register("team_score", max_score, min_score)
            agent_obs_normalizer.register("opponent_score", max_score, min_score)

            for i in range(num_on_team - 1):
                teammate_name = f"teammate_{i}"
                agent_obs_normalizer.register((teammate_name, "bearing"), max_bearing)
                agent_obs_normalizer.register(
                    (teammate_name, "distance"), max_dist, min_dist
                )
                agent_obs_normalizer.register(
                    (teammate_name, "relative_heading"), max_bearing
                )
                agent_obs_normalizer.register(
                    (teammate_name, "speed"), max_speed, min_speed
                )
                agent_obs_normalizer.register(
                    (teammate_name, "has_flag"), max_bool, min_bool
                )
                agent_obs_normalizer.register(
                    (teammate_name, "on_side"), max_bool, min_bool
                )
                agent_obs_normalizer.register(
                    (teammate_name, "tagging_cooldown"), [self.tagging_cooldown], [0.0]
                )
                agent_obs_normalizer.register(
                    (teammate_name, "is_tagged"), max_bool, min_bool
                )

            for i in range(num_on_team):
                opponent_name = f"opponent_{i}"
                agent_obs_normalizer.register((opponent_name, "bearing"), max_bearing)
                agent_obs_normalizer.register(
                    (opponent_name, "distance"), max_dist, min_dist
                )
                agent_obs_normalizer.register(
                    (opponent_name, "relative_heading"), max_bearing
                )
                agent_obs_normalizer.register(
                    (opponent_name, "speed"), max_speed, min_speed
                )
                agent_obs_normalizer.register(
                    (opponent_name, "has_flag"), max_bool, min_bool
                )
                agent_obs_normalizer.register(
                    (opponent_name, "on_side"), max_bool, min_bool
                )
                agent_obs_normalizer.register(
                    (opponent_name, "tagging_cooldown"), [self.tagging_cooldown], [0.0]
                )
                agent_obs_normalizer.register(
                    (opponent_name, "is_tagged"), max_bool, min_bool
                )

            for i in range(num_obstacles):
                agent_obs_normalizer.register(
                    f"obstacle_{i}_distance", max_dist, min_dist
                )
                agent_obs_normalizer.register(f"obstacle_{i}_bearing", max_bearing)

        ### Global State Normalizer ###
        max_heading = [180]
        max_bearing = [180]
        pos_x_max = [self.env_size[0] / 2]
        pos_y_max = [self.env_size[1] / 2]
        max_dist_scrimmage = [self.env_diag]
        min_dist = [0.0]
        max_bool, min_bool = [1.0], [0.0]
        max_speed, min_speed = [MAX_SPEED], [0.0]
        max_score, min_score = [self.max_score], [0.0]

        for player in self.players.values():
            player_name = f"player_{player.id}"

            global_state_normalizer.register((player_name, "player_pos_x"), pos_x_max)
            global_state_normalizer.register((player_name, "player_pos_y"), pos_y_max)
            global_state_normalizer.register(
                (player_name, "player_heading"), max_heading
            )
            global_state_normalizer.register(
                (player_name, "player_scrimmage_line_distance"),
                max_dist_scrimmage,
                min_dist,
            )
            global_state_normalizer.register(
                (player_name, "player_scrimmage_line_bearing"), max_bearing
            )
            global_state_normalizer.register(
                (player_name, "player_speed"), max_speed, min_speed
            )
            global_state_normalizer.register(
                (player_name, "player_is_tagged"), max_bool, min_bool
            )
            global_state_normalizer.register(
                (player_name, "player_has_flag"), max_bool, min_bool
            )
            global_state_normalizer.register(
                (player_name, "player_tagging_cooldown"), [self.tagging_cooldown], [0.0]
            )
            global_state_normalizer.register(
                (player_name, "player_on_side"), max_bool, min_bool
            )
            global_state_normalizer.register(
                (player_name, "player_oob"), max_bool, min_bool
            )

        global_state_normalizer.register("blue_flag_home_x", pos_x_max)
        global_state_normalizer.register("blue_flag_home_y", pos_y_max)
        global_state_normalizer.register("red_flag_home_x", pos_x_max)
        global_state_normalizer.register("red_flag_home_y", pos_y_max)

        global_state_normalizer.register("blue_flag_pos_x", pos_x_max)
        global_state_normalizer.register("blue_flag_pos_y", pos_y_max)
        global_state_normalizer.register("red_flag_pos_x", pos_x_max)
        global_state_normalizer.register("red_flag_pos_y", pos_y_max)

        global_state_normalizer.register("blue_flag_pickup", max_bool, min_bool)
        global_state_normalizer.register("red_flag_pickup", max_bool, min_bool)

        global_state_normalizer.register("blue_team_score", max_score, min_score)
        global_state_normalizer.register("red_team_score", max_score, min_score)

        return agent_obs_normalizer, global_state_normalizer

    def state_to_obs(self, agent_id, normalize=True):
        """
        Returns a local observation space. These observations are
        based entirely on the agent local coordinate frame rather
        than the world frame.
        This was originally designed so that observations can be
        easily shared between different teams and agents.
        Without this the world frame observations from the blue and
        red teams are flipped (e.g., the goal is in the opposite
        direction)
        Observation Space (per agent):
            Opponent home relative bearing (clockwise degrees)
            Opponent home distance (meters)
            Home relative bearing (clockwise degrees)
            Home distance (meters)
            Wall 0 relative bearing (clockwise degrees)
            Wall 0 distance (meters)
            Wall 1 relative bearing (clockwise degrees)
            Wall 1 distance (meters)
            Wall 2 relative bearing (clockwise degrees)
            Wall 2 distance (meters)
            Wall 3 relative bearing (clockwise degrees)
            Wall 3 distance (meters)
            Scrimmage line bearing (clockwise degrees)
            Scrimmage line distance (meters)
            Own speed (meters per second)
            Own flag status (boolean)
            On side (boolean)
            Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            Is tagged (boolean)
            Team score (cummulative flag captures)
            Opponent score (cummulative flag captures)
            For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
                Bearing from you (clockwise degrees)
                Distance (meters)
                Heading of other agent relative to the vector to you (clockwise degrees)
                Speed (meters per second)
                Has flag status (boolean)
                On their side status (boolean)
                Tagging cooldown (seconds)
                Is tagged (boolean)
        Note 1 : the angles are 0 when the agent is pointed directly at the object
                 and increase in the clockwise direction
        Note 2 : the wall distances can be negative when the agent is out of bounds
        Note 3 : the boolean args Tag/Flag status are -1 false and +1 true
        Developer Note 1: changes here should be reflected in _register_state_elements.
        Developer Note 2: check that variables used here are available to PyQuaticusMoosBridge in pyquaticus_moos_bridge.py
        """
        agent = self.players[agent_id]
        obs_dict = OrderedDict()
        obs = OrderedDict()
        own_team = agent.team
        other_team = Team.BLUE_TEAM if own_team == Team.RED_TEAM else Team.RED_TEAM
        np_pos = np.array(agent.pos, dtype=np.float32)

        if self.lidar_obs:
            # Scrimmage line
            scrimmage_line_closest_point = closest_point_on_line(
                self.scrimmage_coords[0], self.scrimmage_coords[1], np_pos
            )
            scrimmage_line_dist, scrimmage_line_bearing = mag_bearing_to(
                np_pos, scrimmage_line_closest_point, agent.heading
            )
            obs["scrimmage_line_bearing"] = scrimmage_line_bearing
            obs["scrimmage_line_distance"] = scrimmage_line_dist

            # Own speed
            obs["speed"] = agent.speed
            # Own flag status
            obs["has_flag"] = agent.has_flag
            # Team has flag
            obs["team_has_flag"] = self.state["flag_taken"][int(other_team)]
            # Opposing team has flag
            obs["opponent_has_flag"] = self.state["flag_taken"][int(own_team)]
            # On side
            obs["on_side"] = agent.on_own_side
            # Tagging cooldown
            obs["tagging_cooldown"] = agent.tagging_cooldown
            # Is tagged
            obs["is_tagged"] = agent.is_tagged

            # Team score and Opponent score
            if agent.team == Team.BLUE_TEAM:
                obs["team_score"] = self.state["captures"][0]
                obs["opponent_score"] = self.state["captures"][1]
            else:
                obs["team_score"] = self.state["captures"][1]
                obs["opponent_score"] = self.state["captures"][0]

            # Lidar
            obs["ray_distances"] = self.state["lidar_distances"][agent_id]
            obs["ray_labels"] = self.obj_ray_detection_states[own_team][
                self.state["lidar_labels"][agent_id]
            ]

        else:
            own_home_loc = self.flags[int(own_team)].home
            opponent_home_loc = self.flags[int(other_team)].home

            # Goal flag
            opponent_home_dist, opponent_home_bearing = mag_bearing_to(
                np_pos, opponent_home_loc, agent.heading
            )
            # Defend flag
            own_home_dist, own_home_bearing = mag_bearing_to(
                np_pos, own_home_loc, agent.heading
            )

            obs["opponent_home_bearing"] = opponent_home_bearing
            obs["opponent_home_distance"] = opponent_home_dist
            obs["own_home_bearing"] = own_home_bearing
            obs["own_home_distance"] = own_home_dist

            # Walls
            for i, wall in enumerate(self._walls[int(own_team)]):
                wall_closest_point = closest_point_on_line(wall[0], wall[1], np_pos)
                wall_dist, wall_bearing = mag_bearing_to(
                    np_pos, wall_closest_point, agent.heading
                )
                obs[f"wall_{i}_bearing"] = wall_bearing
                obs[f"wall_{i}_distance"] = wall_dist

            # Scrimmage line
            scrimmage_line_closest_point = closest_point_on_line(
                self.scrimmage_coords[0], self.scrimmage_coords[1], np_pos
            )
            scrimmage_line_dist, scrimmage_line_bearing = mag_bearing_to(
                np_pos, scrimmage_line_closest_point, agent.heading
            )
            obs["scrimmage_line_bearing"] = scrimmage_line_bearing
            obs["scrimmage_line_distance"] = scrimmage_line_dist

            # Own speed
            obs["speed"] = agent.speed
            # Own flag status
            obs["has_flag"] = agent.has_flag
            # On side
            obs["on_side"] = agent.on_own_side
            # Tagging cooldown
            obs["tagging_cooldown"] = agent.tagging_cooldown
            # Is tagged
            obs["is_tagged"] = agent.is_tagged

            # Team score and Opponent score
            if agent.team == Team.BLUE_TEAM:
                obs["team_score"] = self.state["captures"][0]
                obs["opponent_score"] = self.state["captures"][1]
            else:
                obs["team_score"] = self.state["captures"][1]
                obs["opponent_score"] = self.state["captures"][0]

            # Relative observations to other agents
            # teammates first
            # TODO: consider sorting these by some metric
            #       in an attempt to get permutation invariance
            #       distance or maybe flag status (or some combination?)
            #       i.e. sorted by perceived relevance
            for team in [own_team, other_team]:
                dif_agents = filter(
                    lambda a: a.id != agent.id, self.agents_of_team[team]
                )
                for i, dif_agent in enumerate(dif_agents):
                    entry_name = (
                        f"teammate_{i}" if team == own_team else f"opponent_{i}"
                    )

                    dif_np_pos = np.array(dif_agent.pos, dtype=np.float32)
                    dif_agent_dist, dif_agent_bearing = mag_bearing_to(
                        np_pos, dif_np_pos, agent.heading
                    )
                    _, hdg_to_agent = mag_bearing_to(dif_np_pos, np_pos)
                    hdg_to_agent = hdg_to_agent % 360
                    # bearing relative to the bearing to you
                    obs[(entry_name, "bearing")] = dif_agent_bearing
                    obs[(entry_name, "distance")] = dif_agent_dist
                    obs[(entry_name, "relative_heading")] = angle180(
                        (dif_agent.heading - hdg_to_agent) % 360
                    )
                    obs[(entry_name, "speed")] = dif_agent.speed
                    obs[(entry_name, "has_flag")] = dif_agent.has_flag
                    obs[(entry_name, "on_side")] = dif_agent.on_own_side
                    obs[(entry_name, "tagging_cooldown")] = dif_agent.tagging_cooldown
                    obs[(entry_name, "is_tagged")] = dif_agent.is_tagged

        obs_dict[agent.id] = obs
        if normalize:
            obs_dict[agent.id] = self.agent_obs_normalizer.normalized(
                obs_dict[agent.id]
            )
        return obs_dict[agent.id]

    def state_to_global_obs(self, normalize=True):
        """
        Returns a global observation space. These observations are
        based entirely on the center of the environment coordinate frame.
        Observation Space:
            - Agent 0:
              - position x
              - position y
              - heading
              - distance to scrimmage line
              - bearing wrt scrimmage line bearing
              - speed
              - is tagged
              - has flag
              - tagging cooldown
              - on side
              - out of bounds
            - Agent 1: same as Agent 0
            - Agent 2: same as Agent 0
            - Repeat through Agent n
            - Blue flag home x
            - Blue flag home y
            - Red flag home x
            - Red flag home y
            - Blue flag position x
            - Blue flag position y
            - Red flag position x
            - Red flag position y
            - Blue flag pickup
            - Red flag pickup
            - Blue team score
            - Red team score

        Developer Note 1: changes here should be reflected in _register_state_elements.
        Developer Note 2: check that variables used here are available to PyQuaticusMoosBridge in pyquaticus_moos_bridge.py
        """

        global_obs_dict = dict()

        for i, player in enumerate(self.players.values()):
            player_name = f"player_{player.id}"
            np_pos = np.array(player.pos)

            scrimmage_line_closest_point = closest_point_on_line(
                self.scrimmage_coords[0], self.scrimmage_coords[1], np_pos
            )
            scrimmage_line_dist, _ = mag_bearing_to(
                np_pos, scrimmage_line_closest_point, player.heading
            )

            _, scrimmage_line_bearing = vec_to_mag_heading(self.scrimmage_vec)
            player_scrimmage_line_bearing = player.heading - scrimmage_line_bearing

            global_obs_dict[(player_name, "player_pos_x")] = (
                np_pos[0] - self.env_size[0] / 2
            )
            global_obs_dict[(player_name, "player_pos_y")] = (
                np_pos[1] - self.env_size[1] / 2
            )
            global_obs_dict[(player_name, "player_heading")] = player.heading
            global_obs_dict[(player_name, "player_scrimmage_line_distance")] = (
                scrimmage_line_dist
            )
            global_obs_dict[(player_name, "player_scrimmage_line_bearing")] = (
                player_scrimmage_line_bearing
            )
            global_obs_dict[(player_name, "player_speed")] = player.speed
            global_obs_dict[(player_name, "player_is_tagged")] = player.is_tagged
            global_obs_dict[(player_name, "player_has_flag")] = player.has_flag
            global_obs_dict[(player_name, "player_tagging_cooldown")] = (
                player.tagging_cooldown
            )
            global_obs_dict[(player_name, "player_on_side")] = player.on_own_side
            global_obs_dict[(player_name, "player_oob")] = self.state["agent_oob"][i]

        global_obs_dict["blue_flag_home_x"] = (
            self.flags[0].home[0] - self.env_size[0] / 2
        )
        global_obs_dict["blue_flag_home_y"] = (
            self.flags[0].home[1] - self.env_size[1] / 2
        )
        global_obs_dict["red_flag_home_x"] = (
            self.flags[1].home[0] - self.env_size[0] / 2
        )
        global_obs_dict["red_flag_home_y"] = (
            self.flags[1].home[1] - self.env_size[1] / 2
        )
        global_obs_dict["blue_flag_pos_x"] = self.flags[0].pos[0] - self.env_size[0] / 2
        global_obs_dict["blue_flag_pos_y"] = self.flags[0].pos[1] - self.env_size[1] / 2
        global_obs_dict["red_flag_pos_x"] = self.flags[1].pos[0] - self.env_size[0] / 2
        global_obs_dict["red_flag_pos_y"] = self.flags[1].pos[1] - self.env_size[1] / 2
        global_obs_dict["blue_flag_pickup"] = self.state["flag_taken"][0]
        global_obs_dict["red_flag_pickup"] = self.state["flag_taken"][1]
        global_obs_dict["blue_team_score"] = self.state["captures"][0]
        global_obs_dict["red_team_score"] = self.state["captures"][1]

        if normalize:
            global_obs_array = self.global_state_normalizer.normalized(global_obs_dict)
            return global_obs_array
        else:
            return global_obs_dict

    def get_agent_observation_space(self):
        """Overridden method inherited from `Gym`."""
        if self.normalize:
            agent_obs_space = self.agent_obs_normalizer.normalized_space
        else:
            agent_obs_space = self.agent_obs_normalizer.unnormalized_space
            raise Warning(
                "Unnormalized observation space has not been thoroughly tested"
            )
        return agent_obs_space

    def get_agent_action_space(self):
        """Overridden method inherited from `Gym`."""
        return Discrete(len(self.action_map))

    def _determine_team_wall_orient(self):
        """
        To ensure that the observation space is symmetric for both teams,
        we rotate the order wall observations are reported. Otherwise
        there will be differences between which wall is closest to your
        defend flag vs capture flag.

        For backwards compatability reasons, here is the order:

             _____________ 0 _____________
            |                             |
            |                             |
            |   opp                own    |
            3   flag               flag   1
            |                             |
            |                             |
            |_____________ 2 _____________|

        Note that for each team, the walls will be rotated such that the
        first wall observation is from the wall to the right if facing away
        from your own flag.
        """

        all_walls = [
            [self.env_ul.astype(np.float32), self.env_ur.astype(np.float32)],
            [self.env_ur.astype(np.float32), self.env_lr.astype(np.float32)],
            [self.env_lr.astype(np.float32), self.env_ll.astype(np.float32)],
            [self.env_ll.astype(np.float32), self.env_ul.astype(np.float32)],
        ]

        def rotate_walls(walls, amt):
            rot_walls = copy.deepcopy(walls)
            return rot_walls[amt:] + rot_walls[:amt]

        # determine orientation for each team
        blue_flag = self.flags[int(Team.BLUE_TEAM)].home
        red_flag = self.flags[int(Team.RED_TEAM)].home

        team_flags_midpoint = (blue_flag + red_flag) / 2

        blue_wall_vec = blue_flag - team_flags_midpoint
        blue_wall_ray_end = team_flags_midpoint + self.env_diag * (
            blue_wall_vec / np.linalg.norm(blue_wall_vec)
        )
        blue_wall_ray = LineString((team_flags_midpoint, blue_wall_ray_end))

        red_wall_vec = red_flag - team_flags_midpoint
        red_wall_ray_end = team_flags_midpoint + self.env_diag * (
            red_wall_vec / np.linalg.norm(red_wall_vec)
        )
        red_wall_ray = LineString((team_flags_midpoint, red_wall_ray_end))

        blue_borders = self._point_on_which_border(
            intersection(blue_wall_ray, Polygon(self.env_vertices)).coords[1]
        )
        red_borders = self._point_on_which_border(
            intersection(red_wall_ray, Polygon(self.env_vertices)).coords[1]
        )

        if len(blue_borders) == len(red_borders) == 2:
            # blue wall
            if 3 in blue_borders and 0 in blue_borders:
                blue_border = 0
            else:
                blue_border = max(blue_borders)
            # red wall
            if 3 in blue_borders and 0 in blue_borders:
                red_border = 0
            else:
                red_border = max(red_borders)
        elif len(blue_borders) == 2:
            red_border = red_borders[0]
            blue_border = (red_border + 2) % 4
        elif len(red_borders) == 2:
            blue_border = blue_borders[0]
            red_border = (blue_border + 2) % 4
        else:
            blue_border = blue_borders[0]
            red_border = red_borders[0]

        blue_wall = (
            3 - blue_border
        )  # converting to corresponding wall idx within all_walls for backwards compatibility
        red_wall = (
            3 - red_border
        )  # converting to corresponding wall idx within all_walls for backwards compatibility

        blue_rot_amt = (
            blue_wall - 1
        )  # wall 1 is the flag wall (see wall ordering in function description)
        red_rot_amt = (
            red_wall - 1
        )  # wall 1 is the flag wall (see wall ordering in function description)

        self._walls = {}
        self._walls[int(Team.BLUE_TEAM)] = rotate_walls(all_walls, blue_rot_amt)
        self._walls[int(Team.RED_TEAM)] = rotate_walls(all_walls, red_rot_amt)


class PyQuaticusEnv(PyQuaticusEnvBase):
    """
    ### Description.
    This environment simulates a game of capture the flag with agent dynamics based on MOOS-IvP
    (https://oceanai.mit.edu/ivpman/pmwiki/pmwiki.php?n=IvPTools.USimMarine#section5).

    ### Arguments
    team_size: number of agents per team
    reward_config: a dictionary configuring the reward structure (see rewards.py)
    config_dict: a dictionary configuring the environment (see config_dict_std above)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        team_size: int = 1,
        reward_config: dict = None,
        config_dict=config_dict_std,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.team_size = team_size
        self.num_blue = team_size
        self.num_red = team_size
        self.reward_config = {} if reward_config is None else reward_config
        self.config_dict = config_dict
        self.render_mode = render_mode

        self.reset_count = 0
        self.current_time = 0
        self.learning_iteration = 0

        self.state = {}
        self.dones = {}

        self.seed()

        # Set variables from config
        self.set_config_values(config_dict)

        # Create players, use IDs from [0, (2 * team size) - 1] so their IDs can also be used as indices.
        b_players = []
        r_players = []

        for i in range(0, self.num_blue):
            b_players.append(
                RenderingPlayer(
                    i, Team.BLUE_TEAM, self.agent_render_radius, render_mode
                )
            )
        for i in range(self.num_blue, self.num_blue + self.num_red):
            r_players.append(
                RenderingPlayer(i, Team.RED_TEAM, self.agent_render_radius, render_mode)
            )

        self.players = {
            player.id: player for player in itertools.chain(b_players, r_players)
        }  # maps player ids (or names) to player objects
        self.agents = [agent_id for agent_id in self.players]

        # Agents (player objects) of each team
        self.agents_of_team = {Team.BLUE_TEAM: b_players, Team.RED_TEAM: r_players}
        self.agent_ids_of_team = {
            team: [player.id for player in self.agents_of_team[team]] for team in Team
        }

        # Mappings from agent ids to team member ids and opponent ids
        self.agent_to_team_ids = {
            agent_id: [p.id for p in self.agents_of_team[player.team]]
            for agent_id, player in self.players.items()
        }
        self.agent_to_opp_ids = {
            agent_id: [p.id for p in self.agents_of_team[Team(not player.team.value)]]
            for agent_id, player in self.players.items()
        }

        # Create a PID controller for each agent
        if self.render_mode:
            dt = 1 / self.render_fps
        else:
            dt = self.tau

        self._pid_controllers = {}
        for player in self.players.values():
            self._pid_controllers[player.id] = {
                "speed": PID(dt=dt, kp=1.0, ki=0.0, kd=0.0, integral_max=0.07),
                "heading": PID(dt=dt, kp=0.35, ki=0.0, kd=0.07, integral_max=0.07),
            }

        # Create the list of flags that are indexed by self.flags[int(player.team)]
        self.flags = []
        for team in Team:
            self.flags.append(Flag(team))

        # Obstacles and Lidar
        self.set_geom_config(config_dict)

        # Setup action and observation spaces
        self.action_map = [[self.max_speed * spd, hdg] for (spd, hdg) in ACTION_MAP]
        self.action_spaces = {
            agent_id: self.get_agent_action_space() for agent_id in self.players
        }

        self.agent_obs_normalizer, self.global_state_normalizer = (
            self._register_state_elements(team_size, len(self.obstacles))
        )
        self.observation_spaces = {
            agent_id: self.get_agent_observation_space() for agent_id in self.players
        }

        # Setup rewards and params
        for a in self.players:
            if a not in self.reward_config:
                self.reward_config[a] = None

        self.params = {agent_id: {} for agent_id in self.players}
        self.prev_params = {agent_id: {} for agent_id in self.players}

        # Pygame
        self.screen = None
        self.clock = None
        self.isopen = False
        self.render_ctr = 0
        self.render_buffer = []
        self.traj_render_buffer = {}

        if self.render_mode:
            self.create_background_image()  # create background pygame surface (for faster rendering)

    def seed(self, seed=None):
        """
        Overridden method from Gym inheritance to set seeds in the environment.

        Args:
            seed (optional): Starting seed

        Returns
        -------
            List of seeds used for the environment.
        """
        random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, raw_action_dict):
        """
        Steps the environment forward in time by `tau`, applying actions.

        Args:
            raw_action_dict: Actions IDs from discrete action space for agents to apply

        Returns
        -------
            New observation after stepping

            Some reward

            Indicator for whether or not the env is done/truncated

            Additional info (not currently used)
        """
        if self.state is None:
            raise Exception("Call reset before using step method.")

        if not set(raw_action_dict.keys()) <= set(self.players):
            raise ValueError(
                "Keys of action dict should be player ids but got"
                f" {raw_action_dict.keys()}"
            )

        for i, player in enumerate(self.players.values()):
            if player.tagging_cooldown != self.tagging_cooldown:
                # player is still under a cooldown from tagging, advance their cooldown timer, clip at the configured tagging cooldown
                player.tagging_cooldown = self._min(
                    (player.tagging_cooldown + self.sim_speedup_factor * self.tau),
                    self.tagging_cooldown,
                )
                self.state["agent_tagging_cooldown"][i] = player.tagging_cooldown

        self.flag_collision_bool = np.zeros(self.num_agents)

        if self.action_type == "discrete":
            action_dict = self._to_speed_heading(raw_action_dict)
        elif self.action_type == "continuous":
            action_dict = raw_action_dict
        else:
            raise ValueError(
                f"action_type must be either 'discrete' or 'continuous'. Got '{self.action_type}' instead."
            )

        if self.render_mode:
            for _i in range(self.num_renders_per_step):
                for _j in range(self.sim_speedup_factor):
                    self._move_agents(action_dict, 1 / self.render_fps)
                if self.lidar_obs:
                    self._update_lidar()
                self._render()
        else:
            for _ in range(self.sim_speedup_factor):
                self._move_agents(action_dict, self.tau)
            if self.lidar_obs:
                self._update_lidar()

        # set the time
        self.current_time += self.sim_speedup_factor * self.tau

        # agent and flag capture checks and more
        self._check_oob_vectorized() if self.team_size >= 40 else self._check_oob()
        (
            self._check_pickup_flags_vectorized()
            if self.team_size >= 40
            else self._check_pickup_flags()
        )
        (
            self._check_agent_made_tag_vectorized()
            if self.team_size >= 10
            else self._check_agent_made_tag()
        )
        self._check_flag_captures()
        self._check_untag_vectorized() if self.team_size >= 10 else self._check_untag()
        self._set_dones()
        self._get_dist_bearing_to_obstacles()

        if self.lidar_obs:
            for team in self.agents_of_team:
                for agent_id, player in self.players.items():
                    if player.team == team:
                        detection_class_name = "teammate"
                    else:
                        detection_class_name = "opponent"
                    if player.is_tagged:
                        detection_class_name += "_is_tagged"
                    elif player.has_flag:
                        detection_class_name += "_has_flag"

                    self.obj_ray_detection_states[team][
                        self.ray_int_label_map[f"agent_{agent_id}"]
                    ] = LIDAR_DETECTION_CLASS_MAP[detection_class_name]

        if self.message and self.render_mode:
            print(self.message)

        # Rewards
        rewards = {
            agent_id: self.compute_rewards(agent_id) for agent_id in self.players
        }

        # Observations
        for agent_id in raw_action_dict:
            self.state["obs_hist_buffer"][agent_id][1:] = self.state["obs_hist_buffer"][
                agent_id
            ][:-1]
            self.state["obs_hist_buffer"][agent_id][0] = self.state_to_obs(
                agent_id, self.normalize
            )

        if self.hist_len > 1:
            obs = {
                agent_id: self.state["obs_hist_buffer"][agent_id][self.hist_buffer_inds]
                for agent_id in self.players
            }
        else:
            obs = {
                agent_id: self.state["obs_hist_buffer"][agent_id][0]
                for agent_id in self.players
            }

        # Dones
        terminated = False
        truncated = False

        if self.dones["__all__"]:
            if self.dones["blue"] or self.dones["red"]:
                terminated = True
            else:
                truncated = True

        terminated = {agent: terminated for agent in raw_action_dict}
        truncated = {agent: truncated for agent in raw_action_dict}

        # Info
        self.state["global_state_hist_buffer"][1:] = self.state[
            "global_state_hist_buffer"
        ][:-1]
        self.state["global_state_hist_buffer"][0] = self.state_to_global_obs(
            self.normalize
        )

        if self.hist_len > 1:
            info = {
                "global_state": self.state["global_state_hist_buffer"][
                    self.hist_buffer_inds
                ]
            }
        else:
            info = {"global_state": self.state["global_state_hist_buffer"][0]}

        return obs, rewards, terminated, truncated, info

    def _move_agents(self, action_dict, dt):
        """Moves agents in the space according to the specified speed/heading in `action_dict`."""
        for i, player in enumerate(self.players.values()):

            # Get current position and flag location
            pos_x = player.pos[0]
            pos_y = player.pos[1]
            flag_loc = self.flags[int(player.team)].home

            # Check if agent is on their own side
            player.on_own_side = self._check_on_sides(player.pos, player.team)
            self.state["agent_on_sides"][i] = player.on_own_side

            # convert desired_speed   and  desired_heading to
            #         desired_thrust  and  desired_rudder
            # requested heading is relative so it directly maps to the heading error

            # If agent is tagged, drive at max speed towards home
            if player.is_tagged:
                flag_home = self.flags[int(player.team)].home
                _, heading_error = mag_bearing_to(player.pos, flag_home, player.heading)
                desired_speed = self.config_dict["max_speed"]

            # If agent is out of bounds, drive back in bounds at low speed
            elif self.state["agent_oob"][i]:
                ## compute the closest point in the env
                closest_point_x = max(
                    self.agent_radius, min(self.env_size[0] - self.agent_radius, pos_x)
                )
                closest_point_y = max(
                    self.agent_radius, min(self.env_size[1] - self.agent_radius, pos_y)
                )
                closest_point = [closest_point_x, closest_point_y]
                _, heading_error = mag_bearing_to(
                    player.pos, closest_point, player.heading
                )
                desired_speed = self.config_dict["max_speed"] * self.oob_speed_frac

            # Else get desired speed and heading from action_dict
            else:
                desired_speed, heading_error = action_dict[player.id]

            # Get new speed, heading, and thrust based on desired speed, desired heading, and agent dynamics
            if self.default_dynamics:
                new_speed, new_heading, new_thrust = heron_move_agents(
                    self, player, desired_speed, heading_error, dt
                )
            elif self.dynamics_dict[i] == "heron":
                new_speed, new_heading, new_thrust = heron_move_agents(
                    self, player, desired_speed, heading_error, dt
                )
            elif self.dynamics_dict[i] == "large_usv":
                new_speed, new_heading, new_thrust = large_usv_move_agents(
                    self, player, desired_speed, heading_error, dt
                )
            elif self.dynamics_dict[i] == "drone":
                new_speed, new_heading, new_thrust = drone_move_agents(
                    self, player, desired_speed, heading_error, dt
                )
            elif self.dynamics_dict[i] == "si":
                new_speed, new_heading, new_thrust = si_move_agents(
                    self, player, desired_speed, heading_error, dt
                )

            vel = mag_heading_to_vec(new_speed, new_heading)

            # If the player hits a boundary, return them to their original starting position and skip
            # to the next agent.
            player_hit_obstacle = detect_collision(
                np.asarray([pos_x, pos_y]), self.agent_radius, self.obstacle_geoms
            )

            if player_hit_obstacle:
                if player.team == Team.RED_TEAM:
                    self.state["tags"][0] += 1
                else:
                    self.state["tags"][1] += 1
                if player.has_flag:
                    # If they have a flag, return the flag to it's home area
                    self.flags[int(not int(player.team))].reset()
                    self.state["flag_taken"][int(not int(player.team))] = 0
                    self.state["agent_has_flag"][i] = 0
                    self.state["flag_locations"][int(not int(player.team))] = np.array(
                        self.flags[int(not int(player.team))].pos
                    )
                    player.has_flag = False

                    if player.team == Team.RED_TEAM:
                        self.state["team_has_flag"][1] = False
                    else:
                        self.state["team_has_flag"][0] = False
                if self.tag_on_collision:
                    self.state["agent_is_tagged"][i] = 1
                    player.is_tagged = True
                player.rotate(copy.deepcopy(player.prev_pos))
                self.state["agent_position"][i] = player.pos
                self.state["prev_agent_position"][i] = player.prev_pos
                self.state["agent_spd_hdg"][i] = [player.speed, player.heading]
                continue

            # Check if agent is in keepout region for their own flag
            ag_dis_2_flag = self.get_distance_between_2_points(
                np.asarray([pos_x, pos_y]), np.asarray(flag_loc)
            )
            if (
                ag_dis_2_flag < self.flag_keepout
                and not self.state["flag_taken"][int(player.team)]
                and self.flag_keepout > 0.0
            ):
                self.flag_collision_bool[player.id] = True

                ag_pos = np.array([pos_x, pos_y])
                ag_pos = rc_intersection(
                    np.array([ag_pos, player.prev_pos]),
                    np.asarray(flag_loc),
                    self.flag_keepout,
                )  # point where agent center first intersected with keepout zone
                ag_vel = reflect_vector(ag_pos, vel, np.asarray(flag_loc))

                crd_ref_angle = get_rot_angle(np.asarray(flag_loc), ag_pos)
                vel_ref = rot2d(ag_vel, -crd_ref_angle)
                vel_ref[1] = (
                    0.0  # convention is that vector pointing from keepout intersection to flag center is y' axis in new reference frame
                )

                vel = rot2d(vel_ref, crd_ref_angle)
                pos_x = ag_pos[0]
                pos_y = ag_pos[1]

            if (not self.gps_env and new_speed > 0.1) or (
                self.gps_env and new_speed / self.meters_per_mercator_xy > 0.1
            ):
                # only rely on vel if speed is large enough to recover heading
                new_speed, new_heading = vec_to_mag_heading(vel)

            # Propagate vehicle position based on new_heading and new_speed
            hdg_rad = math.radians(player.heading)
            new_hdg_rad = math.radians(new_heading)
            avg_speed = (new_speed + player.speed) / 2.0
            if self.gps_env:
                avg_speed = avg_speed / self.meters_per_mercator_xy
            s = math.sin(new_hdg_rad) + math.sin(hdg_rad)
            c = math.cos(new_hdg_rad) + math.cos(hdg_rad)
            avg_hdg = math.atan2(s, c)
            # Note: sine/cos swapped because of the heading / angle difference
            new_ag_pos = [
                pos_x + math.sin(avg_hdg) * avg_speed * dt,
                pos_y + math.cos(avg_hdg) * avg_speed * dt,
            ]

            if player.has_flag:
                flg_idx = not int(player.team)
                self.flags[flg_idx].pos = list(new_ag_pos)
                self.state["flag_locations"][int(flg_idx)] = np.array(
                    self.flags[flg_idx].pos
                )
            player.prev_pos = player.pos
            player.pos = np.asarray(new_ag_pos)
            player.speed = clip(new_speed, 0.0, self.max_speed)
            player.heading = angle180(new_heading)
            player.thrust = new_thrust

            self.state["agent_position"][i] = player.pos
            self.state["prev_agent_position"][i] = player.prev_pos
            self.state["agent_spd_hdg"][i] = [player.speed, player.heading]

    def _check_on_sides(self, pos, team):
        scrim2pos = np.asarray(pos) - self.scrimmage_coords[0]
        cp_sign = np.sign(np.cross(self.scrimmage_vec, scrim2pos))

        return cp_sign == self.on_sides_sign[team] or cp_sign == 0

    def _update_lidar(self):
        ray_int_segments = np.copy(self.ray_int_segments)

        # Valid flag intersection segments mask
        flag_int_seg_mask = np.ones(len(self.ray_int_seg_labels), dtype=bool)
        flag_seg_inds = self.seg_label_type_to_inds["flag"]
        flag_int_seg_mask[flag_seg_inds] = np.repeat(
            np.logical_not(self.state["flag_taken"]), self.n_circle_segments
        )

        # Translate non-static ray intersection geometries (flags and agents)
        ray_int_segments[flag_seg_inds] += np.repeat(
            np.tile(self.state["flag_home"], 2), self.n_circle_segments, axis=0
        )
        agent_seg_inds = self.seg_label_type_to_inds["agent"]
        ray_int_segments[agent_seg_inds] += np.repeat(
            np.tile(self.state["agent_position"], 2), self.n_circle_segments, axis=0
        )
        ray_int_segments = ray_int_segments.reshape(1, -1, 4)

        # Agent rays
        ray_origins = np.expand_dims(self.state["agent_position"], axis=1)
        ray_headings_global = np.deg2rad(
            (
                heading_angle_conversion(self.state["agent_spd_hdg"][:, 1]).reshape(
                    -1, 1
                )
                + self.lidar_ray_headings
            )
            % 360
        )
        ray_vecs = np.array(
            [np.cos(ray_headings_global), np.sin(ray_headings_global)]
        ).transpose(1, 2, 0)
        ray_ends = ray_origins + self.lidar_range * ray_vecs
        ray_segments = np.concatenate(
            (np.full(ray_ends.shape, ray_origins), ray_ends), axis=-1
        )
        ray_segments = ray_segments.reshape(self.num_agents, -1, 1, 4)

        # compute ray intersections
        x1, y1, x2, y2 = (
            ray_segments[..., 0],
            ray_segments[..., 1],
            ray_segments[..., 2],
            ray_segments[..., 3],
        )
        x3, y3, x4, y4 = (
            ray_int_segments[..., 0],
            ray_int_segments[..., 1],
            ray_int_segments[..., 2],
            ray_int_segments[..., 3],
        )

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        intersect_x = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denom
        intersect_y = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denom

        # mask invalid intersections (parallel lines, outside of segment bounds, picked up flags, own agent segments)
        mask = (
            (denom != 0)
            & (intersect_x >= np.minimum(x1, x2) - LINE_INTERSECT_TOL)
            & (intersect_x <= np.maximum(x1, x2) + LINE_INTERSECT_TOL)
            & (intersect_y >= np.minimum(y1, y2) - LINE_INTERSECT_TOL)
            & (intersect_y <= np.maximum(y1, y2) + LINE_INTERSECT_TOL)
            & (intersect_x >= np.minimum(x3, x4) - LINE_INTERSECT_TOL)
            & (intersect_x <= np.maximum(x3, x4) + LINE_INTERSECT_TOL)
            & (intersect_y >= np.minimum(y3, y4) - LINE_INTERSECT_TOL)
            & (intersect_y <= np.maximum(y3, y4) + LINE_INTERSECT_TOL)
            & flag_int_seg_mask
            & self.agent_int_seg_mask
        )

        intersect_x = np.where(
            mask, intersect_x, -self.env_diag
        )  # a coordinate out of bounds and far away
        intersect_y = np.where(
            mask, intersect_y, -self.env_diag
        )  # a coordinate out of bounds and far away
        intersections = np.stack(
            (intersect_x.flatten(), intersect_y.flatten()), axis=-1
        ).reshape(intersect_x.shape + (2,))

        # determine lidar ray readings
        ray_origins = np.expand_dims(ray_origins, axis=1)
        intersection_dists = np.linalg.norm(intersections - ray_origins, axis=-1)
        ray_int_inds = np.argmin(intersection_dists, axis=-1)

        ray_int_labels = self.ray_int_seg_labels[ray_int_inds]
        ray_intersections = intersections[
            np.arange(self.num_agents).reshape(-1, 1),
            np.arange(self.num_lidar_rays),
            ray_int_inds,
        ]
        ray_int_dists = intersection_dists[
            np.arange(self.num_agents).reshape(-1, 1),
            np.arange(self.num_lidar_rays),
            ray_int_inds,
        ]

        # correct lidar ray readings for which nothing was detected
        invalid_ray_ints = np.where(np.all(np.logical_not(mask), axis=-1))
        ray_int_labels[invalid_ray_ints] = self.ray_int_label_map["nothing"]
        ray_intersections[invalid_ray_ints] = ray_ends[invalid_ray_ints]
        ray_int_dists[invalid_ray_ints] = self.lidar_range

        # save lidar readings
        for i, agent_id in enumerate(self.players):
            self.state["lidar_labels"][agent_id] = ray_int_labels[i]
            self.state["lidar_ends"][agent_id] = ray_intersections[i]
            self.state["lidar_distances"][agent_id] = ray_int_dists[i]

    def _check_oob_vectorized(self):
        """Checks if players are out of bounds and updates their states accordingly."""
        agent_positions = self.state["agent_position"]
        agent_oob = 1 - (
            (self.agent_radius <= agent_positions[:, 0])
            * (agent_positions[:, 0] <= self.env_size[0] - self.agent_radius)
            * (self.agent_radius <= agent_positions[:, 1])
            * (agent_positions[:, 1] <= self.env_size[1] - self.agent_radius)
        )

        self.state["agent_oob"] = agent_oob
        num_blue_agent_oob = np.sum(agent_oob[0 : self.team_size])
        num_red_agent_oob = np.sum(agent_oob[self.team_size :])
        if num_blue_agent_oob + num_red_agent_oob == 0:  # no one oob
            return

        self.state["tags"] += num_red_agent_oob
        self.state["tags"] += num_blue_agent_oob

        agent_has_flag = self.state["agent_has_flag"]
        agent_oob_inds = np.where(agent_oob == True)
        blue_flag_tagged_ind = np.where(
            agent_oob[self.team_size :] * agent_has_flag[self.team_size :]
        )[0]
        red_flag_tagged_ind = np.where(
            agent_oob[: self.team_size] * agent_has_flag[0 : self.team_size]
        )[0]

        if blue_flag_tagged_ind.size != 0:
            self.flags[0].reset()
            self.players[blue_flag_tagged_ind[0] + self.team_size].has_flag = False
            self.state["team_has_flag"][1] = 0
            self.state["flag_taken"][0] = 0
            self.state["agent_has_flag"][blue_flag_tagged_ind[0] + self.team_size] = 0
            self.state["flag_locations"][0] = np.array(self.flags[0].pos)
        if red_flag_tagged_ind.size != 0:
            self.flags[1].reset()
            self.players[red_flag_tagged_ind[0]].has_flag = False
            self.state["team_has_flag"][0] = 0
            self.state["flag_taken"][1] = 0
            self.state["agent_has_flag"][red_flag_tagged_ind[0]] = 0
            self.state["flag_locations"][1] = np.array(self.flags[1].pos)

        if self.tag_on_collision:
            self.state["agent_is_tagged"][agent_oob_inds] = 1
            for agent_oob_ind in agent_oob_inds:
                self.players[agent_oob_ind].is_tagged = True

    def _check_oob(self):
        """Checks if players are out of bounds and updates their states accordingly."""
        for i, player in enumerate(self.players.values()):
            pos_x = player.pos[0]
            pos_y = player.pos[1]
            if not (
                (self.agent_radius <= pos_x <= self.env_size[0] - self.agent_radius)
                and (self.agent_radius <= pos_y <= self.env_size[1] - self.agent_radius)
            ):
                if player.team == Team.RED_TEAM:
                    self.state["tags"][0] += 1
                else:
                    self.state["tags"][1] += 1
                if player.has_flag:
                    # If they have a flag, return the flag to it's home area
                    self.flags[int(not int(player.team))].reset()
                    self.state["flag_taken"][int(not int(player.team))] = 0
                    self.state["agent_has_flag"][i] = 0
                    self.state["flag_locations"][int(not int(player.team))] = np.array(
                        self.flags[int(not int(player.team))].pos
                    )
                    player.has_flag = False

                    if player.team == Team.RED_TEAM:
                        self.state["team_has_flag"][1] = False
                    else:
                        self.state["team_has_flag"][0] = False
                self.state["agent_oob"][i] = 1
                if self.tag_on_collision:
                    self.state["agent_is_tagged"][i] = 1
                    player.is_tagged = True
            else:
                self.state["agent_oob"][i] = 0

    def _check_pickup_flags_vectorized(self):
        """Updates player states if they picked up the flag."""
        agent_positions = self.state["agent_position"]
        agent_teams = np.concatenate(
            (np.zeros(self.team_size), np.ones(self.team_size))
        )

        agent_is_tagged = self.state["agent_is_tagged"]
        agent_oob = self.state["agent_oob"]
        agent_has_flag = self.state["agent_has_flag"]
        agent_on_sides = self.state["agent_on_sides"]
        flag_locations = self.state["flag_locations"]
        flag_taken = self.state["flag_taken"]

        dists_to_flag = np.linalg.norm(
            agent_positions - flag_locations[1 - agent_teams.astype(int)], axis=1
        )
        agent_flag_pickups = (
            (dists_to_flag < self.catch_radius)
            & (flag_taken[1 - agent_teams.astype(int)] == 0)
            & (agent_is_tagged == 0)
            & (agent_oob == 0)
            & (agent_has_flag == 0)
            & (agent_on_sides == 0)
        )

        ## add the np.cumprod thing here to ensure only one agent picks up the flag at a time
        agent_flag_pickups[1 : self.team_size] = (
            np.cumprod(1 - agent_flag_pickups[0 : self.team_size - 1])
            * agent_flag_pickups[1 : self.team_size]
        ).astype(bool)
        agent_flag_pickups[self.team_size + 1 :] = (
            np.cumprod(1 - agent_flag_pickups[self.team_size : -1])
            * agent_flag_pickups[self.team_size + 1 :]
        ).astype(bool)

        blue_flag_taken = np.sum(agent_flag_pickups[self.team_size :]) > 0
        red_flag_taken = np.sum(agent_flag_pickups[0 : self.team_size]) > 0
        if not blue_flag_taken and not red_flag_taken:
            return

        flag_taken_this_step = np.array([blue_flag_taken, red_flag_taken])
        team_flag_pickup_this_step = np.array([red_flag_taken, blue_flag_taken])

        # update state
        for i, player in enumerate(self.players.values()):
            if agent_flag_pickups[i] == 1:
                player.has_flag = True
        self.state["agent_has_flag"] = (
            agent_has_flag.astype(bool) | agent_flag_pickups
        ) + 0  # the +0 casts the arrays from bool to int
        self.state["flag_taken"] = (flag_taken.astype(bool) | flag_taken_this_step) + 0
        self.state["team_has_flag"] = (
            self.state["team_has_flag"].astype(bool) | team_flag_pickup_this_step
        ) + 0

        self.state["grabs"][0] = self.state["grabs"][0] + np.sum(
            agent_flag_pickups[: self.team_size]
        )
        self.state["grabs"][1] = self.state["grabs"][1] + np.sum(
            agent_flag_pickups[self.team_size :]
        )

    def _check_pickup_flags(self):
        """Updates player states if they picked up the flag."""
        for i, player in enumerate(self.players.values()):
            team = int(player.team)
            other_team = int(not team)
            if (
                not (player.has_flag or self.state["flag_taken"][other_team])
                and (not player.is_tagged)
                and (not player.on_own_side)
            ):
                flag_pos = self.flags[other_team].pos
                distance_to_flag = self.get_distance_between_2_points(
                    player.pos, flag_pos
                )

                if distance_to_flag < self.catch_radius:
                    player.has_flag = True
                    self.state["flag_taken"][other_team] = 1
                    self.state["agent_has_flag"][i] = 1
                    if player.team == Team.BLUE_TEAM:
                        self.state["grabs"][0] += 1
                        self.state["team_has_flag"][0] = True
                    else:
                        self.state["grabs"][1] += 1
                        self.state["team_has_flag"][1] = True

    def _check_untag_vectorized(self):
        """Untags the player if they return to their own flag."""
        agent_positions = self.state["agent_position"]
        blue_distance_to_flag = np.linalg.norm(
            agent_positions[0 : self.team_size] - self.flags[0].home
        )
        red_distance_to_flag = np.linalg.norm(
            agent_positions[self.team_size :] - self.flags[1].home
        )

        blue_agents_untag_ids = np.where(blue_distance_to_flag < self.catch_radius)[0]
        red_agents_untag_ids = (
            np.where(red_distance_to_flag < self.catch_radius)[0] + self.team_size
        )

        self.state["agent_is_tagged"][blue_agents_untag_ids] = 0
        self.state["agent_is_tagged"][red_agents_untag_ids] = 0

        for blue_agent_untag_id in blue_agents_untag_ids:
            self.players[blue_agent_untag_id].is_tagged = False
        for red_agent_untag_id in red_agents_untag_ids:
            self.players[red_agent_untag_id].is_tagged = False

    def _check_untag(self):
        """Untags the player if they return to their own flag."""
        for i, player in enumerate(self.players.values()):
            team = int(player.team)
            flag_home = self.flags[team].home
            distance_to_flag = self.get_distance_between_2_points(player.pos, flag_home)
            if distance_to_flag < self.catch_radius and player.is_tagged:
                self.state["agent_is_tagged"][i] = 0
                player.is_tagged = False

    def _check_agent_made_tag_vectorized(self):
        """Updates player states if they tagged another player."""
        agent_positions = self.state["agent_position"]
        agent_to_agent_diffs = agent_positions - np.expand_dims(agent_positions, 1)
        agent_to_agent_dists = np.linalg.norm(agent_to_agent_diffs, axis=2)

        agent_tagging_cooldowns = self.state["agent_tagging_cooldown"].reshape(
            self.num_agents, 1
        )
        agent_is_tagged = self.state["agent_is_tagged"].reshape(self.num_agents, 1)
        agent_oob = self.state["agent_oob"].reshape(self.num_agents, 1)
        agent_has_flag = self.state["agent_has_flag"].reshape(self.num_agents, 1)
        agent_on_sides = self.state["agent_on_sides"].reshape(self.num_agents, 1)

        blue_agent_got_tagged = (
            (
                agent_to_agent_dists[0 : self.team_size, self.team_size :]
                < self.catch_radius
            )
            * (agent_is_tagged[0 : self.team_size] == 0)
            * (agent_on_sides[0 : self.team_size] == 0)
            * (agent_tagging_cooldowns[self.team_size :].T == self.tagging_cooldown)
            * (agent_is_tagged[self.team_size :].T == 0)
            * (agent_on_sides[self.team_size :].T == 1)
            * (agent_oob[self.team_size :].T == 0)
        )
        red_agent_got_tagged = (
            (
                agent_to_agent_dists[self.team_size :, 0 : self.team_size]
                < self.catch_radius
            )
            * (agent_is_tagged[self.team_size :] == 0)
            * (agent_on_sides[self.team_size :] == 0)
            * (agent_tagging_cooldowns[0 : self.team_size].T == self.tagging_cooldown)
            * (agent_is_tagged[0 : self.team_size].T == 0)
            * (agent_on_sides[0 : self.team_size].T == 1)
            * (agent_oob[0 : self.team_size].T == 0)
        )

        if not (
            np.any(blue_agent_got_tagged) or np.any(red_agent_got_tagged)
        ):  # if no tags occured, we don't need to update any of these states
            return

        for i in range(
            self.team_size
        ):  # make sure that only one agent can tag another agent and that each agent can only be tagged by one agent
            blue_agent_got_tagged[i, 1:] = (
                np.cumprod(1 - blue_agent_got_tagged[i, :-1])
                * blue_agent_got_tagged[i, 1:]
            ).astype(bool)
            blue_agent_got_tagged[1:, i] = (
                np.cumprod(1 - blue_agent_got_tagged[:-1, i])
                * blue_agent_got_tagged[1:, i]
            ).astype(bool)

            red_agent_got_tagged[i, 1:] = (
                np.cumprod(1 - red_agent_got_tagged[i, :-1])
                * red_agent_got_tagged[i, 1:]
            ).astype(bool)
            red_agent_got_tagged[1:, i] = (
                np.cumprod(1 - red_agent_got_tagged[:-1, i])
                * red_agent_got_tagged[1:, i]
            ).astype(bool)

        blue_agents_tagged = np.sum(blue_agent_got_tagged, axis=1)
        red_agents_tagged = np.sum(red_agent_got_tagged, axis=1)
        all_agents_tagged = np.concatenate((blue_agents_tagged, red_agents_tagged))

        blue_agents_tagger = np.sum(red_agent_got_tagged, axis=0)
        red_agents_tagger = np.sum(blue_agent_got_tagged, axis=0)
        all_agents_tagger = np.concatenate((blue_agents_tagger, red_agents_tagger))

        assert np.sum(blue_agents_tagged) == np.sum(red_agents_tagger) and np.sum(
            red_agents_tagged
        ) == np.sum(blue_agents_tagger)

        self.state["tags"][0] += np.sum(red_agents_tagged)
        self.state["tags"][1] += np.sum(blue_agents_tagged)

        self.state["agent_is_tagged"] = (
            self.state["agent_is_tagged"] | all_agents_tagged
        ).astype(int)

        blue_flag_tagged = np.any(red_agents_tagged * agent_has_flag[self.team_size :])
        red_flag_tagged = np.any(
            blue_agents_tagged * agent_has_flag[0 : self.team_size]
        )

        agent_is_tagged_ids = np.where(all_agents_tagged == 1)[0]
        agent_tagger_ids = np.where(all_agents_tagger == 1)[0]

        flags_tagged = np.array([blue_flag_tagged, red_flag_tagged])
        flags_tagged_inds = np.where(flags_tagged == 1)[0]

        self.state["team_has_flag"] = self.state["team_has_flag"] * (
            1 - np.flip(flags_tagged)
        )
        self.state["flag_taken"] = self.state["flag_taken"] * (1 - flags_tagged)
        self.state["agent_has_flag"][agent_is_tagged_ids] = 0

        for flags_tagged_ind in flags_tagged_inds:
            self.state["flag_locations"][flags_tagged_ind] = np.array(
                self.flags[flags_tagged_ind].pos
            )
        [self.flags[flags_tagged_ind].reset() for flags_tagged_ind in flags_tagged_inds]

        self.state["agent_tagging_cooldown"][agent_tagger_ids] = 0.0
        for agent_tagger_id in agent_tagger_ids:
            self.players[agent_tagger_id].tagging_cooldown = 0.0

        for agent_is_tagged_id in agent_is_tagged_ids:
            self.players[agent_is_tagged_id].is_tagged = True
            self.players[agent_is_tagged_id].has_flag = False

            # how to figure out which agent tagged this agent?
            if agent_is_tagged_id < self.team_size:  # blue agent
                agent_tagger_id = (
                    np.where(blue_agent_got_tagged[agent_is_tagged_id][:] == 1)[0][0]
                    + self.team_size
                )
            else:
                agent_tagger_id = np.where(
                    red_agent_got_tagged[agent_is_tagged_id - self.team_size][:] == 1
                )[0][0]
            self.state["agent_made_tag"][agent_tagger_id] = agent_is_tagged_id

    def _check_agent_made_tag(self):
        """Updates player states if they tagged another player."""

        self.state["agent_made_tag"] = [None] * self.num_agents
        for i, player in enumerate(self.players.values()):
            # Only continue logic check if player tagged someone if it's on its own side and is untagged.
            if (
                player.on_own_side
                and (player.tagging_cooldown == self.tagging_cooldown)
                and not player.is_tagged
            ):
                for j, other_player in enumerate(self.players.values()):
                    o_team = int(other_player.team)
                    # Only do the rest if the other player is NOT on sides and they are not on the same team.
                    if (
                        not other_player.is_tagged
                        and not other_player.on_own_side
                        and other_player.team != player.team
                    ):
                        dist_between_agents = self.get_distance_between_2_points(
                            player.pos, other_player.pos
                        )
                        if (
                            dist_between_agents > 0.0
                            and dist_between_agents < self.catch_radius
                        ):
                            if other_player.team == Team.RED_TEAM:
                                self.state["tags"][0] += 1
                            else:
                                self.state["tags"][1] += 1
                            self.state["agent_is_tagged"][j] = 1
                            other_player.is_tagged = True
                            self.state["agent_made_tag"][i] = other_player.id
                            # If we get here, then `player` tagged `other_player` and we need to reset `other_player`

                            if other_player.has_flag:
                                # If the player with the flag was tagged, the flag also needs to be reset.
                                if other_player.team == Team.BLUE_TEAM:
                                    self.state["team_has_flag"][0] = False
                                else:
                                    self.state["team_has_flag"][1] = False
                                self.state["flag_taken"][int(player.team)] = 0
                                self.state["agent_has_flag"][j] = 0
                                self.flags[int(player.team)].reset()
                                self.state["flag_locations"][int(player.team)] = (
                                    np.array(self.flags[int(player.team)].pos)
                                )
                                other_player.has_flag = False

                            # Set players tagging cooldown
                            player.tagging_cooldown = 0.0
                            self.state["agent_tagging_cooldown"][i] = 0.0

                            # Break loop (should not be allowed to tag again during current timestep)
                            break

    def _check_flag_captures(self):
        """Updates states if a player captured a flag."""
        # these are false except at the instance that the flag is captured
        self.blue_team_flag_capture = False
        self.red_team_flag_capture = False
        for i, player in enumerate(self.players.values()):
            if player.on_own_side and player.has_flag:
                if player.team == Team.BLUE_TEAM:
                    self.blue_team_flag_capture = True
                    self.state["captures"][0] += 1
                    self.state["team_has_flag"][0] = False
                else:
                    self.red_team_flag_capture = True
                    self.state["captures"][1] += 1
                    self.state["team_has_flag"][1] = False
                player.has_flag = False
                scored_flag = self.flags[not int(player.team)]
                self.state["flag_taken"][int(scored_flag.team)] = False
                self.state["agent_has_flag"][i] = 0
                scored_flag.reset()
                self.state["flag_locations"][int(scored_flag.team)] = np.array(
                    self.flags[int(scored_flag.team)].pos
                )

    def get_full_state_info(self):
        """Return the full state."""
        return self.state

    def current_iter_value(self, iter_value):
        """Returns the learning iteration."""
        self.learning_iteration = iter_value

    def set_config_values(self, config_dict):
        """
        Sets initial configuration parameters for the environment.

        Args:
            config_dict: The provided configuration. If a key is missing, it is replaced
            with the standard configuration value.
        """
        ### Set Variables from Configuration Dictionary ###
        # Geometry parameters
        self.gps_env = config_dict.get("gps_env", config_dict_std["gps_env"])
        self.topo_contour_eps = config_dict.get(
            "topo_contour_eps", config_dict_std["topo_contour_eps"]
        )

        # MOOS dynamics parameters
        self.default_dynamics = config_dict.get(
            "default_dynamics", config_dict_std["default_dynamics"]
        )
        self.dynamics_dict = config_dict.get(
            "dynamics_dict", config_dict_std["dynamics_dict"]
        )
        self.speed_factor = config_dict.get(
            "speed_factor", config_dict_std["speed_factor"]
        )
        self.thrust_map = config_dict.get("thrust_map", config_dict_std["thrust_map"])
        self.max_thrust = config_dict.get("max_thrust", config_dict_std["max_thrust"])
        self.max_rudder = config_dict.get("max_rudder", config_dict_std["max_rudder"])
        self.turn_loss = config_dict.get("turn_loss", config_dict_std["turn_loss"])
        self.turn_rate = config_dict.get("turn_rate", config_dict_std["turn_rate"])
        self.max_acc = config_dict.get("max_acc", config_dict_std["max_acc"])
        self.max_dec = config_dict.get("max_dec", config_dict_std["max_dec"])
        self.oob_speed_frac = config_dict.get(
            "oob_speed_frac", config_dict_std["oob_speed_frac"]
        )
        self.large_usv_speed_factor = config_dict.get(
            "speed_factor", config_dict_std["speed_factor"]
        )
        self.large_usv_max_speed = config_dict.get(
            "large_usv_max_speed", config_dict_std["large_usv_max_speed"]
        )
        self.large_usv_thrust_map = config_dict.get(
            "large_usv_thrust_map", config_dict_std["large_usv_thrust_map"]
        )
        self.large_usv_max_thrust = config_dict.get(
            "large_usv_max_thrust", config_dict_std["large_usv_max_thrust"]
        )
        self.large_usv_max_rudder = config_dict.get(
            "large_usv_max_rudder", config_dict_std["large_usv_max_rudder"]
        )
        self.large_usv_turn_loss = config_dict.get(
            "large_usv_turn_loss", config_dict_std["large_usv_turn_loss"]
        )
        self.large_usv_turn_rate = config_dict.get(
            "large_usv_turn_rate", config_dict_std["large_usv_turn_rate"]
        )
        self.large_usv_max_acc = config_dict.get(
            "large_usv_max_acc", config_dict_std["large_usv_max_acc"]
        )
        self.large_usv_max_dec = config_dict.get(
            "large_usv_max_dec", config_dict_std["large_usv_max_dec"]
        )
        self.large_usv_oob_speed_frac = config_dict.get(
            "large_usv_oob_speed_frac", config_dict_std["large_usv_oob_speed_frac"]
        )
        self.action_type = config_dict.get(
            "action_type", config_dict_std["action_type"]
        )
        self.si_max_speed = config_dict.get(
            "si_max_speed", config_dict_std["si_max_speed"]
        )
        self.si_max_omega = config_dict.get(
            "si_max_omega", config_dict_std["si_max_omega"]
        )

        # Simulation parameters
        self.tau = config_dict.get("tau", config_dict_std["tau"])
        self.sim_speedup_factor = config_dict.get(
            "sim_speedup_factor", config_dict_std["sim_speedup_factor"]
        )

        # Game parameters
        self.max_score = config_dict.get("max_score", config_dict_std["max_score"])
        self.max_time = config_dict.get("max_time", config_dict_std["max_time"])
        self.tagging_cooldown = config_dict.get(
            "tagging_cooldown", config_dict_std["tagging_cooldown"]
        )
        self.tag_on_collision = config_dict.get(
            "tag_on_collision", config_dict_std["tag_on_collision"]
        )

        # Observation and state parameters
        self.normalize = config_dict.get("normalize", config_dict_std["normalize"])
        self.lidar_obs = config_dict.get("lidar_obs", config_dict_std["lidar_obs"])
        self.num_lidar_rays = config_dict.get(
            "num_lidar_rays", config_dict_std["num_lidar_rays"]
        )
        self.short_hist_length = config_dict.get(
            "short_hist_length", config_dict_std["short_hist_length"]
        )
        self.short_hist_interval = config_dict.get(
            "short_hist_interval", config_dict_std["short_hist_interval"]
        )
        self.long_hist_length = config_dict.get(
            "long_hist_length", config_dict_std["long_hist_length"]
        )
        self.long_hist_interval = config_dict.get(
            "long_hist_interval", config_dict_std["long_hist_interval"]
        )

        # Rendering parameters
        self.render_fps = config_dict.get("render_fps", config_dict_std["render_fps"])
        self.screen_frac = config_dict.get(
            "screen_frac", config_dict_std["screen_frac"]
        )
        self.arena_buffer_frac = config_dict.get(
            "arena_buffer_frac", config_dict_std["arena_buffer_frac"]
        )
        self.render_ids = config_dict.get(
            "render_agent_ids", config_dict_std["render_agent_ids"]
        )
        self.render_field_points = config_dict.get(
            "render_field_points", config_dict_std["render_field_points"]
        )
        self.render_traj_mode = config_dict.get(
            "render_traj_mode", config_dict_std["render_traj_mode"]
        )
        self.render_traj_freq = config_dict.get(
            "render_traj_freq", config_dict_std["render_traj_freq"]
        )
        self.render_traj_cutoff = config_dict.get(
            "render_traj_cutoff", config_dict_std["render_traj_cutoff"]
        )
        self.render_lidar_mode = config_dict.get(
            "render_lidar_mode", config_dict_std["render_lidar_mode"]
        )
        self.render_saving = config_dict.get(
            "render_saving", config_dict_std["render_saving"]
        )
        self.render_transparency_alpha = config_dict.get(
            "render_transparency_alpha", config_dict_std["render_transparency_alpha"]
        )

        # agent spawn parameters
        self.default_init = config_dict.get(
            "default_init", config_dict_std["default_init"]
        )

        # Miscellaneous parameters
        if config_dict.get(
            "suppress_numpy_warnings", config_dict_std["suppress_numpy_warnings"]
        ):
            # Suppress numpy warnings to avoid printing out extra stuff to the console
            np.seterr(all="ignore")

        ### Environment History ###
        short_hist_buffer_inds = np.arange(
            0,
            self.short_hist_length * self.short_hist_interval,
            self.short_hist_interval,
        )
        long_hist_buffer_inds = np.arange(
            0, self.long_hist_length * self.long_hist_interval, self.long_hist_interval
        )
        self.hist_buffer_inds = np.unique(
            np.concatenate((short_hist_buffer_inds, long_hist_buffer_inds))
        )  # indices of history buffer corresponding to history entries

        self.hist_len = len(self.hist_buffer_inds)
        self.hist_buffer_len = self.hist_buffer_inds[-1] + 1

        short_hist_oldest_timestep = (
            self.short_hist_length * self.short_hist_interval - self.short_hist_interval
        )
        long_hist_oldest_timestep = (
            self.long_hist_length * self.long_hist_interval - self.long_hist_interval
        )
        if short_hist_oldest_timestep > long_hist_oldest_timestep:
            raise Warning(
                f"The short term history contains older timestep (-{short_hist_oldest_timestep}) than the long term history (-{long_hist_oldest_timestep})."
            )

        ### Environment Geometry Construction ###
        # Basic environment features
        env_bounds = config_dict.get("env_bounds", config_dict_std["env_bounds"])
        env_bounds_unit = config_dict.get(
            "env_bounds_unit", config_dict_std["env_bounds_unit"]
        )

        flag_homes = {}
        flag_homes[Team.BLUE_TEAM] = config_dict.get(
            "blue_flag_home", config_dict_std["blue_flag_home"]
        )
        flag_homes[Team.RED_TEAM] = config_dict.get(
            "red_flag_home", config_dict_std["red_flag_home"]
        )
        flag_homes_unit = config_dict.get(
            "flag_homes_unit", config_dict_std["flag_homes_unit"]
        )

        scrimmage_coords = config_dict.get(
            "scrimmage_coords", config_dict_std["scrimmage_coords"]
        )
        scrimmage_coords_unit = config_dict.get(
            "scrimmage_coords_unit", config_dict_std["scrimmage_coords_unit"]
        )

        agent_radius = config_dict.get("agent_radius", config_dict_std["agent_radius"])
        flag_radius = config_dict.get("flag_radius", config_dict_std["flag_radius"])
        flag_keepout = config_dict.get("flag_keepout", config_dict_std["flag_keepout"])
        catch_radius = config_dict.get("catch_radius", config_dict_std["catch_radius"])
        max_speed = config_dict.get("max_speed", config_dict_std["max_speed"])
        lidar_range = config_dict.get("lidar_range", config_dict_std["lidar_range"])

        self._build_env_geom(
            env_bounds=env_bounds,
            flag_homes=flag_homes,
            scrimmage_coords=scrimmage_coords,
            env_bounds_unit=env_bounds_unit,
            flag_homes_unit=flag_homes_unit,
            scrimmage_coords_unit=scrimmage_coords_unit,
            agent_radius=agent_radius,
            flag_radius=flag_radius,
            flag_keepout=flag_keepout,
            catch_radius=catch_radius,
            max_speed=max_speed,
            lidar_range=lidar_range,
        )

        # Scale the aquaticus point field by env size
        if not self.gps_env:
            for k in self.config_dict["aquaticus_field_points"]:
                self.config_dict["aquaticus_field_points"][k][0] = (
                    self.config_dict["aquaticus_field_points"][k][0] * self.env_size[0]
                )
                self.config_dict["aquaticus_field_points"][k][1] = (
                    self.config_dict["aquaticus_field_points"][k][1] * self.env_size[1]
                )

        # Environment corners
        self.env_ll = np.array([0.0, 0.0])
        self.env_lr = np.array([self.env_size[0], 0.0])
        self.env_ur = np.array(self.env_size)
        self.env_ul = np.array([0.0, self.env_size[1]])
        self.env_vertices = np.array(
            [self.env_ll, self.env_lr, self.env_ur, self.env_ul]
        )
        # ll = lower left, lr = lower right
        # ul = upper left, ur = upper right

        # Warnings
        if self.default_init and self.gps_env:
            raise Warning(
                "Default init only applies to standard (non-gps) environment."
            )

        ### Environment Rendering ###
        if self.render_mode:
            # pygame orientation vector
            self.PYGAME_UP = Vector2((0.0, 1.0))

            # pygame screen size
            arena_buffer = self.arena_buffer_frac * self.env_diag

            max_screen_size = get_screen_res()
            arena_aspect_ratio = (self.env_size[0] + 2 * arena_buffer) / (
                self.env_size[1] + 2 * arena_buffer
            )
            width_based_height = max_screen_size[0] / arena_aspect_ratio

            if width_based_height <= max_screen_size[1]:
                max_pygame_screen_width = max_screen_size[0]
            else:
                height_based_width = max_screen_size[1] * arena_aspect_ratio
                max_pygame_screen_width = int(height_based_width)

            self.pixel_size = (self.screen_frac * max_pygame_screen_width) / (
                self.env_size[0] + 2 * arena_buffer
            )
            self.screen_width = round(
                (self.env_size[0] + 2 * arena_buffer) * self.pixel_size
            )
            self.screen_height = round(
                (self.env_size[1] + 2 * arena_buffer) * self.pixel_size
            )

            # environemnt element sizes in pixels
            self.arena_width, self.arena_height = self.pixel_size * self.env_size
            self.arena_buffer = self.pixel_size * arena_buffer
            self.boundary_width = 2  # pixels
            self.a2a_line_width = 5  # pixels
            self.flag_render_radius = np.clip(
                self.flag_radius * self.pixel_size, 10, None
            )  # pixels
            self.agent_render_radius = np.clip(
                self.agent_radius * self.pixel_size, 15, None
            )  # pixels

            # miscellaneous
            self.num_renders_per_step = int(self.render_fps * self.tau)
            self.render_boundary_rect = True  # standard rectangular boundary

            # check that time between frames (1/render_fps) is not larger than timestep (tau)
            frame_rate_err_msg = (
                "Specified frame rate ({}) creates time intervals between frames larger"
                " than specified timestep ({})".format(self.render_fps, self.tau)
            )
            assert 1 / self.render_fps <= self.tau, frame_rate_err_msg

            # check that time warp is an integer >= 1
            if self.sim_speedup_factor < 1:
                print(
                    "Warning: sim_speedup_factor must be an integer >= 1! Defaulting to 1."
                )
                self.sim_speedup_factor = 1

            if type(self.sim_speedup_factor) != int:
                self.sim_speedup_factor = int(np.round(self.sim_speedup_factor))
                print(
                    f"Warning: Converted sim_speedup_factor to integer: {self.sim_speedup_factor}"
                )

            # check that render_saving is only True if environment is being rendered
            if self.render_saving:
                assert (
                    self.render_mode is not None
                ), "Render_mode cannot be None to record video."

    def set_geom_config(self, config_dict):
        self.n_circle_segments = config_dict.get(
            "n_circle_segments", config_dict_std["n_circle_segments"]
        )
        n_quad_segs = round(self.n_circle_segments / 4)

        # Obstacles
        obstacle_params = config_dict.get("obstacles", config_dict_std["obstacles"])
        border_contour = None

        if self.gps_env:
            border_contour, island_contours, land_mask = self._get_topo_geom()

            if border_contour is not None:
                border_contours = self._border_contour_to_border_obstacles(
                    border_contour
                )
                if obstacle_params is None:
                    obstacle_params = {"polygon": []}
                obstacle_params["polygon"].extend(border_contours)

            if len(island_contours) > 0:
                if obstacle_params is None:
                    obstacle_params = {"polygon": []}
                obstacle_params["polygon"].extend(island_contours)

        self.obstacles = list()
        self.obstacle_geoms = (
            dict()
        )  # arrays with geometric info for obstacles to be used for vectorized calculations
        if obstacle_params is not None and isinstance(obstacle_params, dict):
            circle_obstacles = obstacle_params.get("circle", None)
            if circle_obstacles is not None and isinstance(circle_obstacles, list):
                self.obstacle_geoms["circle"] = []
                for param in circle_obstacles:
                    self.obstacles.append(
                        CircleObstacle(param[0], (param[1][0], param[1][1]))
                    )
                    self.obstacle_geoms["circle"].append(
                        [param[0], param[1][0], param[1][1]]
                    )
                self.obstacle_geoms["circle"] = np.asarray(
                    self.obstacle_geoms["circle"]
                )
            elif circle_obstacles is not None:
                raise TypeError(
                    f"Expected circle obstacle parameters to be a list of tuples, not {type(circle_obstacles)}"
                )
            poly_obstacle = obstacle_params.get("polygon", None)
            if poly_obstacle is not None and isinstance(poly_obstacle, list):
                self.obstacle_geoms["polygon"] = []
                for param in poly_obstacle:
                    converted_param = [(p[0], p[1]) for p in param]
                    self.obstacles.append(PolygonObstacle(converted_param))
                    self.obstacle_geoms["polygon"].extend(
                        [(p, param[(i + 1) % len(param)]) for i, p in enumerate(param)]
                    )
                self.obstacle_geoms["polygon"] = np.asarray(
                    self.obstacle_geoms["polygon"]
                )
            elif poly_obstacle is not None:
                raise TypeError(
                    f"Expected polygon obstacle parameters to be a list of tuples, not {type(poly_obstacle)}"
                )
        elif obstacle_params is not None:
            raise TypeError(
                f"Expected obstacle_params to be None or a dict, not {type(obstacle_params)}"
            )

        # Adjust scrimmage line
        if border_contour is None:
            scrim_int_segs = [
                (self.env_ll, self.env_lr),
                (self.env_lr, self.env_ur),
                (self.env_ur, self.env_ul),
                (self.env_ul, self.env_ll),
            ]
        else:
            scrim_int_segs = [
                (p, border_contour[(i + 1) % len(border_contour)])
                for i, p in enumerate(border_contour)
            ]

        scrim_ints = []
        scrim_seg = LineString(self.scrimmage_coords)
        for seg in scrim_int_segs:
            seg_int = intersection(scrim_seg, LineString(seg))
            if not seg_int.is_empty:
                scrim_ints.append(seg_int.coords[0])

        scrim_ints = np.asarray(scrim_ints)
        scrim_int_dists = np.linalg.norm(
            scrim_ints.reshape(-1, 1, 2) - scrim_ints, axis=-1
        )
        scrim_end_inds = np.unravel_index(
            np.argmax(scrim_int_dists), scrim_int_dists.shape
        )
        self.scrimmage_coords = scrim_ints[scrim_end_inds, :]

        # Ray casting
        if self.lidar_obs:
            self.lidar_ray_headings = np.linspace(
                0,
                (self.num_lidar_rays - 1) * 360 / self.num_lidar_rays,
                self.num_lidar_rays,
            )

            ray_int_label_names = ["nothing", "obstacle"]
            ray_int_label_names.extend([f"flag_{i}" for i, _ in enumerate(self.flags)])
            ray_int_label_names.extend(
                [f"agent_{agent_id}" for agent_id in self.agents]
            )
            self.ray_int_label_map = OrderedDict(
                {label_name: i for i, label_name in enumerate(ray_int_label_names)}
            )

            self.obj_ray_detection_states = {team: [] for team in self.agents_of_team}
            for team in self.agents_of_team:
                for label_name in self.ray_int_label_map:
                    if label_name == "nothing":
                        detection_class = LIDAR_DETECTION_CLASS_MAP["nothing"]
                    elif label_name == "obstacle":
                        detection_class = LIDAR_DETECTION_CLASS_MAP["obstacle"]
                    elif label_name.startswith("flag"):
                        flag_idx = int(label_name[5:])
                        if team == self.flags[flag_idx].team:
                            detection_class = LIDAR_DETECTION_CLASS_MAP["team_flag"]
                        else:
                            detection_class = LIDAR_DETECTION_CLASS_MAP["opponent_flag"]
                    elif label_name.startswith("agent"):
                        agent_id = int(label_name[6:])
                        if agent_id in self.agent_ids_of_team[team]:
                            detection_class = LIDAR_DETECTION_CLASS_MAP["teammate"]
                        else:
                            detection_class = LIDAR_DETECTION_CLASS_MAP["opponent"]
                    else:
                        raise Exception("Unknown lidar detection class.")

                    self.obj_ray_detection_states[team].append(detection_class)
                self.obj_ray_detection_states[team] = np.asarray(
                    self.obj_ray_detection_states[team]
                )

            ray_int_segments = []
            ray_int_seg_labels = []
            self.seg_label_type_to_inds = {
                (label[: label.find("_")] if label[-1].isnumeric() else label): []
                for label in ray_int_label_names
            }

            # boundaries
            ray_int_segments.extend(
                [
                    [*self.env_ll, *self.env_lr],
                    [*self.env_lr, *self.env_ur],
                    [*self.env_ur, *self.env_ul],
                    [*self.env_ul, *self.env_ll],
                ]
            )
            ray_int_seg_labels.extend(4 * [self.ray_int_label_map["obstacle"]])
            self.seg_label_type_to_inds["obstacle"].extend(np.arange(4))

            # obstacles
            obstacle_segments = [
                [*segment[0], *segment[1]]
                for obstacle in self.obstacles
                for segment in self._generate_segments_from_obstacles(
                    obstacle, n_quad_segs
                )
                if not self._segment_on_border(segment)
            ]
            ray_int_seg_labels.extend(
                len(obstacle_segments) * [self.ray_int_label_map["obstacle"]]
            )
            self.seg_label_type_to_inds["obstacle"].extend(
                np.arange(
                    len(ray_int_segments),
                    len(ray_int_segments) + len(obstacle_segments),
                )
            )
            ray_int_segments.extend(obstacle_segments)

            # flags
            for i, _ in enumerate(self.flags):
                vertices = list(
                    Point(0.0, 0.0)
                    .buffer(self.flag_radius, quad_segs=n_quad_segs)
                    .exterior.coords
                )[
                    :-1
                ]  # approximate circle with an octagon
                segments = [
                    [*vertex, *vertices[(i + 1) % len(vertices)]]
                    for i, vertex in enumerate(vertices)
                ]
                ray_int_seg_labels.extend(
                    len(segments) * [self.ray_int_label_map[f"flag_{i}"]]
                )
                self.seg_label_type_to_inds["flag"].extend(
                    np.arange(
                        len(ray_int_segments), len(ray_int_segments) + len(segments)
                    )
                )
                ray_int_segments.extend(segments)

            # agents
            for agent_id in self.agents:
                vertices = list(
                    Point(0.0, 0.0)
                    .buffer(self.agent_radius, quad_segs=n_quad_segs)
                    .exterior.coords
                )[
                    :-1
                ]  # approximate circle with an octagon
                segments = [
                    [*vertex, *vertices[(i + 1) % len(vertices)]]
                    for i, vertex in enumerate(vertices)
                ]
                ray_int_seg_labels.extend(
                    len(segments) * [self.ray_int_label_map[f"agent_{agent_id}"]]
                )
                self.seg_label_type_to_inds["agent"].extend(
                    np.arange(
                        len(ray_int_segments), len(ray_int_segments) + len(segments)
                    )
                )
                ray_int_segments.extend(segments)

            # arrays
            self.ray_int_segments = np.array(ray_int_segments)
            self.ray_int_seg_labels = np.array(ray_int_seg_labels)

            # agent ray self intersection mask
            agent_int_seg_mask = np.ones(
                (self.num_agents, len(self.ray_int_seg_labels)), dtype=bool
            )
            agent_seg_inds = self.seg_label_type_to_inds["agent"]

            for i in range(self.num_agents):
                seg_inds_start = i * self.n_circle_segments
                agent_int_seg_mask[
                    i,
                    agent_seg_inds[
                        seg_inds_start : seg_inds_start + self.n_circle_segments
                    ],
                ] = False

            self.agent_int_seg_mask = np.expand_dims(agent_int_seg_mask, axis=1)

        # Occupancy map
        if self.gps_env:
            self._generate_valid_start_poses(land_mask)

    def create_background_image(self):
        """ "Creates pygame surface with static objects for faster rendering."""
        pygame.font.init()  # needed to import pygame fonts

        if self.gps_env:
            pygame_background_img = pygame.surfarray.make_surface(
                np.transpose(
                    self.background_img, (1, 0, 2)
                )  # pygame assumes images are (h, w, 3)
            )
            self.pygame_background_img = pygame.transform.scale(
                pygame_background_img, (self.screen_width, self.screen_height)
            )

            # add attribution text
            img_attribution_font = pygame.font.SysFont(
                None, round(0.35 * self.arena_buffer)
            )
            img_attribution_text = img_attribution_font.render(
                self.background_img_attribution, True, "black"
            )
            img_attribution_text_rect = img_attribution_text.get_rect()

            center_x = (
                self.screen_width
                - self.arena_buffer
                - 0.5 * img_attribution_text_rect[2]
            )  # object is [left,top,width,height]
            center_y = self.screen_height - 0.5 * self.arena_buffer
            img_attribution_text_rect.center = [center_x, center_y]

            self.pygame_background_img.blit(
                img_attribution_text, img_attribution_text_rect
            )
        else:
            self.pygame_background_img = pygame.Surface(
                (self.screen_width, self.screen_height)
            )
            self.pygame_background_img.fill((255, 255, 255))

        # Draw boundary
        draw.line(
            self.pygame_background_img,
            (0, 0, 0),
            self.env_to_screen(self.env_ul),
            self.env_to_screen(self.env_ur),
            width=self.boundary_width,
        )
        draw.line(
            self.pygame_background_img,
            (0, 0, 0),
            self.env_to_screen(self.env_ll),
            self.env_to_screen(self.env_lr),
            width=self.boundary_width,
        )
        draw.line(
            self.pygame_background_img,
            (0, 0, 0),
            self.env_to_screen(self.env_ul),
            self.env_to_screen(self.env_ll),
            width=self.boundary_width,
        )
        draw.line(
            self.pygame_background_img,
            (0, 0, 0),
            self.env_to_screen(self.env_ur),
            self.env_to_screen(self.env_lr),
            width=self.boundary_width,
        )

        # Scrimmage line
        draw.line(
            self.pygame_background_img,
            (128, 128, 128),
            self.env_to_screen(self.scrimmage_coords[0]),
            self.env_to_screen(self.scrimmage_coords[1]),
            width=self.boundary_width,
        )

        # Obstacles
        for obstacle_type, geoms in self.obstacle_geoms.items():
            if obstacle_type == "circle":
                for i in range(geoms.shape[0]):
                    draw.circle(
                        self.pygame_background_img,
                        (0, 0, 0),
                        self.env_to_screen(geoms[i, 1:]),
                        radius=geoms[i, 0] * self.pixel_size,
                        width=self.boundary_width,
                    )
            else:  # polygon obstacle
                for i in range(geoms.shape[0]):
                    draw.line(
                        self.pygame_background_img,
                        (0, 0, 0),
                        self.env_to_screen(geoms[i][0]),
                        self.env_to_screen(geoms[i][1]),
                        width=self.boundary_width,
                    )

        # Aquaticus field points
        if self.render_field_points and not self.gps_env:
            for v in self.config_dict["aquaticus_field_points"]:
                draw.circle(
                    self.pygame_background_img,
                    (128, 0, 128),
                    self.env_to_screen(self.config_dict["aquaticus_field_points"][v]),
                    radius=5,
                )

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

    def _set_dones(self):
        """Check all of the end game conditions."""
        # Check if all flags of one team are captured
        if self.state["captures"][1] == self.max_score:
            self.dones["red"] = True
            self.dones["__all__"] = True
            self.message = "Red Wins! Blue Loses"

        elif self.state["captures"][0] == self.max_score:
            self.dones["blue"] = True
            self.dones["__all__"] = True
            self.message = "Blue Wins! Red Loses"

        elif self.current_time >= self.max_time:
            self.dones["__all__"] = True
            if self.state["captures"][0] > self.state["captures"][1]:
                self.message = "Blue Wins! Red Loses"
            elif self.state["captures"][0] > self.state["captures"][1]:
                self.message = "Blue Wins! Red Loses"
            else:
                self.message = "Game Over. No Winner"

    def update_params(self, agent_id):
        # Important Note: Be sure to deep copy anything other than plain-old-data, e.g.,
        # lists from self.state
        # Otherwise it will point to the same object and prev_params/params will be identical
        agent = self.players[agent_id]
        obs = self.state_to_obs(agent.id, False)
        self.params[agent.id]["team"] = agent.team
        self.params[agent.id]["capture_radius"] = self.catch_radius
        self.params[agent.id]["agent_id"] = agent.id
        self.params[agent.id]["agent_oob"] = copy.deepcopy(self.state["agent_oob"])

        # Obstacle Distance/Bearing
        for i, obstacle in enumerate(self.state["dist_bearing_to_obstacles"][agent.id]):
            self.params[agent.id][f"obstacle_{i}_distance"] = obstacle[0]
            self.params[agent.id][f"obstacle_{i}_bearing"] = obstacle[1]

        if agent.team == Team.RED_TEAM:
            # Game Events
            self.params[agent.id]["num_teammates"] = self.num_red
            self.params[agent.id]["num_opponents"] = self.num_blue
            self.params[agent.id]["team_has_flag"] = self.state["team_has_flag"][1]
            self.params[agent.id]["team_flag_capture"] = self.red_team_flag_capture
            self.params[agent.id]["opponent_flag_pickup"] = self.state["team_has_flag"][
                0
            ]
            self.params[agent.id]["opponent_flag_capture"] = self.blue_team_flag_capture
            # Elements
            self.params[agent.id]["team_flag_home"] = (
                self.get_distance_between_2_points(
                    agent.pos, copy.deepcopy(self.state["flag_home"][1])
                )
            )
            self.params[agent.id]["team_flag_bearing"] = obs["own_home_bearing"]
            self.params[agent.id]["team_flag_distance"] = obs["own_home_distance"]
            self.params[agent.id]["opponent_flag_bearing"] = obs[
                "opponent_home_bearing"
            ]
            self.params[agent.id]["opponent_flag_distance"] = obs[
                "opponent_home_distance"
            ]
        else:
            # Game Events
            self.params[agent.id]["num_teammates"] = self.num_blue
            self.params[agent.id]["num_opponents"] = self.num_red
            self.params[agent.id]["team_has_flag"] = self.state["team_has_flag"][0]
            self.params[agent.id]["team_flag_capture"] = self.blue_team_flag_capture
            self.params[agent.id]["opponent_flag_pickup"] = self.state["team_has_flag"][
                1
            ]
            self.params[agent.id]["opponent_flag_capture"] = self.red_team_flag_capture
            # Elements
            self.params[agent.id]["team_flag_home"] = (
                self.get_distance_between_2_points(
                    agent.pos, copy.deepcopy(self.state["flag_home"][0])
                )
            )
            self.params[agent.id]["team_flag_bearing"] = obs["own_home_bearing"]
            self.params[agent.id]["team_flag_distance"] = obs["own_home_distance"]
            self.params[agent.id]["opponent_flag_bearing"] = obs[
                "opponent_home_bearing"
            ]
            self.params[agent.id]["opponent_flag_distance"] = obs[
                "opponent_home_distance"
            ]
        self.params[agent.id]["num_players"] = len(self.players)
        self.params[agent.id]["speed"] = agent.speed
        self.params[agent.id]["tagging_cooldown"] = not agent.tagging_cooldown >= 10.0
        self.params[agent.id]["thrust"] = agent.thrust
        self.params[agent.id]["has_flag"] = agent.has_flag
        self.params[agent.id]["on_own_side"] = agent.on_own_side
        self.params[agent.id]["heading"] = agent.heading
        # Distances to boundaries
        self.params[agent.id]["wall_0_bearing"] = obs["wall_0_bearing"]
        self.params[agent.id]["wall_0_distance"] = obs["wall_0_distance"]
        self.params[agent.id]["wall_1_bearing"] = obs["wall_1_bearing"]
        self.params[agent.id]["wall_1_distance"] = obs["wall_1_distance"]
        self.params[agent.id]["wall_2_bearing"] = obs["wall_2_bearing"]
        self.params[agent.id]["wall_2_distance"] = obs["wall_2_distance"]
        self.params[agent.id]["wall_3_bearing"] = obs["wall_3_bearing"]
        self.params[agent.id]["wall_3_distance"] = obs["wall_3_distance"]
        self.params[agent.id]["wall_distances"] = self._get_dists_to_boundary()[
            agent.id
        ]
        self.params[agent.id]["agent_made_tag"] = copy.deepcopy(
            self.state["agent_made_tag"]
        )
        self.params[agent.id]["agent_is_tagged"] = copy.deepcopy(
            self.state["agent_is_tagged"]
        )
        own_team = agent.team
        other_team = Team.BLUE_TEAM if own_team == Team.RED_TEAM else Team.RED_TEAM
        # Add Teamate and Opponent Information
        for team in [own_team, other_team]:
            dif_agents = filter(lambda a: a.id != agent.id, self.agents_of_team[team])
            for i, dif_agent in enumerate(dif_agents):
                entry_name = f"teammate_{i}" if team == own_team else f"opponent_{i}"
                status = "teammate" if team == own_team else "opponent"
                # bearing relative to the bearing to you
                self.params[agent.id][f"{status}_{dif_agent.id}_bearing"] = obs[
                    (entry_name, "bearing")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_distance"] = obs[
                    (entry_name, "distance")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_relative_heading"] = (
                    obs[(entry_name, "relative_heading")]
                )
                self.params[agent.id][f"{status}_{dif_agent.id}_speed"] = obs[
                    (entry_name, "speed")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_has_flag"] = obs[
                    (entry_name, "has_flag")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_on_side"] = obs[
                    (entry_name, "on_side")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_tagging_cooldown"] = (
                    obs[(entry_name, "tagging_cooldown")]
                )

    def compute_rewards(self, agent_id):
        if self.reward_config[agent_id] is None:
            return 0
        # Update Prev Params
        self.prev_params[agent_id] = copy.deepcopy(self.params[agent_id])
        # Update Params
        self.update_params(agent_id)
        if self.prev_params[agent_id] == {}:
            self.prev_params[agent_id] = copy.deepcopy(self.params[agent_id])
        # Get reward based on the passed in reward function
        return self.reward_config[agent_id](
            agent_id, self.params[agent_id], self.prev_params[agent_id]
        )

    def _reset_dones(self):
        """Resets the environments done indicators."""
        dones = {}
        dones["red"] = False
        dones["blue"] = False
        dones["__all__"] = False
        return dones

    def reset(self, seed=None, return_info=False, options: Optional[dict] = None):
        """
        Resets the environment so that it is ready to be used.

        Args:
            seed (optional): Starting seed.
            options (optional): Additonal options for resetting the environment:
                -"normalize": Whether or not to normalize observations and global state
                -"state_dict": self.state value from a previous episode
                -"init_dict": partial state_dict containing any information found in self.state aside from the following:
                    *prev_agent_position, agent_on_sides, agent_oob, agent_made_tag, dist_bearing_to_obstacles, flag_locations, flag_taken,
                     team_has_flag, obs_history_buffer, global_state_hist_buffer, lidar_labels, lidar_ends, lidar_distances
        """
        if seed is not None:
            self.seed(seed=seed)

        if return_info:
            raise DeprecationWarning(
                "return_info has been deprecated by PettingZoo -- https://github.com/Farama-Foundation/PettingZoo/pull/890"
            )

        if options is not None:
            self.normalize = options.get("normalize", config_dict_std["normalize"])
            state_dict = options.get("state_dict", None)
            init_dict = options.get("init_dict", None)
            if state_dict != None and init_dict != None:
                raise Exception(
                    "Cannot reset environment with both state_dict and init_dict (onl)."
                )
        else:
            state_dict = None
            init_dict = None

        if state_dict != None:
            # reset env from state_dict (a self.state value from a previous episode)
            # note: state_dict should be produced by the same or equivalent instance
            # of the Pyquaticus class, otherwise the obervations may be inconsistent
            if self.reset_count == 0:
                raise Exception(
                    "Resetting from state_dict should only be done for environment that has been previously reset"
                )
            self.state = copy.deepcopy(state_dict)
            self._set_player_attributes_from_state()
            self._set_flag_attributes_from_state()

        else:
            if init_dict != None:
                self._set_state_from_init_dict()
            else:
                flag_locations = np.asarray(list(self.flag_homes.values()))
                agent_positions, agent_spd_hdg, agent_on_sides = (
                    self._generate_agent_starts(flag_locations)
                )

                self.state = {
                    "agent_position": agent_positions,
                    "prev_agent_position": copy.deepcopy(agent_positions),
                    "agent_spd_hdg": agent_spd_hdg,
                    "agent_on_sides": agent_on_sides,
                    "agent_oob": np.zeros(
                        self.num_agents
                    ),  # if this agent is out of bounds
                    "agent_has_flag": np.zeros(self.num_agents),
                    "agent_made_tag": np.array(
                        [None] * self.num_agents
                    ),  # whether this agent tagged something at the current timestep (will be index of tagged agent if so)
                    "agent_is_tagged": np.zeros(
                        self.num_agents
                    ),  # if this agent is tagged
                    "agent_tagging_cooldown": np.array(
                        [self.tagging_cooldown] * self.num_agents
                    ),
                    "dist_bearing_to_obstacles": {
                        agent_id: np.zeros((len(self.obstacles), 2))
                        for agent_id in self.players
                    },
                    "flag_home": copy.deepcopy(flag_locations),
                    "flag_locations": copy.deepcopy(flag_locations),
                    "flag_taken": np.zeros(len(self.flags)),
                    "team_has_flag": np.zeros(
                        len(self.agents_of_team)
                    ),  # whether a member of this team has a flag of the other team's
                    "captures": np.zeros(
                        len(self.agents_of_team)
                    ),  # number of flag captures made by this team
                    "tags": np.zeros(
                        len(self.agents_of_team)
                    ),  # number of tags made by this team
                    "grabs": np.zeros(
                        len(self.agents_of_team)
                    ),  # number of flag grabs made by this team
                    "obs_hist_buffer": dict(),
                    "global_state_hist_buffer": list(),
                }

            # set player and flag attributes
            self._set_player_attributes_from_state()
            self._set_flag_attributes_from_state()

            # team wall orientation
            self._determine_team_wall_orient()

            # obstacles
            self._get_dist_bearing_to_obstacles()

            # lidar
            if self.lidar_obs:
                # reset lidar readings
                self.state["lidar_labels"] = dict()
                self.state["lidar_ends"] = dict()
                self.state["lidar_distances"] = dict()

                for agent_id in self.players:
                    self.state["lidar_labels"][agent_id] = np.zeros(self.num_lidar_rays)
                    self.state["lidar_ends"][agent_id] = np.zeros(
                        (self.num_lidar_rays, 2)
                    )
                    self.state["lidar_distances"][agent_id] = np.zeros(
                        self.num_lidar_rays
                    )
                    self._update_lidar()

                for team in self.agents_of_team:
                    for label_name, label_idx in self.ray_int_label_map.items():
                        if label_name.startswith("agent"):
                            # reset agent lidar detection states (not tagged and do not have flag)
                            if int(label_name[6:]) in self.agent_ids_of_team[team]:
                                self.obj_ray_detection_states[team][label_idx] = (
                                    LIDAR_DETECTION_CLASS_MAP["teammate"]
                                )
                            else:
                                self.obj_ray_detection_states[team][label_idx] = (
                                    LIDAR_DETECTION_CLASS_MAP["opponent"]
                                )

            # observation history
            reset_obs = {
                agent_id: self.state_to_obs(agent_id, self.normalize)
                for agent_id in self.players
            }
            for agent_id in self.players:
                self.state["obs_hist_buffer"][agent_id] = np.array(
                    self.hist_buffer_len * [reset_obs[agent_id]]
                )

            # global state history
            reset_global_state = self.state_to_global_obs(self.normalize)
            self.state["global_state_hist_buffer"] = np.array(
                self.hist_buffer_len * [reset_global_state]
            )

        self.message = ""
        self.current_time = 0
        self.reset_count += 1
        self.dones = self._reset_dones()

        # Rendering
        if self.render_mode:
            if self.render_saving:
                max_renders = (
                    1
                    + ceil(self.max_time / (self.sim_speedup_factor * self.tau))
                    * self.num_renders_per_step
                )
                self.render_buffer = np.zeros(
                    (max_renders, self.screen_height, self.screen_width, 3)
                )
            if self.render_traj_mode:
                self.traj_render_buffer = {
                    agent_id: {"traj": [], "agent": [], "history": []}
                    for agent_id in self.players
                }

            self.render_ctr = -1  # reset render doesn't count
            self._render()

        # Observations
        if self.hist_len > 1:
            obs = {
                agent_id: self.state["obs_hist_buffer"][agent_id][self.hist_buffer_inds]
                for agent_id in self.players
            }
        else:
            obs = {
                agent_id: self.state["obs_hist_buffer"][agent_id][0]
                for agent_id in self.players
            }

        # Return
        if return_info:
            # Info
            if self.hist_len > 1:
                info = {
                    "global_state": self.state["global_state_hist_buffer"][
                        self.hist_buffer_inds
                    ]
                }
            else:
                info = {"global_state": self.state["global_state_hist_buffer"][0]}

            return obs, info
        else:
            return obs

    def _set_state_from_init_dict(self, init_dict):
        """ "Initialize env state from a previous run's self.state value."""
        pass

    def _set_player_attributes_from_state(self):
        for i, player in enumerate(self.players.values()):
            player.pos = self.state["agent_position"][i]
            player.prev_pos = self.state["prev_agent_position"][i]
            player.speed = self.state["agent_spd_hdg"][i][0]
            player.heading = self.state["agent_spd_hdg"][i][1]
            player.has_flag = self.state["agent_has_flag"][i]
            player.on_own_side = self.state["agent_on_sides"][i]
            player.tagging_cooldown = self.state["agent_tagging_cooldown"][i]
            player.is_tagged = self.state["agent_is_tagged"][i]

    def _set_flag_attributes_from_state(self):
        for flag in self.flags:
            flag.home = self.state["flag_home"][int(flag.team)]
            flag.pos = self.state["flag_locations"][int(flag.team)]

    def _generate_agent_starts(self, flag_locations):
        """
        Generates starting positions for all players based on flag locations.

        If `default_init` is `False`, then agent positons are generated randomly using their home
        flag and random offsets.

        This allows players to be equidistant vertically from one another and all be in a straight line 1/4 of the way
        from the boundary.

        Args:
            flag_locations: The locations of all flags.

        Returns
        -------
            Initial player positions
            Initial player orientations
            Initial player velocities (always 0)
            Intial player on_sides (also always True)

        """
        r = self.agent_radius

        agent_locations = []
        agent_spd_hdg = []
        agent_on_sides = []

        if self.gps_env:
            valid_start_pos_inds = [i for i in range(len(self.valid_start_poses))]
            for player in self.players.values():
                # location
                valid_pos = False
                while not valid_pos:
                    start_pos_idx = np.random.choice(valid_start_pos_inds)
                    valid_start_pos_inds.remove(start_pos_idx)

                    player.pos = self.valid_start_poses[start_pos_idx]
                    valid_pos = self._check_valid_pos(
                        player.pos, agent_locations, flag_locations
                    )

                # heading
                player.heading = 360 * np.random.rand() - 180

                # other
                player.is_tagged = False
                player.thrust = 0.0
                player.speed = 0.0
                player.has_flag = False
                player.on_own_side = self._check_on_sides(player.pos, player.team)
                player.tagging_cooldown = self.tagging_cooldown
                player.home = copy.deepcopy(player.pos)
                player.prev_pos = copy.deepcopy(player.pos)

                # add to lists
                agent_locations.append(player.pos)
                agent_spd_hdg.append([player.speed, player.heading])
                agent_on_sides.append(player.on_own_side)
        else:
            for player in self.players.values():
                player.is_tagged = False
                player.thrust = 0.0
                player.speed = 0.0
                player.has_flag = False
                player.on_own_side = True
                player.tagging_cooldown = self.tagging_cooldown

                if not self.default_init:
                    while True:
                        # pick random point
                        # check if it's onsides, not hitting other agents and not hitting other flag
                        start_pos_x = (
                            np.random.rand() * self.env_size[0]
                        )  ## does this need to be world coordinates or env coordinates
                        start_pos_y = np.random.rand() * self.env_size[1]
                        start_point = [start_pos_x, start_pos_y]

                        # check if position is too close to obstacle
                        point_onsides = self._check_on_sides(start_point, player.team)
                        valid_pos = self._check_valid_pos(
                            start_point, agent_locations, flag_locations
                        )
                        obst_collision = detect_collision(
                            np.array(start_point),
                            self.agent_radius,
                            self.obstacle_geoms,
                        )
                        if point_onsides and valid_pos and (not obst_collision):
                            player.pos = start_point
                            break
                    player.heading = 360 * np.random.rand() - 180

                else:
                    player_flag_loc = self.flag_homes[player.team]
                    closest_scrim_line_point = closest_point_on_line(
                        self.scrimmage_coords[0],
                        self.scrimmage_coords[1],
                        player_flag_loc,
                    )
                    halfway_point = (
                        np.asarray(player_flag_loc) + closest_scrim_line_point
                    ) / 2

                    mag, angle = vec_to_mag_heading(halfway_point - player_flag_loc)
                    if mag < self.agent_radius + self.flag_keepout:
                        raise Exception("Flag is too close to scrimmage line.")

                    scrimmage_vec = self.scrimmage_coords[1] - self.scrimmage_coords[0]
                    spawn_line_env_intersection_1 = self._get_polygon_intersection(
                        halfway_point, scrimmage_vec, self.env_vertices
                    )[1]
                    spawn_line_env_intersection_2 = self._get_polygon_intersection(
                        halfway_point, -scrimmage_vec, self.env_vertices
                    )[1]
                    spawn_line_mag = np.linalg.norm(
                        spawn_line_env_intersection_1 - spawn_line_env_intersection_2
                    )
                    spawn_line_unit_vec = (
                        spawn_line_env_intersection_2 - spawn_line_env_intersection_1
                    ) / spawn_line_mag

                    if player.team == Team.BLUE_TEAM:
                        spawn_point = (
                            spawn_line_env_intersection_1
                            + (spawn_line_mag * (player.id + 1) / (self.team_size + 1))
                            * spawn_line_unit_vec
                        )
                    else:
                        spawn_point = (
                            spawn_line_env_intersection_1
                            + (
                                spawn_line_mag
                                * ((player.id + 1) - self.team_size)
                                / (self.team_size + 1)
                            )
                            * spawn_line_unit_vec
                        )

                    ## if spawn point isn't in the env bounds, then project back into the env
                    spawn_point[0] = max(
                        self.agent_radius,
                        min(self.env_size[0] - self.agent_radius, spawn_point[0]),
                    )
                    spawn_point[1] = max(
                        self.agent_radius,
                        min(self.env_size[1] - self.agent_radius, spawn_point[1]),
                    )

                    if detect_collision(
                        spawn_point, self.agent_radius, self.obstacle_geoms
                    ):  ## TODO: add check to make sure agent isn't spawned inside an obstacle
                        print(
                            "Initial agent position collides with obstacle. Picking random point instead."
                        )
                        ## pick random point
                        while True:
                            start_pos_x = np.random.rand() * self.env_size[0]
                            start_pos_y = np.random.rand() * self.env_size[1]
                            start_point = [start_pos_x, start_pos_y]

                            # check if position is too close to obstacle
                            point_onsides = self._check_on_sides(
                                start_point, player.team
                            )
                            valid_pos = self._check_valid_pos(
                                start_point, agent_locations, flag_locations
                            )
                            obst_collision = detect_collision(
                                np.array(start_point),
                                self.agent_radius,
                                self.obstacle_geoms,
                            )
                            if point_onsides and valid_pos and (not obst_collision):
                                spawn_point = start_point
                                break
                        angle = 360 * np.random.rand() - 180

                    player.pos = list(spawn_point)
                    player.heading = angle

                player.prev_pos = copy.deepcopy(player.pos)

                player.home = copy.deepcopy(player.pos)
                agent_locations.append(player.pos)
                agent_spd_hdg.append([player.speed, player.heading])
                agent_on_sides.append(True)

        return (
            np.asarray(agent_locations),
            np.asarray(agent_spd_hdg),
            np.asarray(agent_on_sides),
        )

    def _check_valid_pos(self, new_pos, agent_locations, flag_locations):
        agent_positions = np.array(agent_locations)
        flag_positions = np.array(flag_locations)

        if len(agent_positions) > 0:
            ag_distance = np.linalg.norm(agent_positions - new_pos, axis=1)
            if np.any(ag_distance <= self.catch_radius):
                return False

        flag_distance = np.linalg.norm(flag_positions - new_pos, axis=1)
        if np.any(flag_distance <= self.catch_radius):
            return False

        return True

    def _get_dists_to_boundary(self):
        """
        Returns a list of numbers of length self.num_agents
        where each number is the corresponding agents min distance to a boundary wall.
        """
        distances_to_walls = defaultdict(dict)
        for player in self.players.values():
            i = player.id
            x_pos = player.pos[0]
            y_pos = player.pos[1]

            distances_to_walls[i]["left_dist"] = x_pos
            distances_to_walls[i]["right_dist"] = self.env_size[0] - x_pos
            distances_to_walls[i]["bottom_dist"] = y_pos
            distances_to_walls[i]["top_dist"] = self.env_size[1] - y_pos

        return distances_to_walls

    def _get_dists_between_agents(self):
        """Returns dictionary of distances between agents indexed by agent numbers."""
        agt_to_agt_vecs = {}

        for player in self.players.values():
            i = player.id
            agt_to_agt_vecs[i] = {}
            i_pos = player.pos
            for other_player in self.players.values():
                j = other_player.id
                j_pos = other_player.pos
                agt_to_agt_vecs[i][j] = [j_pos[0] - i_pos[0], j_pos[1] - i_pos[1]]

        return agt_to_agt_vecs

    def _get_dist_to_flags(self):
        """Returns a dictionary mapping observation keys to 2d arrays."""
        flag_vecs = {}

        for player in self.players.values():
            flag_vecs[player.id] = {}
            team_idx = int(player.team)
            i_pos = player.pos
            proFlag_pos = self.flags[team_idx].pos
            retFlag_pos = self.flags[int(not team_idx)].pos
            flag_vecs[player.id]["own_home_dist"] = [
                proFlag_pos[0] - i_pos[0],
                proFlag_pos[1] - i_pos[1],
            ]
            flag_vecs[player.id]["opponent_home_dist"] = [
                retFlag_pos[0] - i_pos[0],
                retFlag_pos[1] - i_pos[1],
            ]

        return flag_vecs

    def _get_dist_bearing_to_obstacles(self):
        """Computes the distance and heading from each player to each obstacle"""
        dist_bearing_to_obstacles = dict()
        for player in self.players.values():
            player_pos = player.pos
            player_dists_to_obstacles = list()
            for obstacle in self.obstacles:
                # TODO: vectorize
                dist_to_obstacle = obstacle.distance_from(
                    player_pos, radius=self.agent_radius, heading=player.heading
                )
                player_dists_to_obstacles.append(dist_to_obstacle)
            dist_bearing_to_obstacles[player.id] = player_dists_to_obstacles
        self.state["dist_bearing_to_obstacles"] = dist_bearing_to_obstacles

    def _build_env_geom(
        self,
        env_bounds,
        flag_homes,
        scrimmage_coords,
        env_bounds_unit: str,
        flag_homes_unit: str,
        scrimmage_coords_unit: str,
        agent_radius: float,
        flag_radius: float,
        flag_keepout: float,
        catch_radius: float,
        max_speed: float,
        lidar_range: float,
    ):
        if self._is_auto_string(env_bounds) and (
            self._is_auto_string(flag_homes[Team.BLUE_TEAM])
            or self._is_auto_string(flag_homes[Team.RED_TEAM])
        ):
            raise Exception(
                "Either env_bounds or blue AND red flag homes must be set in config_dict (cannot both be 'auto')"
            )

        if self.gps_env:
            ### environment bounds ###
            if self._is_auto_string(env_bounds):
                flag_home_blue = np.asarray(flag_homes[Team.BLUE_TEAM])
                flag_home_red = np.asarray(flag_homes[Team.RED_TEAM])

                if flag_homes_unit == "m":
                    raise Exception(
                        "Flag homes must be specified in aboslute coordinates (lat/long or web mercator xy) to auto-generate gps environment bounds"
                    )
                elif flag_homes_unit == "ll":
                    # convert flag poses to web mercator xy
                    flag_home_blue = np.asarray(mt.xy(*flag_home_blue[-1::-1]))
                    flag_home_red = np.asarray(mt.xy(*flag_home_red[-1::-1]))

                flag_vec = flag_home_blue - flag_home_red
                flag_distance = np.linalg.norm(flag_vec)
                flag_unit_vec = flag_vec / flag_distance
                flag_perp_vec = np.array([-flag_unit_vec[1], flag_unit_vec[0]])

                # assuming default aquaticus field size ratio drawn on web mercator, these bounds will contain it
                bounds_pt1 = (
                    flag_home_blue
                    + (flag_distance / 6) * flag_unit_vec
                    + (flag_distance / 3) * flag_perp_vec
                )
                bounds_pt2 = (
                    flag_home_blue
                    + (flag_distance / 6) * flag_unit_vec
                    + (flag_distance / 3) * -flag_perp_vec
                )
                bounds_pt3 = (
                    flag_home_red
                    + (flag_distance / 6) * -flag_unit_vec
                    + (flag_distance / 3) * flag_perp_vec
                )
                bounds_pt4 = (
                    flag_home_red
                    + (flag_distance / 6) * -flag_unit_vec
                    + (flag_distance / 3) * -flag_perp_vec
                )
                bounds_points = np.array(
                    [bounds_pt1, bounds_pt2, bounds_pt3, bounds_pt4]
                )

                # environment bounds will be in web mercator xy
                env_bounds = np.zeros((2, 2))
                env_bounds[0][0] = np.min(bounds_points[:, 0])
                env_bounds[0][1] = np.min(bounds_points[:, 1])
                env_bounds[1][0] = np.max(bounds_points[:, 0])
                env_bounds[1][1] = np.max(bounds_points[:, 1])
            else:
                env_bounds = np.asarray(env_bounds)

                if env_bounds_unit == "m":
                    # check for exceptions
                    if (
                        self._is_auto_string(flag_homes[Team.BLUE_TEAM])
                        or self._is_auto_string(flag_homes[Team.RED_TEAM])
                        or flag_homes_unit == "m"
                    ):
                        raise Exception(
                            "Flag locations must be specified in aboslute coordinates (lat/long or web mercator xy) \
when gps environment bounds are specified in meters"
                        )

                    if len(env_bounds.shape) == 1:
                        env_bounds = np.array([(0.0, 0.0), env_bounds])
                    if np.any(env_bounds[1] == 0.0):
                        raise Exception(
                            "Environment max bounds must be > 0 when specified in meters"
                        )

                    # get flag midpoint
                    if flag_homes_unit == "wm_xy":
                        flag_home_blue = np.flip(_sm2ll(*flag_homes[Team.BLUE_TEAM]))
                        flag_home_red = np.flip(_sm2ll(*flag_homes[Team.RED_TEAM]))

                    geodict_flags = Geodesic.WGS84.Inverse(
                        lat1=flag_home_blue[0],
                        lon1=flag_home_blue[1],
                        lat2=flag_home_red[0],
                        lon2=flag_home_red[1],
                    )
                    geodict_flag_midpoint = Geodesic.WGS84.Direct(
                        lat1=flag_home_blue[0],
                        lon1=flag_home_blue[1],
                        azi1=geodict_flags["azi1"],
                        s12=geodict_flags["s12"] / 2,
                    )
                    flag_midpoint = (
                        geodict_flag_midpoint["lat2"],
                        geodict_flag_midpoint["lon2"],
                    )

                    # vertical bounds
                    env_top = Geodesic.WGS84.Direct(
                        lat1=flag_midpoint[0],
                        lon1=flag_midpoint[1],
                        azi1=0,  # degrees
                        s12=0.5 * env_bounds[1][1],
                    )["lat2"]
                    env_bottom = Geodesic.WGS84.Direct(
                        lat1=flag_midpoint[0],
                        lon1=flag_midpoint[1],
                        azi1=180,  # degrees
                        s12=0.5 * env_bounds[1][1],
                    )["lat2"]

                    # horizontal bounds
                    geoc_lat = np.arctan(
                        (POLAR_RADIUS / EQUATORIAL_RADIUS) ** 2
                        * np.tan(np.deg2rad(flag_midpoint[0]))
                    )
                    small_circle_circum = (
                        np.pi * 2 * EQUATORIAL_RADIUS * np.cos(geoc_lat)
                    )
                    env_left = flag_midpoint[1] - 360 * (
                        0.5 * env_bounds[1][0] / small_circle_circum
                    )
                    env_right = flag_midpoint[1] + 360 * (
                        0.5 * env_bounds[1][0] / small_circle_circum
                    )

                    env_left = angle180(env_left)
                    env_right = angle180(env_right)

                    # convert bounds to web mercator xy
                    env_bounds = np.array(
                        [mt.xy(env_left, env_bottom), mt.xy(env_right, env_top)]
                    )
                elif env_bounds_unit == "ll":
                    # convert bounds to web mercator xy
                    wm_xy_bounds = np.array(
                        [mt.xy(*env_bounds[0][-1::-1]), mt.xy(*env_bounds[1][-1::-1])]
                    )
                    left = np.min(wm_xy_bounds[:, 0])
                    bottom = np.min(wm_xy_bounds[:, 1])
                    right = np.max(wm_xy_bounds[:, 0])
                    top = np.max(wm_xy_bounds[:, 1])
                    env_bounds = np.array([[left, bottom], [right, top]])
                else:  # web mercator xy
                    left = np.min(env_bounds[:, 0])
                    bottom = np.min(env_bounds[:, 1])
                    right = np.max(env_bounds[:, 0])
                    top = np.max(env_bounds[:, 1])
                    env_bounds = np.array([[left, bottom], [right, top]])
            # unit
            env_bounds_unit = "wm_xy"

            # environment size
            self.env_size = np.diff(env_bounds, axis=0)[0]
            self.env_diag = np.linalg.norm(self.env_size)

            # vertices
            env_bounds_vertices = np.array(
                [
                    env_bounds[0],
                    (env_bounds[1][0], env_bounds[0][1]),
                    env_bounds[1],
                    (env_bounds[0][0], env_bounds[1][1]),
                ]
            )

            ### flags home ###
            # auto home
            if self._is_auto_string(
                flag_homes[Team.BLUE_TEAM]
            ) and self._is_auto_string(flag_homes[Team.RED_TEAM]):
                flag_homes[Team.BLUE_TEAM] = env_bounds[0] + np.array(
                    [7 / 8 * self.env_size[0], 0.5 * self.env_size[0]]
                )
                flag_homes[Team.RED_TEAM] = env_bounds[0] + np.array(
                    [1 / 8 * self.env_size[0], 0.5 * self.env_size[0]]
                )
            elif self._is_auto_string(
                flag_homes[Team.BLUE_TEAM]
            ) or self._is_auto_string(flag_homes[Team.RED_TEAM]):
                raise Exception(
                    "Flag homes should be either all 'auto', or all specified"
                )
            else:
                if flag_homes_unit == "m":
                    raise Exception(
                        "'m' (meters) should only be used to specify flag homes when gps_env is False"
                    )

                flag_homes[Team.BLUE_TEAM] = np.asarray(flag_homes[Team.BLUE_TEAM])
                flag_homes[Team.RED_TEAM] = np.asarray(flag_homes[Team.RED_TEAM])

                if flag_homes_unit == "ll":
                    # convert flag poses to web mercator xy
                    flag_homes[Team.BLUE_TEAM] = mt.xy(
                        *flag_homes[Team.BLUE_TEAM][-1::-1]
                    )
                    flag_homes[Team.RED_TEAM] = mt.xy(
                        *flag_homes[Team.RED_TEAM][-1::-1]
                    )

            # blue flag
            if np.any(flag_homes[Team.BLUE_TEAM] <= env_bounds[0]) or np.any(
                flag_homes[Team.BLUE_TEAM] >= env_bounds[1]
            ):
                raise Exception(
                    f"Blue flag home {flag_homes[Team.BLUE_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}"
                )

            # red flag
            if np.any(flag_homes[Team.RED_TEAM] <= env_bounds[0]) or np.any(
                flag_homes[Team.RED_TEAM] >= env_bounds[1]
            ):
                raise Exception(
                    f"Red flag home {flag_homes[Team.RED_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}"
                )

            # normalize relative to environment bounds
            flag_homes[Team.BLUE_TEAM] -= env_bounds[0]
            flag_homes[Team.RED_TEAM] -= env_bounds[0]

            # unit
            flag_homes_unit = "wm_xy"

            ### scrimmage line ###
            if self._is_auto_string(scrimmage_coords):
                flags_vec = flag_homes[Team.BLUE_TEAM] - flag_homes[Team.RED_TEAM]

                scrim_vec1 = np.array([-flags_vec[1], flags_vec[0]])
                scrim_vec2 = np.array([flags_vec[1], -flags_vec[0]])
                flags_midpoint = (
                    0.5 * (flag_homes[Team.BLUE_TEAM] + flag_homes[Team.RED_TEAM])
                    + env_bounds[0]
                )

                scrimmage_coord1 = self._get_polygon_intersection(
                    flags_midpoint, scrim_vec1, env_bounds_vertices
                )[1]
                scrimmage_coord2 = self._get_polygon_intersection(
                    flags_midpoint, scrim_vec2, env_bounds_vertices
                )[1]
                scrimmage_coords = (
                    np.asarray([scrimmage_coord1, scrimmage_coord2]) - env_bounds[0]
                )
            else:
                if scrimmage_coords_unit == "m":
                    raise Exception(
                        "'m' (meters) should only be used to specify flag homes when gps_env is False"
                    )
                elif scrimmage_coords_unit == "wm_xy":
                    pass
                elif scrimmage_coords_unit == "ll":
                    scrimmage_coords_1 = mt.xy(*scrimmage_coords[0])
                    scrimmage_coords_2 = mt.xy(*scrimmage_coords[1])
                    scrimmage_coords = np.array(
                        [scrimmage_coords_1, scrimmage_coords_2]
                    )
                else:
                    raise Exception(
                        f"Unit '{scrimmage_coords_unit}' not recognized. Please choose from 'll' or 'wm_xy' for gps environments."
                    )

                if np.all(scrimmage_coords[0] == scrimmage_coords[1]):
                    raise Exception(
                        "Scrimmage line must be specified with two DIFFERENT coordinates"
                    )

                if scrimmage_coords[0][0] == scrimmage_coords[1][0] and (
                    scrimmage_coords[0][0] == env_bounds[0][0]
                    or scrimmage_coords[0][0] == env_bounds[1][0]
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} cannot lie on the same edge of the env boundary"
                    )
                if scrimmage_coords[0][1] == scrimmage_coords[1][1] and (
                    scrimmage_coords[0][1] == env_bounds[0][1]
                    or scrimmage_coords[0][1] == env_bounds[1][1]
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} cannot lie on the same edge of the env boundary"
                    )

                if scrimmage_coords[1][1] == scrimmage_coords[0][1]:  # horizontal line
                    extended_point_1 = [
                        env_bounds[0][0] + (env_bounds[0][0] - scrimmage_coords[0][0]),
                        scrimmage_coords[0][1],
                    ]
                    extended_point_2 = [
                        env_bounds[1][0] + (env_bounds[1][0] - scrimmage_coords[1][0]),
                        scrimmage_coords[1][1],
                    ]
                elif scrimmage_coords[1][0] == scrimmage_coords[0][0]:  # vertical line
                    extended_point_1 = [
                        scrimmage_coords[0][0],
                        env_bounds[0][1] + (env_bounds[0][1] - scrimmage_coords[0][1]),
                    ]
                    extended_point_2 = [
                        scrimmage_coords[1][0],
                        env_bounds[1][1] + (env_bounds[1][1] - scrimmage_coords[1][1]),
                    ]
                else:
                    scrimmage_slope = (
                        scrimmage_coords[1][1] - scrimmage_coords[0][1]
                    ) / (scrimmage_coords[1][0] - scrimmage_coords[0][0])

                    # compute intersection point for the first scrimmage_coord
                    t_env_bound_x1 = (
                        env_bounds[0][0] - scrimmage_coords[0][0]
                    ) * scrimmage_slope
                    t_env_bound_y1 = (
                        (env_bounds[0][1] - scrimmage_coords[0][1]) / scrimmage_slope
                        if scrimmage_slope != 0
                        else 0
                    )
                    t_env_bound_x2 = (
                        env_bounds[1][0] - scrimmage_coords[0][0]
                    ) * scrimmage_slope
                    t_env_bound_y2 = (
                        (env_bounds[1][1] - scrimmage_coords[0][1]) / scrimmage_slope
                        if scrimmage_slope != 0
                        else 0
                    )
                    t_env_bounds = [
                        t_env_bound_x1,
                        t_env_bound_y1,
                        t_env_bound_x2,
                        t_env_bound_y2,
                    ]
                    max_t = max(t_env_bounds) * 10
                    min_t = min(t_env_bounds) * 10

                    extended_point_1 = (
                        scrimmage_coords[0] + np.array([max_t, max_t * scrimmage_slope])
                        if max_t > 0
                        else scrimmage_coords[0]
                    )
                    extended_point_2 = (
                        scrimmage_coords[0] + np.array([min_t, min_t * scrimmage_slope])
                        if min_t < 0
                        else scrimmage_coords[0]
                    )

                extended_scrimmage_coords = np.array(
                    [extended_point_1, extended_point_2]
                )
                full_scrim_line = LineString(extended_scrimmage_coords)
                scrim_line_env_intersection = intersection(
                    full_scrim_line, Polygon(env_bounds_vertices)
                )

                if (
                    scrim_line_env_intersection.is_empty
                    or len(scrim_line_env_intersection.coords)
                    == 1  # only intersects a vertex
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                    )
                else:
                    scrim_line_env_intersection = np.array(
                        scrim_line_env_intersection.coords
                    )

                    # intersection points should lie on boundary (if they don't then the line doesn't bisect the env)
                    if not (
                        (
                            (
                                env_bounds[0][0]
                                <= scrim_line_env_intersection[0][0]
                                <= env_bounds[1][0]
                            )
                            and (
                                (scrim_line_env_intersection[0][1] == env_bounds[0][1])
                                or (
                                    scrim_line_env_intersection[0][1]
                                    == env_bounds[1][1]
                                )
                            )
                        )
                        or (
                            (
                                env_bounds[0][1]
                                <= scrim_line_env_intersection[0][1]
                                <= env_bounds[1][1]
                            )
                            and (
                                (scrim_line_env_intersection[0][0] == env_bounds[0][0])
                                or (
                                    scrim_line_env_intersection[0][0]
                                    == env_bounds[1][0]
                                )
                            )
                        )
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                        )
                    if not (
                        (
                            (
                                env_bounds[0][0]
                                <= scrim_line_env_intersection[1][0]
                                <= env_bounds[1][0]
                            )
                            and (
                                (scrim_line_env_intersection[1][1] == env_bounds[0][1])
                                or (
                                    scrim_line_env_intersection[1][1]
                                    == env_bounds[1][1]
                                )
                            )
                        )
                        or (
                            (
                                env_bounds[0][1]
                                <= scrim_line_env_intersection[1][1]
                                <= env_bounds[1][1]
                            )
                            and (
                                (scrim_line_env_intersection[1][0] == env_bounds[0][0])
                                or (
                                    scrim_line_env_intersection[1][0]
                                    == env_bounds[1][0]
                                )
                            )
                        )
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                        )
                    scrimmage_coords = scrim_line_env_intersection
                scrimmage_coords = scrimmage_coords - env_bounds[0]

            # unit
            scrimmage_coords_unit = "wm_xy"

            ### agent and flag geometries ###
            lon1, lat1 = _sm2ll(*env_bounds[0])
            lon2, lat2 = _sm2ll(*env_bounds[1])
            lon_diff = self._longitude_diff_west2east(lon1, lon2)

            if np.abs(lat1) > np.abs(lat2):
                lat = lat1
            else:
                lat = lat2

            geoc_lat = np.arctan(
                (POLAR_RADIUS / EQUATORIAL_RADIUS) ** 2 * np.tan(np.deg2rad(lat))
            )
            small_circle_circum = np.pi * 2 * EQUATORIAL_RADIUS * np.cos(geoc_lat)

            # use most warped (squished) horizontal environment border to underestimate the number of
            # meters per mercator xy, therefore overestimate how close objects are to one another
            self.meters_per_mercator_xy = (
                small_circle_circum * (lon_diff / 360) / self.env_size[0]
            )
            agent_radius /= self.meters_per_mercator_xy
            flag_radius /= self.meters_per_mercator_xy
            catch_radius /= self.meters_per_mercator_xy
            flag_keepout /= self.meters_per_mercator_xy
            lidar_range /= self.meters_per_mercator_xy

        else:
            ### environment bounds ###
            if env_bounds_unit != "m":
                raise Exception(
                    "Environment bounds unit must be meters ('m') when gps_env is False"
                )

            if self._is_auto_string(env_bounds):
                if np.any(
                    np.sign([flag_homes[Team.BLUE_TEAM], flag_homes[Team.RED_TEAM]])
                    == -1
                ):
                    raise Exception(
                        "Flag coordinates must be in the positive quadrant when gps_env is False"
                    )

                if np.any(
                    np.sign([flag_homes[Team.BLUE_TEAM], flag_homes[Team.RED_TEAM]])
                    == 0
                ):
                    raise Exception(
                        "Flag coordinates must not lie on the axes of the positive quadrant when gps_env is False"
                    )

                # environment size
                flag_xmin = min(
                    flag_homes[Team.BLUE_TEAM][0], flag_homes[Team.RED_TEAM][0]
                )
                flag_ymin = min(
                    flag_homes[Team.BLUE_TEAM][1], flag_homes[Team.RED_TEAM][1]
                )

                flag_xmax = max(
                    flag_homes[Team.BLUE_TEAM][0], flag_homes[Team.RED_TEAM][0]
                )
                flag_ymax = max(
                    flag_homes[Team.BLUE_TEAM][1], flag_homes[Team.RED_TEAM][1]
                )

                self.env_size = np.array([flag_xmax + flag_xmin, flag_ymax + flag_ymin])
                env_bounds = np.array([(0.0, 0.0), self.env_size])
            else:
                env_bounds = np.asarray(env_bounds)

                if len(env_bounds.shape) == 1:
                    if np.any(env_bounds == 0.0):
                        raise Exception(
                            "Environment max bounds must be > 0 when specified in meters"
                        )

                    # environment size
                    self.env_size = env_bounds
                    env_bounds = np.array([(0.0, 0.0), env_bounds])
                else:
                    if not np.all(env_bounds[0] == 0.0):
                        raise Exception(
                            "Environment min bounds must be 0 when specified in meters"
                        )

                    if np.any(env_bounds[1] == 0.0):
                        raise Exception(
                            "Environment max bounds must be > 0 when specified in meters"
                        )

            # environment diagonal and vertices
            self.env_diag = np.linalg.norm(self.env_size)
            env_bounds_vertices = np.array(
                [
                    env_bounds[0],
                    (env_bounds[1][0], env_bounds[0][1]),
                    env_bounds[1],
                    (env_bounds[0][0], env_bounds[1][1]),
                ]
            )

            ### flags home ###
            # auto home
            if self._is_auto_string(
                flag_homes[Team.BLUE_TEAM]
            ) and self._is_auto_string(flag_homes[Team.RED_TEAM]):
                if flag_homes_unit == "ll" or flag_homes_unit == "wm_xy":
                    raise Exception(
                        "'ll' (Lat/Long) and 'wm_xy' (web mercator xy) units should only be used when gps_env is True"
                    )
                flag_homes[Team.BLUE_TEAM] = np.array(
                    [7 / 8 * self.env_size[0], 0.5 * self.env_size[1]]
                )
                flag_homes[Team.RED_TEAM] = np.array(
                    [1 / 8 * self.env_size[0], 0.5 * self.env_size[1]]
                )
            elif self._is_auto_string(
                flag_homes[Team.BLUE_TEAM]
            ) or self._is_auto_string(flag_homes[Team.RED_TEAM]):
                raise Exception("Flag homes are either all 'auto', or all specified")
            else:
                flag_homes[Team.BLUE_TEAM] = np.asarray(flag_homes[Team.BLUE_TEAM])
                flag_homes[Team.RED_TEAM] = np.asarray(flag_homes[Team.RED_TEAM])

            # blue flag
            if np.any(flag_homes[Team.BLUE_TEAM] <= env_bounds[0]) or np.any(
                flag_homes[Team.BLUE_TEAM] >= env_bounds[1]
            ):
                raise Exception(
                    f"Blue flag home {flag_homes[Team.BLUE_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}"
                )

            # red flag
            if np.any(flag_homes[Team.RED_TEAM] <= env_bounds[0]) or np.any(
                flag_homes[Team.RED_TEAM] >= env_bounds[1]
            ):
                raise Exception(
                    f"Red flag home {flag_homes[Team.RED_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}"
                )

            ### scrimmage line ###
            if self._is_auto_string(scrimmage_coords):
                flags_vec = flag_homes[Team.BLUE_TEAM] - flag_homes[Team.RED_TEAM]

                scrim_vec1 = np.array([-flags_vec[1], flags_vec[0]])
                scrim_vec2 = np.array([flags_vec[1], -flags_vec[0]])
                flags_midpoint = 0.5 * (
                    flag_homes[Team.BLUE_TEAM] + flag_homes[Team.RED_TEAM]
                )

                scrimmage_coord1 = self._get_polygon_intersection(
                    flags_midpoint, scrim_vec1, env_bounds_vertices
                )[1]
                scrimmage_coord2 = self._get_polygon_intersection(
                    flags_midpoint, scrim_vec2, env_bounds_vertices
                )[1]
                scrimmage_coords = np.asarray([scrimmage_coord1, scrimmage_coord2])
            else:
                if scrimmage_coords_unit == "ll" or scrimmage_coords_unit == "wm_xy":
                    raise Exception(
                        "'ll' (Lat/Long) and 'wm_xy' (web mercator xy) units should only be used when gps_env is True"
                    )

                scrimmage_coords = np.asarray(scrimmage_coords)

                if np.all(scrimmage_coords[0] == scrimmage_coords[1]):
                    raise Exception(
                        "Scrimmage line must be specified with two DIFFERENT coordinates"
                    )

                if scrimmage_coords[0][0] == scrimmage_coords[1][0] and (
                    scrimmage_coords[0][0] == env_bounds[0][0]
                    or scrimmage_coords[0][0] == env_bounds[1][0]
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} cannot lie on the same edge of the env boundary"
                    )
                if scrimmage_coords[0][1] == scrimmage_coords[1][1] and (
                    scrimmage_coords[0][1] == env_bounds[0][1]
                    or scrimmage_coords[0][1] == env_bounds[1][1]
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} cannot lie on the same edge of the env boundary"
                    )

                # env biseciton check
                full_scrim_line = LineString(scrimmage_coords)
                scrim_line_env_intersection = intersection(
                    full_scrim_line, Polygon(env_bounds_vertices)
                )

                if (
                    scrim_line_env_intersection.is_empty
                    or len(scrim_line_env_intersection.coords)
                    == 1  # only intersects a vertex
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                    )
                else:
                    scrim_line_env_intersection = np.array(
                        scrim_line_env_intersection.coords
                    )

                    # intersection points should lie on boundary (if they don't then the line doesn't bisect the env)
                    if not (
                        (
                            (
                                env_bounds[0][0]
                                <= scrim_line_env_intersection[0][0]
                                <= env_bounds[1][0]
                            )
                            and (
                                (scrim_line_env_intersection[0][1] == env_bounds[0][1])
                                or (
                                    scrim_line_env_intersection[0][1]
                                    == env_bounds[1][1]
                                )
                            )
                        )
                        or (
                            (
                                env_bounds[0][1]
                                <= scrim_line_env_intersection[0][1]
                                <= env_bounds[1][1]
                            )
                            and (
                                (scrim_line_env_intersection[0][0] == env_bounds[0][0])
                                or (
                                    scrim_line_env_intersection[0][0]
                                    == env_bounds[1][0]
                                )
                            )
                        )
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                        )
                    if not (
                        (
                            (
                                env_bounds[0][0]
                                <= scrim_line_env_intersection[1][0]
                                <= env_bounds[1][0]
                            )
                            and (
                                (scrim_line_env_intersection[1][1] == env_bounds[0][1])
                                or (
                                    scrim_line_env_intersection[1][1]
                                    == env_bounds[1][1]
                                )
                            )
                        )
                        or (
                            (
                                env_bounds[0][1]
                                <= scrim_line_env_intersection[1][1]
                                <= env_bounds[1][1]
                            )
                            and (
                                (scrim_line_env_intersection[1][0] == env_bounds[0][0])
                                or (
                                    scrim_line_env_intersection[1][0]
                                    == env_bounds[1][0]
                                )
                            )
                        )
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                        )

        ### Set Attributes ###
        # environment geometries
        self.env_bounds = env_bounds
        self.env_bounds_unit = env_bounds_unit

        self.flag_homes = flag_homes
        self.flag_homes_unit = flag_homes_unit

        self.scrimmage_coords = scrimmage_coords
        self.scrimmage_coords_unit = scrimmage_coords_unit
        self.scrimmage_vec = scrimmage_coords[1] - scrimmage_coords[0]

        # on sides
        scrim2blue = self.flag_homes[Team.BLUE_TEAM] - scrimmage_coords[0]
        scrim2red = self.flag_homes[Team.RED_TEAM] - scrimmage_coords[0]

        self.on_sides_sign = {}
        self.on_sides_sign[Team.BLUE_TEAM] = np.sign(
            np.cross(self.scrimmage_vec, scrim2blue)
        )
        self.on_sides_sign[Team.RED_TEAM] = np.sign(
            np.cross(self.scrimmage_vec, scrim2red)
        )

        # flag bisection check
        if self.on_sides_sign[Team.BLUE_TEAM] == self.on_sides_sign[Team.RED_TEAM]:
            raise Exception(
                "The specified flag locations and scrimmage line coordinates are not valid because the scrimmage line does not divide the flag locations"
            )

        closest_point_blue_flag_to_scrimmage_line = closest_point_on_line(
            self.scrimmage_coords[0],
            self.scrimmage_coords[1],
            self.flag_homes[Team.BLUE_TEAM],
        )
        closest_point_red_flag_to_scrimmage_line = closest_point_on_line(
            self.scrimmage_coords[0],
            self.scrimmage_coords[1],
            self.flag_homes[Team.RED_TEAM],
        )

        dist_blue_flag_to_scrimmage_line = np.linalg.norm(
            closest_point_blue_flag_to_scrimmage_line - self.flag_homes[Team.BLUE_TEAM]
        )
        dist_red_flag_to_scrimmage_line = np.linalg.norm(
            closest_point_red_flag_to_scrimmage_line - self.flag_homes[Team.RED_TEAM]
        )

        if dist_blue_flag_to_scrimmage_line < LINE_INTERSECT_TOL:
            raise Exception("The blue flag is too close to the scrimmage line.")
        elif dist_red_flag_to_scrimmage_line < LINE_INTERSECT_TOL:
            raise Exception("The red flag is too close to the scrimmage line.")

        # agent and flag geometries
        if self.lidar_obs:
            self.lidar_range = lidar_range

        self.agent_radius = agent_radius
        self.flag_radius = flag_radius
        self.catch_radius = catch_radius
        self.flag_keepout = flag_keepout
        self.max_speed = max_speed

    def _get_line_intersection(
        self, origin: np.ndarray, vec: np.ndarray, line: np.ndarray
    ):
        """
        origin: a point within the environment (not on environment bounds)
        """
        vec_end = origin + self.env_diag * vec / np.linalg.norm(vec)
        vec_line = LineString((origin, vec_end))
        inter = intersection(vec_line, LineString(line))

        if inter.is_empty:
            return None
        else:
            return np.asarray(inter.coords[0])

    def _get_polygon_intersection(
        self, origin: np.ndarray, vec: np.ndarray, polygon: np.ndarray
    ):
        """
        origin: a point within the environment (not on environment bounds)
        """
        vec_end = origin + self.env_diag * vec / np.linalg.norm(vec)
        vec_line = LineString((origin, vec_end))
        inter = intersection(vec_line, Polygon(polygon))

        if inter.is_empty:
            return None
        else:
            if hasattr(inter, "coords"):
                return np.asarray(inter.coords)
            else:
                return np.asarray([np.asarray(ls.coords) for ls in inter.geoms])

    def _is_auto_string(self, var):
        return isinstance(var, str) and var == "auto"

    def _longitude_diff_west2east(self, lon1, lon2):
        """Calculate the longitude difference from westing (lon1) to easting (lon2)"""
        diff = lon2 - lon1

        # adjust for crossing the 180/-180 boundary
        if diff < 0:
            diff += 360

        return diff

    def _get_topo_geom(self):
        ### Environment Map Retrieval and Caching ###
        map_caching_dir = str(
            pathlib.Path(__file__).resolve().parents[1] / "__mapcache__"
        )
        if not os.path.isdir(map_caching_dir):
            os.mkdir(map_caching_dir)

        lon1, lat1 = np.round(_sm2ll(*self.env_bounds[0]), 7)
        lon2, lat2 = np.round(_sm2ll(*self.env_bounds[1]), 7)

        map_cache_path = os.path.join(
            map_caching_dir, f"tile@(({lat1},{lon1}), ({lat2},{lon2})).pkl"
        )

        if os.path.exists(map_cache_path):
            # load cached environment map(s)
            with open(map_cache_path, "rb") as f:
                map_cache = pickle.load(f)

            topo_img = map_cache["topographical_image"]
            self.background_img = map_cache["render_image"]
            self.background_img_attribution = map_cache["attribution"]
        else:
            # retrieve maps from tile provider
            topo_tile_source = cx.providers.CartoDB.DarkMatterNoLabels  # DO NOT CHANGE!
            render_tile_source = cx.providers.CartoDB.Voyager  # DO NOT CHANGE!
            self.background_img_attribution = render_tile_source.get("attribution")

            render_tile_bounds = self.env_bounds + self.arena_buffer_frac * np.asarray(
                [[-self.env_diag], [self.env_diag]]
            )

            topo_tile, topo_ext = cx.bounds2img(
                *self.env_bounds.flatten(),
                zoom="auto",
                source=topo_tile_source,
                ll=False,
                wait=0,
                max_retries=2,
                n_connections=1,
                use_cache=False,
                zoom_adjust=None,
            )
            render_tile, render_ext = cx.bounds2img(
                *render_tile_bounds.flatten(),
                zoom="auto",
                source=render_tile_source,
                ll=False,
                wait=0,
                max_retries=2,
                n_connections=1,
                use_cache=False,
                zoom_adjust=None,
            )

            topo_img = self._crop_tiles(
                topo_tile[:, :, :-1], topo_ext, *self.env_bounds.flatten(), ll=False
            )
            self.background_img = self._crop_tiles(
                render_tile[:, :, :-1],
                render_ext,
                *render_tile_bounds.flatten(),
                ll=False,
            )

            # cache maps
            map_cache = {
                "topographical_image": topo_img,
                "render_image": self.background_img,
                "attribution": self.background_img_attribution,
            }
            with open(map_cache_path, "wb") as f:
                pickle.dump(map_cache, f)

        ### Topology Construction ###
        # mask by water color on topo image
        water_x, water_y = self.flag_homes[Team.BLUE_TEAM]  # assume flag is in water
        water_pixel_x = ceil(topo_img.shape[1] * (water_x / self.env_size[0])) - 1
        water_pixel_y = ceil(topo_img.shape[0] * (1 - water_y / self.env_size[1])) - 1

        water_pixel_color = topo_img[water_pixel_y, water_pixel_x]
        mask = np.all(topo_img == water_pixel_color, axis=-1)
        water_connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled_mask, _ = label(mask, structure=water_connectivity)
        target_label = labeled_mask[water_pixel_y, water_pixel_x]

        grayscale_topo_img = cv2.cvtColor(topo_img, cv2.COLOR_RGB2GRAY)
        water_pixel_color_gray = grayscale_topo_img[water_pixel_y, water_pixel_x]

        land_mask = (labeled_mask == target_label) + (
            water_pixel_color_gray <= grayscale_topo_img
        ) * (grayscale_topo_img <= water_pixel_color_gray + 2)

        # water contours
        land_mask_binary = 255 * land_mask.astype(np.uint8)
        water_contours, _ = cv2.findContours(
            land_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # https://docs.opencv.org/4.10.0/d4/d73/tutorial_py_contours_begin.html
        # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html

        border_contour = max(water_contours, key=cv2.contourArea)
        # TODO: check if this is just the environment bounds, then non-convex approximation will go to the largest island
        border_land_mask = cv2.drawContours(
            np.zeros_like(land_mask_binary), [border_contour], -1, 255, -1
        )

        # island contours
        water_mask = np.logical_not(land_mask)
        island_binary = 255 * (border_land_mask * water_mask).astype(np.uint8)
        island_contours, _ = cv2.findContours(
            island_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # approximate outer contour (border land)
        eps = self.topo_contour_eps * cv2.arcLength(border_contour, True)
        border_cnt_approx = cv2.approxPolyDP(border_contour, eps, True)

        border_land_mask_approx = cv2.drawContours(
            np.zeros_like(land_mask_binary), [border_cnt_approx], -1, 255, -1
        )
        border_land_mask_approx = cv2.drawContours(
            border_land_mask_approx, [border_cnt_approx], -1, 0, 0
        )

        labeled_border_land_mask_approx, _ = label(
            border_land_mask_approx, structure=water_connectivity
        )
        target_water_label = labeled_border_land_mask_approx[
            water_pixel_y, water_pixel_x
        ]
        border_land_mask_approx = labeled_border_land_mask_approx == target_water_label

        # approximate island contours
        island_cnts_approx = []
        for i, cnt in enumerate(island_contours):
            if cnt.shape[0] != 1:
                eps = self.topo_contour_eps * cv2.arcLength(cnt, True)
                cnt_approx = cv2.approxPolyDP(cnt, eps, True)
                cvx_hull = cv2.convexHull(cnt_approx)
                island_cnts_approx.append(cvx_hull)

        island_mask_approx = cv2.drawContours(
            255 * np.ones_like(island_binary), island_cnts_approx, -1, 0, -1
        )  # convex island masks

        # final approximate land mask
        land_mask_approx = border_land_mask_approx * island_mask_approx / 255

        # squeeze contours
        border_cnt = self._img2env_coords(border_cnt_approx.squeeze(), topo_img.shape)
        island_cnts = [
            self._img2env_coords(cnt.squeeze(), topo_img.shape)
            for cnt in island_cnts_approx
        ]

        return border_cnt, island_cnts, land_mask_approx

    def _crop_tiles(self, img, ext, w, s, e, n, ll=True):
        """
        img : ndarray
            Image as a 3D array of RGB values
        ext : tuple
            Bounding box [minX, maxX, minY, maxY] of the returned image
        w : float
            West edge
        s : float
            South edge
        e : float
            East edge
        n : float
            North edge
        ll : Boolean
            [Optional. Default: True] If True, `w`, `s`, `e`, `n` are
            assumed to be lon/lat as opposed to Spherical Mercator.
        """
        # convert lat/lon bounds to Web Mercator XY (EPSG:3857)
        if ll:
            left, bottom = mt.xy(w, s)
            right, top = mt.xy(e, n)
        else:
            left, bottom = w, s
            right, top = e, n

        # determine crop
        X_size = ext[1] - ext[0]
        Y_size = ext[3] - ext[2]

        img_size_x = img.shape[1]
        img_size_y = img.shape[0]

        crop_start_x = ceil(img_size_x * (left - ext[0]) / X_size) - 1
        crop_end_x = ceil(img_size_x * (right - ext[0]) / X_size) - 1

        crop_start_y = ceil(img_size_y * (ext[2] - top) / Y_size)
        crop_end_y = ceil(img_size_y * (ext[2] - bottom) / Y_size) - 1

        # crop image
        cropped_img = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :]

        return cropped_img

    def _img2env_coords(self, cnt, image_shape):
        cnt = cnt.astype(
            float
        )  # convert contour array to float64 so as not to lose precision
        cnt[:, 0] = self.env_size[0] * cnt[:, 0] / (image_shape[1] - 1)
        cnt[:, 1] = self.env_size[1] * (1 - cnt[:, 1] / (image_shape[0] - 1))

        return cnt

    def _border_contour_to_border_obstacles(self, border_cnt):
        border_pt_inds = np.where(self._point_on_border(border_cnt))[0]

        if len(border_pt_inds) == 0:
            return [border_cnt]
        else:
            border_cnt = np.roll(border_cnt, -(border_pt_inds[0] + 1), axis=0)

            contours = []
            current_cnt = [border_cnt[-1]]
            n_cnt_border_pts = 1
            for i, p in enumerate(border_cnt):
                current_cnt.append(p)
                n_cnt_border_pts += self._point_on_border(p)
                if n_cnt_border_pts == 2:
                    if len(current_cnt) > 2:
                        # contour start wall
                        cnt_start_borders = self._point_on_which_border(current_cnt[0])
                        if len(cnt_start_borders) == 2:
                            if 3 in cnt_start_borders and 0 in cnt_start_borders:
                                # moving counterclockwise wall 0 comes after wall 3
                                cnt_start_border = 0
                            else:
                                cnt_start_border = max(cnt_start_borders)
                        else:
                            cnt_start_border = cnt_start_borders[0]

                        # contour end wall
                        cnt_end_borders = self._point_on_which_border(current_cnt[-1])
                        if len(cnt_end_borders) == 2:
                            if 3 in cnt_end_borders and 0 in cnt_end_borders:
                                # moving counterclockwise wall 0 comes after wall 3
                                cnt_end_border = 0
                            else:
                                cnt_end_border = max(cnt_end_borders)
                        else:
                            cnt_end_border = cnt_end_borders[0]

                        # add boundary vertices if necessary
                        if cnt_start_border != cnt_end_border:
                            missing_borders = []
                            current_border = cnt_start_border
                            while current_border != cnt_end_border:
                                missing_borders.append(current_border)
                                current_border = (current_border + 1) % 4

                            for j in missing_borders:
                                current_cnt.insert(0, self.env_vertices[j])

                        contours.append(np.array(current_cnt))

                    # next border contour
                    current_cnt = [p]
                    n_cnt_border_pts = 1

            return contours

    def _point_on_border(self, p):
        """p can be a single point or multiple points"""
        p = np.asarray(p)
        return np.any(p == 0, axis=-1) | np.any(p == self.env_size, axis=-1)

    def _point_on_which_border(self, p):
        """
        p can be a single point or multiple points

        Wall convention:
         _____________ 3 _____________
        |                             |
        |                             |
        |                             |
        0                             2
        |                             |
        |                             |
        |_____________ 1 _____________|

        """
        p = np.asarray(p)
        p_borders_bool = np.concatenate((p == 0, p == self.env_size), axis=-1)

        if p_borders_bool.ndim == 1:
            p_borders = np.where(p_borders_bool)[0]
        else:
            p_borders = [
                np.where(pt_borders_bool)[0] for pt_borders_bool in p_borders_bool
            ]

        return p_borders

    def _segment_on_border(self, segment):
        p1_borders, p2_borders = self._point_on_which_border(segment)
        same_border = (
            len(np.intersect1d(p1_borders, p2_borders, assume_unique=True)) > 0
        )

        return same_border

    def _generate_segments_from_obstacles(self, obstacle, n_quad_segs):
        if isinstance(obstacle, PolygonObstacle):
            vertices = obstacle.anchor_points
        else:  # CircleObstacle
            radius = obstacle.radius
            center = obstacle.center_point
            vertices = list(
                Point(*center).buffer(radius, quad_segs=n_quad_segs).exterior.coords
            )[
                :-1
            ]  # approximate circle with an octagon

        segments = [
            [vertex, vertices[(i + 1) % len(vertices)]]
            for i, vertex in enumerate(vertices)
        ]

        return segments

    def _generate_valid_start_poses(self, land_mask):
        # Conversion factor from mask pixels to environment coordinates
        x_scale = self.env_size[0] / land_mask.shape[1]
        y_scale = self.env_size[1] / land_mask.shape[0]

        # Get coordinates of water pixels in environment units
        water_coords = np.flip(np.column_stack(np.where(land_mask)), axis=-1)
        water_coords[:, 1] = (
            land_mask.shape[0] - water_coords[:, 1]
        )  # adjust y-coords to go from bottom to top
        water_coords_env = (water_coords + 0.5) * [x_scale, y_scale]

        # Create a list of valid positions
        poses_in_collision = detect_collision(
            water_coords_env, self.agent_radius, self.obstacle_geoms
        )
        self.valid_start_poses = water_coords_env[
            np.where(np.logical_not(poses_in_collision))[0]
        ]
        # TODO: valid_team_positions (based on the on sides)
        # TODO: remove start poses that happen to be trapped (maybe do this by using contours with buffer pixel size rounded up based on agent radius)

    def render(self):
        """Overridden method inherited from `Gym`."""
        return self._render()

    def _render(self):
        """
        Overridden method inherited from `Gym`.

        Draws all players/flags/etc on the pygame screen.
        """
        # Create screen
        if self.screen is None:
            pygame.init()

            if self.render_mode:
                self.agent_font = pygame.font.SysFont(
                    None, int(2 * self.agent_render_radius)
                )

                if self.render_mode == "human":
                    pygame.display.set_caption("Capture The Flag")
                    self.screen = pygame.display.set_mode(
                        (self.screen_width, self.screen_height)
                    )
                    self.isopen = True
                elif self.render_mode == "rgb_array":
                    self.screen = pygame.Surface(
                        (self.screen_width, self.screen_height)
                    )
            else:
                raise Exception(
                    f"Sorry, render modes other than f{self.metadata['render_modes']} are not supported"
                )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state == {}:
            return None

        # Background
        self.screen.blit(self.pygame_background_img, (0, 0))

        # Flags
        for team in Team:
            flag = self.flags[int(team)]
            color = "blue" if team == Team.BLUE_TEAM else "red"

            # team flag (not picked up)
            if not self.state["flag_taken"][int(team)]:
                flag_pos_screen = self.env_to_screen(flag.pos)
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    radius=self.flag_render_radius,
                )
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    radius=(self.flag_keepout - self.agent_radius) * self.pixel_size,
                    width=self.boundary_width,
                )

            # team home region
            home_center_screen = self.env_to_screen(self.flags[int(team)].home)
            draw.circle(
                self.screen,
                (128, 128, 128),
                home_center_screen,
                radius=self.catch_radius * self.pixel_size,
                width=self.boundary_width,
            )

        # Trajectories
        if self.render_traj_mode:
            # traj
            if self.render_traj_mode.startswith("traj"):
                for i in range(
                    1, len(self.traj_render_buffer[self.agents[0]]["traj"]) + 1
                ):
                    for player in self.players.values():
                        color = "blue" if player.team == Team.BLUE_TEAM else "red"
                        draw.circle(
                            self.screen,
                            color,
                            self.traj_render_buffer[player.id]["traj"][-i],
                            radius=2,
                            width=0,
                        )
            # agent
            if self.render_traj_mode.endswith("agent"):
                for i in range(
                    1, len(self.traj_render_buffer[self.agents[0]]["agent"]) + 1
                ):
                    for agent_id in self.players:
                        prev_rot_blit_pos, prev_agent_surf = self.traj_render_buffer[
                            agent_id
                        ]["agent"][-i]
                        prev_agent_surf.set_alpha(self.render_transparency_alpha)
                        self.screen.blit(prev_agent_surf, prev_rot_blit_pos)
            # history
            elif self.render_traj_mode.endswith("history"):
                for i in reversed(
                    self.hist_buffer_inds[1:] - 1
                ):  # current state of agent is not included in history buffer
                    for agent_id in self.players:
                        if i < len(self.traj_render_buffer[agent_id]["history"]):
                            prev_rot_blit_pos, prev_agent_surf = (
                                self.traj_render_buffer[agent_id]["history"][i]
                            )
                            render_tranparency = 255 - (
                                (255 - self.render_transparency_alpha)
                                * (i + 1)
                                / (self.hist_buffer_len - 1)
                            )
                            prev_agent_surf.set_alpha(render_tranparency)
                            self.screen.blit(prev_agent_surf, prev_rot_blit_pos)

        # Players
        for team in Team:
            teams_players = self.agents_of_team[team]
            color = "blue" if team == Team.BLUE_TEAM else "red"
            opp_color = "red" if team == Team.BLUE_TEAM else "blue"

            for player in teams_players:
                blit_pos = self.env_to_screen(player.pos)

                # lidar
                if self.lidar_obs and self.render_lidar_mode:
                    ray_headings_global = np.deg2rad(
                        (
                            heading_angle_conversion(player.heading)
                            + self.lidar_ray_headings
                        )
                        % 360
                    )
                    ray_vecs = np.array(
                        [np.cos(ray_headings_global), np.sin(ray_headings_global)]
                    ).T
                    lidar_starts = player.pos + self.agent_radius * ray_vecs
                    for i in range(self.num_lidar_rays):
                        if self.render_lidar_mode == "full" or (
                            self.render_lidar_mode == "detection"
                            and self.state["lidar_labels"][player.id][i]
                            != self.ray_int_label_map["nothing"]
                        ):
                            draw.line(
                                self.screen,
                                color,
                                self.env_to_screen(lidar_starts[i]),
                                self.env_to_screen(
                                    self.state["lidar_ends"][player.id][i]
                                ),
                                width=1,
                            )
                # tagging
                player.render_tagging(self.tagging_cooldown)

                # heading
                orientation = Vector2(list(mag_heading_to_vec(1.0, player.heading)))
                ref_angle = -orientation.angle_to(self.PYGAME_UP)

                # transform position to pygame coordinates
                rotated_surface = rotozoom(player.pygame_agent, ref_angle, 1.0)
                rotated_surface_size = np.array(rotated_surface.get_size())
                rotated_blit_pos = blit_pos - 0.5 * rotated_surface_size

                # flag pickup
                if player.has_flag:
                    draw.circle(
                        rotated_surface,
                        opp_color,
                        0.5 * rotated_surface_size,
                        radius=0.55 * self.agent_render_radius,
                    )

                # agent id
                if self.render_ids:
                    if self.gps_env:
                        font_color = "white"
                    else:
                        font_color = "white" if team == Team.BLUE_TEAM else "black"

                    player_number_label = self.agent_font.render(
                        str(player.id), True, font_color
                    )
                    player_number_label_rect = player_number_label.get_rect()
                    player_number_label_rect.center = (
                        0.5 * rotated_surface_size[0],
                        0.52 * rotated_surface_size[1],
                    )  # Using 0.52 for the y-coordinate because it looks nicer than 0.5
                    rotated_surface.blit(player_number_label, player_number_label_rect)

                # blit agent onto screen
                self.screen.blit(rotated_surface, rotated_blit_pos)

                # save agent surface for trajectory rendering
                if (
                    self.render_traj_mode
                    and self.render_ctr % self.num_renders_per_step == 0
                ):
                    # add traj/ agent render data
                    if self.render_traj_mode.startswith("traj"):
                        self.traj_render_buffer[player.id]["traj"].insert(0, blit_pos)

                    if (
                        self.render_traj_mode.endswith("agent")
                        and (self.render_ctr / self.num_renders_per_step)
                        % self.render_traj_freq
                        == 0
                    ):
                        self.traj_render_buffer[player.id]["agent"].insert(
                            0, (rotated_blit_pos, rotated_surface)
                        )

                    elif (
                        self.render_traj_mode.endswith("history")
                        and self.render_ctr % self.num_renders_per_step == 0
                    ):
                        self.traj_render_buffer[player.id]["history"].insert(
                            0, (rotated_blit_pos, rotated_surface)
                        )

                    # truncate traj
                    if self.render_traj_cutoff is not None:
                        agent_render_cutoff = floor(
                            self.render_traj_cutoff / self.render_traj_freq
                        ) + (
                            (
                                (self.render_ctr / self.num_renders_per_step)
                                % self.render_traj_freq
                                + self.render_traj_freq
                                * floor(self.render_traj_cutoff / self.render_traj_freq)
                            )
                            <= self.render_traj_cutoff
                        )
                        self.traj_render_buffer[player.id]["traj"] = (
                            self.traj_render_buffer[player.id]["traj"][
                                : self.render_traj_cutoff
                            ]
                        )
                        self.traj_render_buffer[player.id]["agent"] = (
                            self.traj_render_buffer[player.id]["agent"][
                                :agent_render_cutoff
                            ]
                        )

                    self.traj_render_buffer[player.id]["history"] = (
                        self.traj_render_buffer[player.id]["history"][
                            : self.hist_buffer_len
                        ]
                    )

        # Agent-to-agent distances
        for blue_player in self.agents_of_team[Team.BLUE_TEAM]:
            if not blue_player.is_tagged or (
                blue_player.is_tagged and blue_player.on_own_side
            ):
                for red_player in self.agents_of_team[Team.RED_TEAM]:
                    if not red_player.is_tagged or (
                        red_player.is_tagged and red_player.on_own_side
                    ):
                        blue_player_pos = np.asarray(blue_player.pos)
                        red_player_pos = np.asarray(red_player.pos)
                        a2a_dis = np.linalg.norm(blue_player_pos - red_player_pos)
                        if a2a_dis <= 2 * self.catch_radius:
                            hsv_hue = (a2a_dis - self.catch_radius) / (
                                2 * self.catch_radius - self.catch_radius
                            )
                            hsv_hue = 0.33 * np.clip(hsv_hue, 0, 1)
                            line_color = tuple(
                                255 * np.asarray(colorsys.hsv_to_rgb(hsv_hue, 0.9, 0.9))
                            )

                            draw.line(
                                self.screen,
                                line_color,
                                self.env_to_screen(blue_player_pos),
                                self.env_to_screen(red_player_pos),
                                width=self.a2a_line_width,
                            )

        # Render
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        # Record
        if self.render_saving:
            self.render_buffer[self.render_ctr + 1] = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        # Update counter
        self.render_ctr += 1

    def env_to_screen(self, pos):
        screen_pos = self.pixel_size * np.asarray(pos)
        screen_pos[0] += self.arena_buffer
        screen_pos[1] = self.arena_height - screen_pos[1] + self.arena_buffer

        return screen_pos

    def buffer_to_video(self, recording_compression=False):
        """Convert and save current render buffer as a video"""
        if not self.render_saving:
            print(
                "Warning: Environment rendering is disabled. Cannot save the video. See the render_saving option in the config."
            )
            print()
        elif self.render_ctr > 0:
            video_file_dir = str(pathlib.Path(__file__).resolve().parents[1] / "videos")
            if not os.path.isdir(video_file_dir):
                os.mkdir(video_file_dir)

            now = datetime.now()  # get date and time
            video_id = now.strftime("%m-%d-%Y_%H-%M-%S")

            video_file_name = f"pyquaticus_{video_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_file_path = os.path.join(video_file_dir, video_file_name)
            out = cv2.VideoWriter(
                video_file_path,
                fourcc,
                self.render_fps,
                (self.screen_width, self.screen_height),
            )
            for img in self.render_buffer:
                out.write(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))

            out.release()

            if recording_compression:
                compressed_video_file_name = f"pyquaticus_{video_id}_compressed.mp4"

                compressed_video_file_path = os.path.join(
                    video_file_dir, compressed_video_file_name
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-i",
                        video_file_path,
                        "-c:v",
                        "libx264",
                        compressed_video_file_path,
                    ]
                )
        else:
            print("Attempted to save video but render_buffer is empty!")
            print()

    def save_screenshot(self):
        """ "Save an image of the most recently rendered env."""

        if not self.render_saving:
            print(
                "Warning: Environment rendering is disabled. Cannot save the screenshot. See the render_saving option in the config."
            )
            print()
        elif self.render_mode is not None:
            # we may not need to check the self.screen here
            image_file_dir = str(
                pathlib.Path(__file__).resolve().parents[1] / "screenshots"
            )
            if not os.path.isdir(image_file_dir):
                os.mkdir(image_file_dir)

            now = datetime.now()  # get date and time
            image_id = now.strftime("%m-%d-%Y_%H-%M-%S")
            image_file_name = f"pyquaticus_{image_id}.png"
            image_file_path = os.path.join(image_file_dir, image_file_name)

            cv2.imwrite(
                image_file_path,
                cv2.cvtColor(self.render_buffer[self.render_ctr], cv2.COLOR_RGB2BGR),
            )

        else:
            raise Exception(
                "Envrionment was not rendered. See the render_mode option in the config."
            )

    def close(self):
        """Overridden method inherited from `Gym`."""
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

    def _min(self, a, b):
        """Convenience method for determining a minimum value. The standard `min()` takes much longer to run."""
        if a < b:
            return a
        else:
            return b

    def state_to_obs(self, agent_id, normalize=True):
        """
        Modified method to convert the state to agent observations. In addition to the
        logic performed in the superclass state_to_obs, this method adds the distance
        and bearing to obstacles into the observation and then performs the
        normalization.

        Args:
            agent_id: The agent who's observation is being generated
            normalize: Flag to normalize the values in the observation
        Returns
            A dictionary containing the agents observation
        """
        orig_obs = super().state_to_obs(agent_id, normalize=False)

        if not self.lidar_obs:
            # Obstacle Distance/Bearing
            for i, obstacle in enumerate(
                self.state["dist_bearing_to_obstacles"][agent_id]
            ):
                orig_obs[f"obstacle_{i}_distance"] = obstacle[0]
                orig_obs[f"obstacle_{i}_bearing"] = obstacle[1]

        if normalize:
            orig_obs = self.agent_obs_normalizer.normalized(orig_obs)

        return orig_obs
