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
import copy
import cv2
import functools
import itertools
import math
import numpy as np
import os
import pathlib
import pickle
import pygame
import random
import subprocess

from abc import ABC
from datetime import datetime
from gymnasium.spaces import Box, Discrete
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
    EPSG_3857_EXT_X,
    EPSG_3857_EXT_Y,
    get_afp,
    LINE_INTERSECT_TOL,
    lidar_detection_classes,
    LIDAR_DETECTION_CLASS_MAP,
    POLAR_RADIUS
)
from pyquaticus.dynamics.dynamics_registry import dynamics_registry
from pyquaticus.dynamics.dynamics import Dynamics
from pyquaticus.structs import (
    CircleObstacle,
    Flag,
    PolygonObstacle,
    Team
)
from pyquaticus.utils.obs_utils import ObsNormalizer
from pyquaticus.utils.pid import PID
from pyquaticus.utils.utils import (
    angle180,
    check_segment_intersections,
    clip,
    closest_line,
    closest_point_on_line,
    crop_tiles,
    vector_to,
    detect_collision,
    flatten_generic,
    get_rot_angle,
    get_screen_res,
    heading_angle_conversion,
    longitude_diff_west2east,
    mag_bearing_to,
    mag_heading_to_vec,
    rc_intersection,
    reflect_vector,
    rigid_transform,
    rot2d,
    vec_to_mag_heading,
    wrap_mercator_x,
    wrap_mercator_x_dist
)
from scipy.ndimage import label
from shapely import intersection, LineString, Point, Polygon
from typing import Optional, Union


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

    ### Action Space Options

    Discrete action space ('discrete'): all combinations of max speed, half speed; and 45 degree heading intervals
    Continuous action space ('continuous'): speed from 0 to max speed, desired relative headings from -180 to 180
    Aquaticus field points ('afp'): strings (see config.py) indicating desired pos from aquaticus point field

    ### Observation Space

    Default Observation Space (per agent):
        - Opponent home relative bearing (clockwise degrees)
        - Opponent home distance (meters)
        - Home relative bearing (clockwise degrees)
        - Home distance (meters)
        - Wall 0 relative bearing (clockwise degrees)
        - Wall 0 distance (meters)
        - Wall 1 relative bearing (clockwise degrees)
        - Wall 1 distance (meters)
        - Wall 2 relative bearing (clockwise degrees)
        - Wall 2 distance (meters)
        - Wall 3 relative bearing (clockwise degrees)
        - Wall 3 distance (meters)
        - Scrimmage line bearing (clockwise degrees)
        - Scrimmage line distance (meters)
        - Own speed (meters per second)
        - Has flag status (boolean)
        - On side status (boolean)
        - Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
        - Is tagged status (boolean)
        - Team score (cummulative flag captures by agent's team)
        - Opponent score (cummulative flag captures by opposing team)
        - For each other agent (teammates first):
            - Bearing from you (clockwise degrees)
            - Distance (meters)
            - Heading of other agent relative to the vector to you (clockwise degrees)
            - Speed (meters per second)
            - Has flag status (boolean)
            - On side status (boolean)
            - Tagging cooldown (seconds)
            - Is tagged status (boolean)

    Lidar Observation Space (per agent):
        - Opponent home relative bearing (clockwise degrees)
        - Opponent home distance (meters)
        - Home relative bearing (clockwise degrees)
        - Home distance (meters)
        - Scrimmage line bearing (clockwise degrees)
        - Scrimmage line distance (meters)
        - Own speed (meters per second)
        - Has flag status (boolean)
        - Team has opponent's flag status (boolean)
        - Opponent has team's flag status (boolean)
        - On side status (boolean)
        - Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
        - Is tagged status (boolean)
        - Team score (cummulative flag captures by agent's team)
        - Opponent score (cummulative flag captures by opposing team)
        - Lidar ray distances (meters)
        - Lidar ray labels (see lidar_detection_classes in config.py)

    Note 1: the angles are 0 when the agent is pointed directly at the object
            and increase in the clockwise direction
    Note 2: when normalized, the boolean args are -1 False and +1 True
    Note 3: the values are normalized by default
    Note 4: units with 'meters' are either in actual meters or mercator xy meters depending if
            self.gps_env is True or not (except for speed which is always meters per second)

    Developer Note 1: changes here should be reflected in _register_state_elements.
    Developer Note 2: check that variables used here are available to PyQuaticusMoosBridge in pyquaticus_moos_bridge.py
    """
    def _seed(self, seed=None):
        """
        Handles numpy and python random seeding.

        Adapted from Gymnasium 1.1.1:
            https://github.com/Farama-Foundation/Gymnasium/blob/1c7c709f6bb3bfc4e8928dc40752780c3d89b965/gymnasium/core.py#L157

        Args:
            seed (optional): starting seed
        """
        if seed is not None:
            random.seed(seed)
            self._np_random, self._np_random_seed = seeding.np_random(seed)

    def _to_speed_heading(self, raw_action, player, act_space_match, act_space_str):
        """
        Processes the raw action for a player object (acge)

        Args:
            player: Player object
            raw_action: discrete (int), continuous (array), or afp (str) action

        Returns:
            dict from agent id -> (speed, relative heading)
            Note: we use relative heading here so that it can be used directly
                  as the heading error in the PID controller
        """
        if act_space_match:
            # Continuous actions
            if act_space_str == "continuous":
                speed = raw_action[0] * self.max_speeds[player.idx]
                rel_heading = raw_action[1] * 180
            # Discrete action space
            elif act_space_str == "discrete":    
                speed, rel_heading = self._discrete_action_to_speed_relheading(raw_action)
                speed = self.max_speeds[player.idx] * speed #scale speed to agent's max speed
            # Aquaticus point field
            else:
                speed, rel_heading = self._afp_to_speed_relheading(raw_action, player)
        else:
            # Continuous actions
            if isinstance(raw_action, (list, tuple, np.ndarray)):
                speed = raw_action[0] * self.max_speeds[player.idx]
                rel_heading = raw_action[1] * 180
            # Aquaticus point field
            elif isinstance(raw_action, str):
                speed, rel_heading = self._afp_to_speed_relheading(raw_action, player)
            # Discrete action space
            else:
                speed, rel_heading = self._discrete_action_to_speed_relheading(raw_action)
                speed = self.max_speeds[player.idx] * speed #scale speed to agent's max speed

        return speed, rel_heading

    def _discrete_action_to_speed_relheading(self, action):
        return self.discrete_action_map[action]

    def _afp_to_speed_relheading(self, raw_action, agent):
        #make aquaticus point field the same on both blue and red sides
        if agent.team == Team.RED_TEAM:
            if "P" in raw_action:
                raw_action = "S" + raw_action[1:]
            elif "S" in raw_action:
                raw_action = "P" + raw_action[1:]
            if "X" not in raw_action and raw_action not in ["SC", "CC", "PC"]:
                raw_action += "X"
            elif raw_action not in ["SC", "CC", "PC"]:
                raw_action = raw_action[:-1]

        _, rel_heading = mag_bearing_to(
            agent.pos, self.aquaticus_field_points[raw_action], agent.heading
        )
        if self.get_distance_between_2_points(agent.pos, self.aquaticus_field_points[raw_action]) <= self.agent_radius[agent.idx]:
            speed = 0.0
        else:
            speed = self.max_speeds[agent.idx]

        return speed, rel_heading

    def _relheading_to_global_heading(self, player_heading, relheading):
        return angle180((player_heading + relheading) % 360)

    def _register_state_elements(self, num_on_team, num_obstacles, n_envs):
        """Initializes the normalizers."""
        agent_obs_normalizer = ObsNormalizer(False, n_envs)
        global_state_normalizer = ObsNormalizer(False, n_envs)

        ### Agent Observation Normalizer ###
        if self.lidar_obs:
            max_bearing = [180]
            max_dist = [self.env_diag]
            min_dist = [0.0]
            max_dist_lidar = self.num_lidar_rays * [self.lidar_range]
            min_dist_lidar = self.num_lidar_rays * [0.0]
            max_bool, min_bool = [1.0], [0.0]
            max_speed, min_speed = [max(self.max_speeds)], [0.0]
            max_lidar_label = self.num_lidar_rays * [len(LIDAR_DETECTION_CLASS_MAP) - 1]
            min_lidar_label = self.num_lidar_rays * [0.0]

            agent_obs_normalizer.register("opponent_home_bearing", max_bearing)
            agent_obs_normalizer.register("opponent_home_distance", max_dist, min_dist)
            agent_obs_normalizer.register("own_home_bearing", max_bearing)
            agent_obs_normalizer.register("own_home_distance", max_dist, min_dist)
            agent_obs_normalizer.register("scrimmage_line_bearing", max_bearing)
            agent_obs_normalizer.register("scrimmage_line_distance", max_dist, [0.0])
            agent_obs_normalizer.register("speed", max_speed, min_speed)
            agent_obs_normalizer.register("has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("team_has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("opponent_has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("on_side", max_bool, min_bool)
            agent_obs_normalizer.register("tagging_cooldown", [self.tagging_cooldown], [0.0])
            agent_obs_normalizer.register("is_tagged", max_bool, min_bool)
            agent_obs_normalizer.register("ray_distances", max_dist_lidar, min_dist_lidar)
            agent_obs_normalizer.register("ray_labels", max_lidar_label, min_lidar_label)

        else:
            max_bearing = [180]
            max_dist = [self.env_diag]
            min_dist = [0.0]
            max_bool, min_bool = [1.0], [0.0]
            max_speed, min_speed = [max(self.max_speeds)], [0.0]

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
            agent_obs_normalizer.register("tagging_cooldown", [self.tagging_cooldown], [0.0])
            agent_obs_normalizer.register("is_tagged", max_bool, min_bool)
            agent_obs_normalizer.register("out_of_bounds", max_bool, min_bool)
            agent_obs_normalizer.register("in_flag_keepout", max_bool, min_bool)

            for i in range(num_on_team - 1):
                teammate_name = f"teammate_{i}"
                agent_obs_normalizer.register((teammate_name, "bearing"), max_bearing)
                agent_obs_normalizer.register((teammate_name, "distance"), max_dist, min_dist)
                agent_obs_normalizer.register((teammate_name, "relative_heading"), max_bearing)
                agent_obs_normalizer.register((teammate_name, "speed"), max_speed, min_speed)
                agent_obs_normalizer.register((teammate_name, "has_flag"), max_bool, min_bool)
                agent_obs_normalizer.register((teammate_name, "on_side"), max_bool, min_bool)
                agent_obs_normalizer.register((teammate_name, "tagging_cooldown"), [self.tagging_cooldown], [0.0])
                agent_obs_normalizer.register((teammate_name, "is_tagged"), max_bool, min_bool)
                agent_obs_normalizer.register((teammate_name, "out_of_bounds"), max_bool, min_bool)
                agent_obs_normalizer.register((teammate_name, "in_flag_keepout"), max_bool, min_bool)

            for i in range(num_on_team):
                opponent_name = f"opponent_{i}"
                agent_obs_normalizer.register((opponent_name, "bearing"), max_bearing)
                agent_obs_normalizer.register((opponent_name, "distance"), max_dist, min_dist)
                agent_obs_normalizer.register((opponent_name, "relative_heading"), max_bearing)
                agent_obs_normalizer.register((opponent_name, "speed"), max_speed, min_speed)
                agent_obs_normalizer.register((opponent_name, "has_flag"), max_bool, min_bool)
                agent_obs_normalizer.register((opponent_name, "on_side"), max_bool, min_bool)
                agent_obs_normalizer.register((opponent_name, "tagging_cooldown"), [self.tagging_cooldown], [0.0])
                agent_obs_normalizer.register((opponent_name, "is_tagged"), max_bool, min_bool)
                agent_obs_normalizer.register((opponent_name, "out_of_bounds"), max_bool, min_bool)
                agent_obs_normalizer.register((opponent_name, "in_flag_keepout"), max_bool, min_bool)

            for i in range(num_obstacles):
                agent_obs_normalizer.register(f"obstacle_{i}_distance", max_dist, min_dist)
                agent_obs_normalizer.register(f"obstacle_{i}_bearing", max_bearing)

        ### Global State Normalizer ###
        max_heading = [180]
        max_bearing = [180]
        pos_max = self.env_size + 5*max(self.agent_radius) #add a normalization buffer
        pos_min = len(self.env_size) * [-5*max(self.agent_radius)] #add a normalization buffer
        max_dist = [self.env_diag]
        min_dist = [0.0]
        max_bool, min_bool = [1.0], [0.0]
        max_speed, min_speed = [max(self.max_speeds)], [0.0]

        global_state_normalizer.register("blue_flag_home", pos_max, pos_min)
        global_state_normalizer.register("red_flag_home", pos_max, pos_min)

        for player in self.players.values():
            player_name = player.id

            global_state_normalizer.register((player_name, "pos"), pos_max, pos_min)
            global_state_normalizer.register((player_name, "heading"), max_heading)
            global_state_normalizer.register((player_name, "speed"), max_speed, min_speed)
            global_state_normalizer.register((player_name, "has_flag"), max_bool, min_bool)
            global_state_normalizer.register((player_name, "on_side"), max_bool, min_bool)
            global_state_normalizer.register((player_name, "tagging_cooldown"), [self.tagging_cooldown], [0.0])
            global_state_normalizer.register((player_name, "is_tagged"), max_bool, min_bool)
            global_state_normalizer.register((player_name, "out_of_bounds"), max_bool, min_bool)
            global_state_normalizer.register((player_name, "in_flag_keepout"), max_bool, min_bool)

            for i in range(num_obstacles):
                global_state_normalizer.register((player_name, f"obstacle_{i}_distance"), max_dist, min_dist)
                global_state_normalizer.register((player_name, f"obstacle_{i}_bearing"), max_bearing)

        return agent_obs_normalizer, global_state_normalizer

    def state_to_obs(self, agent_id, normalize=True, env_idxs=None):
        """
        Returns a local observation space. These observations are
        based entirely on the agent local coordinate frame rather
        than the world frame.

        This was originally designed so that observations can be
        easily shared between different teams and agents.
        Without this the world frame observations from the blue and
        red teams are flipped (e.g., the goal is in the opposite
        direction)

        Default Observation Space (per agent):
            - Opponent home relative bearing (clockwise degrees)
            - Opponent home distance (meters)
            - Home relative bearing (clockwise degrees)
            - Home distance (meters)
            - Wall 0 relative bearing (clockwise degrees)
            - Wall 0 distance (meters)
            - Wall 1 relative bearing (clockwise degrees)
            - Wall 1 distance (meters)
            - Wall 2 relative bearing (clockwise degrees)
            - Wall 2 distance (meters)
            - Wall 3 relative bearing (clockwise degrees)
            - Wall 3 distance (meters)
            - Scrimmage line bearing (clockwise degrees)
            - Scrimmage line distance (meters)
            - Own speed (meters per second)
            - Has flag status (boolean)
            - On side status (boolean)
            - Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            - Is tagged status (boolean)
            - Team score (cummulative flag captures by agent's team)
            - Opponent score (cummulative flag captures by opposing team)
            - For each other agent (teammates first):
              - Bearing from you (clockwise degrees)
              - Distance (meters)
              - Heading of other agent relative to the vector to you (clockwise degrees)
              - Speed (meters per second)
              - Has flag status (boolean)
              - On side status (boolean)
              - Tagging cooldown (seconds)
              - Is tagged status (boolean)

        Lidar Observation Space (per agent):
            - Opponent home relative bearing (clockwise degrees)
            - Opponent home distance (meters)
            - Home relative bearing (clockwise degrees)
            - Home distance (meters)
            - Scrimmage line bearing (clockwise degrees)
            - Scrimmage line distance (meters)
            - Own speed (meters per second)
            - Has flag status (boolean)
            - Team has opponent's flag status (boolean)
            - Opponent has team's flag status (boolean)
            - On side status (boolean)
            - Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            - Is tagged status (boolean)
            - Team score (cummulative flag captures by agent's team)
            - Opponent score (cummulative flag captures by opposing team)
            - Lidar ray distances (meters)
            - Lidar ray labels (see lidar_detection_classes in config.py)

        Note 1: the angles are 0 when the agent is pointed directly at the object
                and increase in the clockwise direction
        Note 2: when normalized, the boolean args are -1 False and +1 True
        Note 3: the values are normalized by default
        Note 4: units with 'meters' are either in actual meters or mercator xy meters depending if
                self.gps_env is True or not (except for speed which is always meters per second)

        Developer Note 1: changes here should be reflected in _register_state_elements.
        Developer Note 2: check that variables used here are available to PyQuaticusMoosBridge in pyquaticus_moos_bridge.py
        Developer Note 3: assumes there are only 2 teams (blue and red) and one flag per team
        """
        obs = dict()
        agent = self.players[agent_id]

        own_team = agent.team
        other_team = Team.BLUE_TEAM if own_team == Team.RED_TEAM else Team.RED_TEAM

        team_idx = int(own_team)
        other_team_idx = int(other_team)

        pos = self.state["agent_position"][env_idxs, agent.idx]
        heading = self.state["agent_heading"][env_idxs, agent.idx]

        own_home_loc = self.flags[team_idx].home
        opponent_home_loc = self.flags[other_team_idx].home

        if self.lidar_obs:
            raise NotImplementedError("Vector environment with Lidar not implemented.")
            # # Goal flag
            # opponent_home_dist, opponent_home_bearing = mag_bearing_to(
            #     pos, opponent_home_loc, heading
            # )
            # obs["opponent_home_bearing"] = opponent_home_bearing
            # obs["opponent_home_distance"] = opponent_home_dist

            # # Defend flag
            # own_home_dist, own_home_bearing = mag_bearing_to(
            #     pos, own_home_loc, heading
            # )
            # obs["own_home_bearing"] = own_home_bearing
            # obs["own_home_distance"] = own_home_dist

            # # Scrimmage line
            # scrimmage_line_closest_point = closest_point_on_line(
            #     self.scrimmage_coords[0], self.scrimmage_coords[1], pos
            # )
            # scrimmage_line_dist, scrimmage_line_bearing = mag_bearing_to(
            #     pos, scrimmage_line_closest_point, heading
            # )
            # obs["scrimmage_line_bearing"] = scrimmage_line_bearing
            # obs["scrimmage_line_distance"] = scrimmage_line_dist

            # # Own speed
            # obs["speed"] = self.state["agent_speed"][agent.idx]
            # # Own flag status
            # obs["has_flag"] = self.state["agent_has_flag"][agent.idx]
            # # Team has flag
            # obs["team_has_flag"] = self.state["flag_taken"][other_team_idx]
            # # Opposing team has flag
            # obs["opponent_has_flag"] = self.state["flag_taken"][team_idx]
            # # On sides
            # obs["on_side"] = self.state["agent_on_sides"][agent.idx]
            # # Tagging cooldown
            # obs["tagging_cooldown"] = self.state["agent_tagging_cooldown"][agent.idx]
            # # Is tagged
            # obs["is_tagged"] = self.state["agent_is_tagged"][agent.idx]

            # # Lidar
            # obs["ray_distances"] = self.state["lidar_distances"][agent_id]
            # obs["ray_labels"] = self.obj_ray_detection_states[own_team][self.state["lidar_labels"][agent_id]]

        else:
            # Goal flag
            opponent_home_dist, opponent_home_bearing = mag_bearing_to(
                pos, opponent_home_loc, heading
            )
            obs["opponent_home_bearing"] = opponent_home_bearing
            obs["opponent_home_distance"] = opponent_home_dist

            # Defend flag
            own_home_dist, own_home_bearing = mag_bearing_to(
                pos, own_home_loc, heading
            )
            obs["own_home_bearing"] = own_home_bearing
            obs["own_home_distance"] = own_home_dist

            # Walls
            for i, wall in enumerate(self._walls[team_idx]):
                wall_closest_point = closest_point_on_line(
                    wall[0], wall[1], pos
                )
                wall_dist, wall_bearing = mag_bearing_to(
                    pos, wall_closest_point, heading
                )
                obs[f"wall_{i}_bearing"] = wall_bearing
                obs[f"wall_{i}_distance"] = wall_dist

            # Scrimmage line
            scrimmage_line_closest_point = closest_point_on_line(
                self.scrimmage_coords[0], self.scrimmage_coords[1], pos
            )
            scrimmage_line_dist, scrimmage_line_bearing = mag_bearing_to(
                pos, scrimmage_line_closest_point, heading
            )
            obs["scrimmage_line_bearing"] = scrimmage_line_bearing
            obs["scrimmage_line_distance"] = scrimmage_line_dist

            # Own speed
            obs["speed"] = self.state["agent_speed"][env_idxs, agent.idx]
            # Own flag status
            obs["has_flag"] = self.state["agent_has_flag"][env_idxs, agent.idx]
            # On side
            obs["on_side"] = self.state["agent_on_sides"][env_idxs, agent.idx]
            # Tagging cooldown
            obs["tagging_cooldown"] = self.state["agent_tagging_cooldown"][env_idxs, agent.idx]
            # Is tagged
            obs["is_tagged"] = self.state["agent_is_tagged"][env_idxs, agent.idx]
            # Out-of-bounds
            obs["out_of_bounds"] = self.state["agent_oob"][env_idxs, agent.idx]
            # In flag keepout
            obs["in_flag_keepout"] = self.state["agent_in_flag_keepout"][env_idxs, agent.idx]

            # Relative observations to other agents (teammates first)
            for team in [own_team, other_team]:
                dif_agents = filter(lambda a: a.id != agent.id, self.agents_of_team[team])
                for i, dif_agent in enumerate(dif_agents):
                    entry_name = f"teammate_{i}" if team == own_team else f"opponent_{i}"

                    dif_pos = self.state["agent_position"][env_idxs, dif_agent.idx]
                    dif_heading = self.state["agent_heading"][env_idxs, dif_agent.idx]

                    dif_agent_dist, dif_agent_bearing = mag_bearing_to(pos, dif_pos, heading)
                    _, hdg_to_agent = mag_bearing_to(dif_pos, pos)
                    hdg_to_agent = hdg_to_agent % 360

                    obs[(entry_name, "bearing")] = dif_agent_bearing #bearing relative to the bearing to you
                    obs[(entry_name, "distance")] = dif_agent_dist
                    obs[(entry_name, "relative_heading")] = angle180((dif_heading - hdg_to_agent) % 360)
                    obs[(entry_name, "speed")] = self.state["agent_speed"][env_idxs, dif_agent.idx]
                    obs[(entry_name, "has_flag")] = self.state["agent_has_flag"][env_idxs, dif_agent.idx]
                    obs[(entry_name, "on_side")] = self.state["agent_on_sides"][env_idxs, dif_agent.idx]
                    obs[(entry_name, "tagging_cooldown")] = self.state["agent_tagging_cooldown"][env_idxs, dif_agent.idx]
                    obs[(entry_name, "is_tagged")] = self.state["agent_is_tagged"][env_idxs, dif_agent.idx]
                    obs[(entry_name, "out_of_bounds")] = self.state["agent_oob"][env_idxs, dif_agent.idx]
                    obs[(entry_name, "in_flag_keepout")] = self.state["agent_in_flag_keepout"][env_idxs, dif_agent.idx]

        if normalize:
            return self.agent_obs_normalizer.normalized(obs), obs
        else:
            return obs, None
    
    def state_to_global_state(self, env_idxs, normalize=True):
        """
        Returns a global environment state:
            - Agent 0:
              - Position (xy meters)
              - Heading (clockwise degrees where North is 0)
              - Scrimmage line bearing (clockwise degrees)
              - Scrimmage line distance (meters)
              - Own speed (meters per second)
              - Has flag status (boolean)
              - On side status (boolean)
              - Out of bounds (boolean)
              - Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
              - Is tagged status (boolean)
              - Relative distance and bearing to each obstacle (meters and clockwise degrees respectively)
            - Agent 1: same as Agent 0
            - Agent 2: same as Agent 0
            .
            .
            .
            - Agent n: same as Agent 0
            - Blue flag home (xy meters)
            - Red flag home (xy meters)
            - Blue flag position (xy meters)
            - Red flag position (xy meters)
            - Blue flag pickup (boolean)
            - Red flag pickup (boolean)
            - Blue team score (cummulative flag captures by blue team)
            - Red team score (cummulative flag captures by red team)

        Note 1: the angles are 0 when the agent is pointed directly at the object
                and increase in the clockwise direction
        Note 2: when normalized, the boolean args are -1 False and +1 True
        Note 3: the values are normalized by default
        Note 4: units with 'meters' are either in actual meters or mercator xy meters depending if
                self.gps_env is True or not (except for speed which is always meters per second)

        Developer Note 1: changes here should be reflected in _register_state_elements.
        Developer Note 2: check that variables used here are available to PyQuaticusMoosBridge in pyquaticus_moos_bridge.py
        """
        global_state = dict()

        # agent info
        for i, (agent_id, agent) in enumerate(self.players.items()):
            pos = self.state["agent_position"][env_idxs, agent.idx]
            heading = self.state["agent_heading"][env_idxs, agent.idx]

            global_state[(agent_id, "pos")] = self._standard_pos(pos)
            global_state[(agent_id, "heading")] = self._standard_heading(heading)
            global_state[(agent_id, "speed")] = self.state["agent_speed"][env_idxs, agent.idx]
            global_state[(agent_id, "has_flag")] = self.state["agent_has_flag"][env_idxs, agent.idx]
            global_state[(agent_id, "on_side")] = self.state["agent_on_sides"][env_idxs, agent.idx]
            global_state[(agent_id, "tagging_cooldown")] = self.state["agent_tagging_cooldown"][env_idxs, agent.idx]
            global_state[(agent_id, "is_tagged")] = self.state["agent_is_tagged"][env_idxs, agent.idx]
            global_state[(agent_id, "out_of_bounds")] = self.state["agent_oob"][env_idxs, agent.idx]
            global_state[(agent_id, "in_flag_keepout")] = self.state["agent_in_flag_keepout"][env_idxs, agent.idx]

            #TODO: fix obstacle state
            #obstacle distance/bearing
            # for i, obstacle in enumerate(
            #     self.state["dist_bearing_to_obstacles"][agent_id]
            # ):
            #     global_state[(agent_id, f"obstacle_{i}_distance")] = obstacle[0]
            #     global_state[(agent_id, f"obstacle_{i}_bearing")] = obstacle[1]

        # flag info
        global_state["blue_flag_home"] = self._standard_pos(self.flags[int(Team.BLUE_TEAM)].home)
        global_state["red_flag_home"] = self._standard_pos(self.flags[int(Team.RED_TEAM)].home)

        if normalize:
            return self.global_state_normalizer.normalized(global_state)
        else:
            return global_state

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id: str):
        return self.action_spaces[agent_id]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id: str):
        return self.observation_spaces[agent_id]

    def get_agent_action_space(self, action_space: str, agent_idx: int):
        """Legacy Gym method"""
        if action_space == "discrete":    
            return Discrete(len(self.discrete_action_map))
        elif action_space == "continuous":
            return Box(
                low=np.array([0, -1], dtype=np.float32), #speed, relative heading
                high=np.array([1, 1], dtype=np.float32) #speed, relative heading
            )
        elif action_space == "afp":
            return Discrete(len(self.aquaticus_field_points))
        else:
            raise Exception(f"Action space type '{action_space}' not recognized. Choose from: 'discrete', 'continuous', or 'afp'")

    def get_agent_observation_space(self):
        """Legacy Gym method"""
        if self.normalize_obs:
            return self.agent_obs_normalizer.normalized_space
        else:
            return self.agent_obs_normalizer.unnormalized_space

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
            [self.env_ul, self.env_ur],
            [self.env_ur, self.env_lr],
            [self.env_lr, self.env_ll],
            [self.env_ll, self.env_ul]
        ]

        def rotate_walls(walls, amt):
            rot_walls = copy.deepcopy(walls)
            return rot_walls[amt:] + rot_walls[:amt]

        # determine orientation for each team
        blue_flag = self.flags[int(Team.BLUE_TEAM)].home
        red_flag  = self.flags[int(Team.RED_TEAM)].home

        team_flags_midpoint = (blue_flag + red_flag)/2

        blue_wall_vec = blue_flag - team_flags_midpoint
        blue_wall_ray_end = team_flags_midpoint + self.env_diag * (blue_wall_vec / np.linalg.norm(blue_wall_vec))
        blue_wall_ray = np.asarray([team_flags_midpoint, blue_wall_ray_end])

        red_wall_vec = red_flag - team_flags_midpoint
        red_wall_ray_end = team_flags_midpoint + self.env_diag * (red_wall_vec / np.linalg.norm(red_wall_vec))
        red_wall_ray = np.asarray([team_flags_midpoint, red_wall_ray_end])

        edges_reordered = np.roll(self.env_edges, 1, axis=0) #match _point_on_which_border() wall ordering
        blue_borders = check_segment_intersections(edges_reordered, blue_wall_ray)
        red_borders = check_segment_intersections(edges_reordered, red_wall_ray)

        if len(blue_borders) == len(red_borders) == 2:
            #blue wall
            if 3 in blue_borders and 0 in blue_borders:
                blue_border = 0
            else:
                blue_border = max(blue_borders)
            #red wall
            if 3 in red_borders and 0 in red_borders:
                red_border = 0
            else:
                red_border = max(red_borders)
        elif len(blue_borders) == 2:
            red_border = red_borders[0]
            if 3 in blue_borders and 0 in blue_borders:
                blue_border = 3
            else:
                blue_border = min(blue_borders)
        elif len(red_borders) == 2:
            blue_border = blue_borders[0]
            if 3 in red_borders and 0 in red_borders:
                red_border = 3
            else:
                red_border = min(red_borders)
        else:
            blue_border = blue_borders[0]
            red_border = red_borders[0]

        blue_wall = 3 - blue_border #converting to corresponding wall idx within all_walls for backwards compatibility
        red_wall = 3 - red_border #converting to corresponding wall idx within all_walls for backwards compatibility

        blue_rot_amt = blue_wall - 1 #wall 1 is the flag wall (see wall ordering in function description)
        red_rot_amt = red_wall - 1 #wall 1 is the flag wall (see wall ordering in function description)

        self._walls = {}
        self._walls[int(Team.BLUE_TEAM)] = rotate_walls(all_walls, blue_rot_amt)
        self._walls[int(Team.RED_TEAM)] = rotate_walls(all_walls, red_rot_amt)

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
        p = self._standard_pos(p)
        p_borders_bool = np.concatenate((p == 0, p == self.env_size), axis=-1)

        if p_borders_bool.ndim == 1:
            p_borders = np.where(p_borders_bool)[0]
        else:
            p_borders = [np.where(pt_borders_bool)[0] for pt_borders_bool in p_borders_bool]

        return p_borders

    def _standard_pos(self, pos):
        """
        Converts pos into env reference frame based on the boundary.
        """
        return rigid_transform(pos, self.env_ll, self.env_rot_matrix.T)

    def _standard_heading(self, heading):
        """
        Converts heading into env reference frame based on the boundary.
        """
        return angle180(heading + np.rad2deg(self.env_rot_angle)) #nautical headings are cw (not ccw)

    def _check_on_sides(self, pos, team):
        """pos can be a single point or multiple points"""
        scrim2pos = np.asarray(pos) - self.scrimmage_coords[0]
        cp_sign = np.sign(np.cross(self.scrimmage_vec, scrim2pos))

        return (cp_sign == self._on_sides_sign[team]) | (cp_sign == 0)

    def _check_agent_collisions(self, env_idxs):
        """
        Updates game state attribute agent_collisions
        Note: Checks collisions between all players teammates and opponents
        """
        agent_poses = self.state['agent_position'][env_idxs]
        dists = np.linalg.norm(agent_poses[:, :, None, :] - agent_poses[:, None, :, :], axis=-1)
        sum_agent_radii = self.agent_radius[:, :, None] + self.agent_radius[:, None, :]

        active_collisions = (dists <= sum_agent_radii) & (~np.eye(self.num_agents, dtype=bool)) #remove self-collisions
        new_active_collisions = active_collisions & ~self.active_collisions

        for team, agent_idxs in self.agent_inds_of_team.items():
            env_agent_ixgrid = np.ix_(env_idxs, agent_idxs)
            self.state["agent_collisions"][env_agent_ixgrid] += np.sum(new_active_collisions[env_agent_ixgrid], axis=-1)
            self.game_events[team]["collisions"][env_idxs] += np.sum(new_active_collisions[env_agent_ixgrid], axis=(-2, -1))

        self.active_collisions = active_collisions

    def _set_game_events_from_state(self):
        for team in self.game_events:
            self.game_events[team]['captures'] = self.state['captures'][:, int(team)]
            self.game_events[team]['tags'] = self.state['tags'][:, int(team)]
            self.game_events[team]['grabs'] = self.state['grabs'][:, int(team)]
            self.game_events[team]['collisions'] = np.sum(self.state['agent_collisions'][:, self.agent_inds_of_team[team]], axis=-1)


class PyQuaticusEnv(PyQuaticusEnvBase):
    """
    ### Description.
    This environment simulates a game of capture the flag with agent dynamics based on MOOS-IvP
    (https://oceanai.mit.edu/ivpman/pmwiki/pmwiki.php?n=IvPTools.USimMarine#section5).

    ### Arguments
    n_envs: number of vectorized environments to run

    team_size: number of agents per team

    action_repeat: number of times the step function is called before game-state checks are run
    
    action_space: type of action space for each agent ('discrete', 'continuous', or 'afp')
        (1) 'discrete': discrete action space with all combinations of max speed, half speed; and 45 degree relative heading intervals
        (2) 'continuous': continuous action space for speed [0, max speed] and desired relative heading [0, 359]
        (3) 'afp': discrete action space of target positions from AQUATICUS_FIELD_POINTS (see config.py)

        Note 1: If different action spaces are desired for different agents, provide a list / tuple / array of length 2*team_size like:
                ['discrete', 'discrete', 'continuous', 'afp']
                Each action space type will be applied to the agent at the corresponding index in self.agents.

        Note 2: All agents can take each type of action input to the step function regardless of the type action space specified.
                This parameter is just used to set the PettingZoo standard action_spaces attribute.

        Note 3: Inputs to the step function for the 'afp' action space will be strings from AQUATICUS_FIELD_POINTS (see config.py)
    
    reward_config: a dictionary configuring the reward structure (see rewards.py)
    
    config_dict: a dictionary configuring the environment (see config_dict_std above)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pyquaticus_v0"
    }

    def __init__(
        self,
        n_envs: int = 1,
        team_size: int = 1,
        action_repeat: int = 1,
        action_space: Union[str, list[str], dict[str, str]] = "discrete",
        reward_config: dict = None,
        config_dict = config_dict_std,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.n_envs = n_envs
        self.team_size = team_size
        self.action_repeat = action_repeat
        self.num_blue = team_size
        self.num_red = team_size
        self.reward_config = {} if reward_config is None else reward_config
        self.config_dict = config_dict
        self.render_mode = render_mode

        self.reset_count = np.zeros(n_envs, dtype=int)
        self.step_count = np.zeros(n_envs, dtype=int)
        self.current_time = np.zeros(n_envs)
        self.state = None
        self.prev_state = None
        self.dist_bearing_to_obstacles = None
        self.dones = {
            "blue": np.zeros(n_envs, dtype=bool),
            "red": np.zeros(n_envs, dtype=bool),
            "__all__": np.zeros(n_envs, dtype=bool)
        }
        self.aquaticus_field_points = None
        self.afp_sym = True
        self.active_collisions = np.zeros((n_envs, 2*team_size, 2*team_size), dtype=bool) #current collisions between all agents
        self.game_events = {
            team: {
                "captures": np.zeros(n_envs, dtype=int),
                "grabs": np.zeros(n_envs, dtype=int),
                "tags": np.zeros(n_envs, dtype=int),
                "collisions": np.zeros(n_envs, dtype=int),
            }
            for team in Team
        }
        self.cli_message = ""

        # Set variables from config
        self.set_config_values(config_dict)

        # Create players
        b_players = []
        r_players = []
        for i in range(0, self.num_blue):
            b_players.append(
                dynamics_registry[self.dynamics[i]](
                    gps_env=self.gps_env,
                    meters_per_mercator_xy=getattr(self, "meters_per_mercator_xy", None),
                    dt=self.dt,
                    id=f'agent_{i}',
                    idx=i,
                    team=Team.BLUE_TEAM,
                    render_radius=getattr(self, "agent_render_radius", [None] * 2*team_size)[i],
                    render_mode=render_mode,
                )
            )
        for i in range(self.num_blue, self.num_blue + self.num_red):
            r_players.append(
                dynamics_registry[self.dynamics[i]](
                    gps_env=self.gps_env,
                    meters_per_mercator_xy=getattr(self, "meters_per_mercator_xy", None),
                    dt=self.dt,
                    id=f'agent_{i}',
                    idx=i,
                    team=Team.RED_TEAM,
                    render_radius=getattr(self, "agent_render_radius", [None] * 2*team_size)[i],
                    render_mode=render_mode,
                )
            )

        self.players: dict[str, Dynamics] = {player.id: player for player in itertools.chain(b_players, r_players)}  #maps player ids (or names) to player objects
        self.agents = [agent_id for agent_id in self.players] #maps agent indices to ids
        self.possible_agents = [agent_id for agent_id in self.players]
        self.max_speeds = [player.get_max_speed() for player in self.players.values()]

        # Agents (player objects) of each team
        self.agents_of_team = {Team.BLUE_TEAM: b_players, Team.RED_TEAM: r_players}
        self.agent_ids_of_team = {team: [player.id for player in self.agents_of_team[team]] for team in Team}
        self.agent_inds_of_team = {team: np.array([player.idx for player in self.agents_of_team[team]]) for team in Team}

        # Create the list of flags that are indexed by self.flags[int(player.team)]
        if len(self.flag_homes) != len(Team):
            raise Exception(f"Length of self.flag_homes ({len(self.flag_homes)}) is not equal to that of {Team} struct ({len(Team)}).")

        self.flags = [None for _ in range(len(self.flag_homes))]
        for team in Team:
            flag = Flag(team)
            flag.home = np.array(self.flag_homes[team])
            self.flags[int(team)] = flag

        # Team wall orientation
        self._determine_team_wall_orient()

        # Obstacles and Lidar
        self.set_geom_config(config_dict)

        # Setup action and observation spaces
        self.discrete_action_map = [[spd, hdg] for (spd, hdg) in ACTION_MAP]
        self.act_space_str = self.multiagent_var(action_space, type(action_space), "action_space")

        if not isinstance(self.act_space_str, dict):
            self.act_space_str = {agent_id: self.act_space_str[i] for i, agent_id in enumerate(self.players)}

        self.action_spaces = {agent_id: self.get_agent_action_space(self.act_space_str[agent_id], i) for i, agent_id in enumerate(self.players)}
        self.act_space_checked = {agent_id: False for agent_id in self.players}
        self.act_space_match = {agent_id: True for agent_id in self.players}

        self.agent_obs_normalizer, self.global_state_normalizer = self._register_state_elements(team_size, len(self.obstacles), n_envs)
        self.observation_spaces = {agent_id: self.get_agent_observation_space() for agent_id in self.players}

        # Set up rewards
        for agent_id in self.players:
            if agent_id not in self.reward_config:
                self.reward_config[agent_id] = None

        # Pygame
        self.screen = None
        self.clock = None
        self.render_ctr = np.zeros(n_envs, dtype=int)
        self.render_buffer = [[] for _ in range(n_envs)]
        self.traj_render_buffer = {
            agent_id: {
                "traj": [[] for _ in range(n_envs)],
                "agent": [[] for _ in range(n_envs)],
                "history": [[] for _ in range(n_envs)]
            } for agent_id in self.players
        }

        if self.render_mode:
            self.create_background_image()  # create background pygame surface (for faster rendering)

        # RRT policies for driving back home
        self.rrt_policies = []
        if len(self.obstacles) > 0:
            global EnvWaypointPolicy
            from pyquaticus.base_policies.env_waypoint_policy import EnvWaypointPolicy
            for i in range(self.n_envs):
                self.rrt_policies.append([])
                for j in range(self.num_agents):
                    self.rrt_policies[i].append(
                        EnvWaypointPolicy(
                            self.obstacles,
                            self.env_size,
                            self.max_speeds[j],
                            capture_radius=0.45*self.catch_radius,
                            slip_radius=self.slip_radius[j],
                            avoid_radius=2*self.agent_radius[j]
                        )
                    )

    def step(self, raw_action_dict, env_idxs):
        """
        Steps the environment forward in time by self.dt seconds, applying actions.

        Args:
            raw_action_dict: Actions from discrete or continuous action space for agents to apply

        Returns
        -------
            New observation after stepping

            Some reward

            Indicators for whether or not the env is done/truncated

            Additional info:
                -global state
                -unnormalized observations
        """
        if self.state is None:
            raise Exception("Call reset before using step method.")

        if not set(raw_action_dict.keys()) <= set(self.players):
            raise ValueError(
                "Keys of action dict should be player ids but got"
                f" {raw_action_dict.keys()}"
            )

        # Previous state
        self.prev_state = copy.deepcopy(self.state)

        # Tagging cooldown
        for i, player in enumerate(self.players.values()):
            if player.tagging_cooldown != self.tagging_cooldown:
                # player is still under a cooldown from tagging, advance their cooldown timer, clip at the configured tagging cooldown
                player.tagging_cooldown = self._min(
                    (player.tagging_cooldown + self.sim_speedup_factor * self.dt),
                    self.tagging_cooldown,
                )
                self.state["agent_tagging_cooldown"][i] = player.tagging_cooldown
        
        # Process incoming actions
        action_dict = {}
        for player in self.players.values():
            if player.id in raw_action_dict:
                if not self.act_space_checked[player.id]:
                    if isinstance(raw_action_dict[player.id], str):
                        self.act_space_match[player.id] = self.action_spaces[player.id].contains(
                            np.asarray(raw_action_dict[player.id])
                        )
                    else:
                        self.act_space_match[player.id] = self.action_spaces[player.id].contains(
                            np.asarray(raw_action_dict[player.id], dtype=self.action_spaces[player.id].dtype)
                        )
                    self.act_space_checked[player.id] = True

                    if not self.act_space_match[player.id]:
                        action_print = repr(raw_action_dict[player.id]) if isinstance(raw_action_dict[player.id], str) else raw_action_dict[player.id]
                        print(f"Warning! Action passed in for {player.id} ({action_print}) is not contained in agent's action space ({self.action_spaces[player.id]}).")
                        print(f"Auto-detecting action space for {player.id}")
                        print()

                speed, rel_heading = self._to_speed_heading(
                    raw_action=raw_action_dict[player.id],
                    player=player,
                    act_space_match=self.act_space_match[player.id],
                    act_space_str=self.act_space_str[player.id]
                )
            else:
                #if no action provided, no-op
                speed, rel_heading = 0.0, 0.0

            action_dict[player.id] = np.array([speed, rel_heading], dtype=np.float32)

        # Move agents and render
        for _ in range(self.sim_speedup_factor):
            self._move_agents(action_dict)
        if self.lidar_obs:
            self._update_lidar()

        # Set the time
        self.current_time += self.sim_speedup_factor * self.dt

        # Agent and flag checks and more
        self._check_oob(env_idxs)
        self._check_untag_and_flag_keepout(env_idxs)
        self._check_agent_made_tag(env_idxs)
        self._check_flag_pickups(env_idxs)
        self._check_flag_captures(env_idxs)
        cli_message = self._set_dones(env_idxs)
        self._update_dist_bearing_to_obstacles(env_idxs)
        self._check_agent_collisions(env_idxs)

        if self.lidar_obs:
            raise NotImplementedError("Vector environment with Lidar not implemented.")
            # for team in self.agents_of_team:
            #     for agent_id, player in self.players.items():
            #         if player.team == team:
            #             detection_class_name = "teammate"
            #         else:
            #             detection_class_name = "opponent"
            #         if player.is_tagged:
            #             detection_class_name += "_is_tagged"
            #         elif player.has_flag:
            #             detection_class_name += "_has_flag"

            #         self.obj_ray_detection_states[team][self.ray_int_label_map[agent_id]] = LIDAR_DETECTION_CLASS_MAP[detection_class_name]

        # Message
        if cli_message and self.render_mode == 'human':
            print(cli_message)

        # Observations
        for agent_id in raw_action_dict:
            next_obs, next_unnorm_obs = self.state_to_obs(agent_id, self.normalize_obs)

            self.state["obs_hist_buffer"][agent_id][1:] = self.state["obs_hist_buffer"][agent_id][:-1]
            self.state["obs_hist_buffer"][agent_id][0] = next_obs

            if self.normalize_obs:
                self.state["unnorm_obs_hist_buffer"][agent_id][1:] = self.state["unnorm_obs_hist_buffer"][agent_id][:-1]
                self.state["unnorm_obs_hist_buffer"][agent_id][0] = next_unnorm_obs

        obs = {agent_id: self._history_to_obs(agent_id, "obs_hist_buffer") for agent_id in self.players}

        # Global State
        self.state["global_state_hist_buffer"][1:] = self.state["global_state_hist_buffer"][:-1]
        self.state["global_state_hist_buffer"][0] = self.state_to_global_state(self.normalize_state, env_idxs)
        global_state = self._history_to_state() #common to all agents

        # Rewards
        rewards = {agent_id: self.compute_rewards(agent_id, player.team) for agent_id, player in self.players.items()}

        # Dones
        terminated = False
        truncated = False

        if self.dones["__all__"]:
            if self.dones["blue"] or self.dones["red"]:
                terminated = True
            else:
                truncated = True

        terminated = {agent_id: terminated for agent_id in self.players}
        truncated = {agent_id: truncated for agent_id in self.players}
        terminated["__all__"] = self.dones["__all__"]
        truncated["__all__"] = self.dones["__all__"]

        # Info
        info = {agent_id: {} for agent_id in self.players}
        for agent_id in self.agents:
            #global state
            info[agent_id]["global_state"] = global_state

            #unnormalized obs
            if self.normalize_obs:
                info[agent_id]["unnorm_obs"] = self._history_to_obs(agent_id, "unnorm_obs_hist_buffer")

            #low-level action
            info[agent_id]["low_level_action"] = action_dict[agent_id]

        return obs, rewards, terminated, truncated, info

    def _move_agents(self, action_dict):
        """Moves agents in the space according to the specified speed/heading in `action_dict`."""
        for i, player in enumerate(self.players.values()):
            team_idx = int(player.team)
            other_team_idx = int(not team_idx)

            flag_home = self.flags[team_idx].home

            # Check if agent is on their own side
            player.on_own_side = self._check_on_sides(player.pos, player.team)
            self.state["agent_on_sides"][i] = player.on_own_side

            # If the player hits an obstacle, send back to previous position and rotate, then skip to next agent
            player_hit_obstacle = detect_collision(player.pos, self.agent_radius[i], self.obstacle_geoms)

            if player_hit_obstacle:
                if self.tag_on_collision:
                    player.is_tagged = True
                    self.state['agent_is_tagged'][i] = 1

                if player.has_flag:
                    # If they have a flag, return the flag to it's home area
                    player.has_flag = False
                    self.state['agent_has_flag'][i] = 0

                    self.flags[other_team_idx].reset()
                    self.state['flag_position'][other_team_idx] = self.flags[other_team_idx].pos
                    self.state['flag_taken'][other_team_idx] = 0

                player.rotate()
                self.state['agent_position'][i] = player.pos
                self.state['prev_agent_position'][i] = player.prev_pos
                self.state['agent_speed'][i] = player.speed
                self.state['agent_heading'][i] = player.heading
                self.state['agent_dynamics'][i] = player.state
                continue

            if len(self.obstacles) > 0:
                if len(self.rrt_policies[i].wps) > 0 and not player.is_tagged:
                    self.rrt_policies[i].wps = []

            # If agent is tagged, drive at max speed towards home
            if player.is_tagged:
                #if we are in an environment with obstacles, use RRT*
                if len(self.obstacles) > 0:
                    policy = self.rrt_policies[i]
                    assert isinstance(policy, EnvWaypointPolicy)
                    if len(policy.wps) == 0 and not policy.planning:
                        policy.plan(player.pos, self.flags[team_idx].home)
                        desired_speed = 0
                        heading_error = 0
                    else:
                        desired_speed, heading_error = policy.compute_action(player.pos, player.heading)
                        if player.oob:
                            desired_speed = min(desired_speed, player.get_max_speed() * self.oob_speed_frac) #TODO: optimize based on MOOS behvior
            
                #else go directly to home
                else:
                    _, heading_error = mag_bearing_to(player.pos, flag_home, player.heading)
                    if player.oob:
                        desired_speed = player.get_max_speed() * self.oob_speed_frac #TODO: optimize based on MOOS behvior
                    else:
                        desired_speed = player.get_max_speed()

            # If agent is out of bounds, drive back in bounds at fraction of max speed
            elif player.oob:
                heading_error = self._get_oob_recover_rel_heading(player.pos, player.heading)
                desired_speed = player.get_max_speed() * self.oob_speed_frac

            # Else get desired speed and heading from action_dict
            else:
                desired_speed, heading_error = action_dict[player.id]

            # Move agent
            player._move_agent(desired_speed, heading_error)

            # Check if agent is in keepout region for their own flag
            ag_dis_2_flag = self.get_distance_between_2_points(player.pos, np.asarray(flag_home))
            if (
                ag_dis_2_flag < self.flag_keepout_radius + self.agent_radius[i]
                and not self.flags[team_idx].taken
                and self.flag_keepout_radius > 0.
            ):
                ag_pos = rc_intersection(
                    np.array([player.pos, player.prev_pos]),
                    np.asarray(flag_home),
                    self.flag_keepout_radius + self.agent_radius[i],
                )  # point where agent center first intersected with keepout zone
                vel = mag_heading_to_vec(player.speed, player.heading)

                ag_vel = reflect_vector(ag_pos, vel, np.asarray(flag_home))

                crd_ref_angle = get_rot_angle(np.asarray(flag_home), ag_pos)
                vel_ref = rot2d(ag_vel, -crd_ref_angle)
                vel_ref[1] = 0.0  # convention is that vector pointing from keepout intersection to flag center is y' axis in new reference frame

                vel = rot2d(vel_ref, crd_ref_angle)
                player.pos = ag_pos
                speed, heading = vec_to_mag_heading(vel)
                player.speed = speed
                player.heading = heading

            # Move flag (if necessary)
            if player.has_flag:
                self.flags[other_team_idx].pos = player.pos
                self.state['flag_position'][other_team_idx] = np.array(self.flags[other_team_idx].pos)

            # Update environment state
            self.state['agent_position'][i] = player.pos
            self.state['prev_agent_position'][i] = player.prev_pos
            self.state['agent_speed'][i] = player.speed
            self.state['agent_heading'][i] = player.heading
            self.state['agent_dynamics'][i] = player.state

    def _get_oob_recover_rel_heading(self, pos, heading):
        #compute the closest env edge and steer towards heading perpendicular to edge
        closest_env_edge_idx = closest_line(pos, self.env_edges)
        edge_vec = np.diff(self.env_edges[closest_env_edge_idx], axis=0)[0]
        desired_vec = np.array([-edge_vec[1], edge_vec[0]]) #this points inwards because edges are defined ccw
        _, desired_heading = vec_to_mag_heading(desired_vec)
        
        heading_error = angle180((desired_heading - heading) % 360)

        return heading_error

    def _update_lidar(self):
        raise NotImplementedError("Vector environment with Lidar not implemented.")
        # ray_int_segments = np.copy(self.ray_int_segments)

        # # Valid flag intersection segments mask
        # flag_int_seg_mask = np.ones(len(self.ray_int_seg_labels), dtype=bool)
        # flag_seg_inds = self.seg_label_type_to_inds["flag"]
        # flag_int_seg_mask[flag_seg_inds] = np.repeat(np.logical_not(self.state["flag_taken"]), self.n_circle_segments)

        # # Translate non-static ray intersection geometries (flags and agents)
        # ray_int_segments[flag_seg_inds] += np.repeat(
        #     np.tile(self.state["flag_home"], 2),
        #     self.n_circle_segments,
        #     axis=0
        # )
        # agent_seg_inds = self.seg_label_type_to_inds["agent"]
        # ray_int_segments[agent_seg_inds] += np.repeat(
        #     np.tile(self.state["agent_position"], 2),
        #     self.n_circle_segments,
        #     axis=0
        # )
        # ray_int_segments = ray_int_segments.reshape(1, -1, 4)

        # # Agent rays
        # ray_origins = np.expand_dims(self.state["agent_position"], axis=1)
        # ray_headings_global = np.deg2rad(
        #     (heading_angle_conversion(self.state["agent_heading"]).reshape(-1, 1) + self.lidar_ray_headings) % 360
        # )
        # ray_vecs = np.array([np.cos(ray_headings_global), np.sin(ray_headings_global)]).transpose(1, 2, 0)
        # ray_ends = ray_origins + self.lidar_range * ray_vecs
        # ray_segments = np.concatenate(
        #     (np.full(ray_ends.shape, ray_origins), ray_ends),
        #     axis=-1
        # )
        # ray_segments = ray_segments.reshape(self.num_agents, -1, 1, 4)

        # # compute ray intersections
        # x1, y1, x2, y2 = (
        #     ray_segments[..., 0],
        #     ray_segments[..., 1],
        #     ray_segments[..., 2],
        #     ray_segments[..., 3],
        # )
        # x3, y3, x4, y4 = (
        #     ray_int_segments[..., 0],
        #     ray_int_segments[..., 1],
        #     ray_int_segments[..., 2],
        #     ray_int_segments[..., 3],
        # )

        # denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        # intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        # intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4) ) / denom

        # # mask invalid intersections (parallel lines, outside of segment bounds, picked up flags, own agent segments)
        # mask = (denom != 0) & \
        #     (intersect_x >= np.minimum(x1, x2) - LINE_INTERSECT_TOL) & (intersect_x <= np.maximum(x1, x2) + LINE_INTERSECT_TOL) & \
        #     (intersect_y >= np.minimum(y1, y2) - LINE_INTERSECT_TOL) & (intersect_y <= np.maximum(y1, y2) + LINE_INTERSECT_TOL) & \
        #     (intersect_x >= np.minimum(x3, x4) - LINE_INTERSECT_TOL) & (intersect_x <= np.maximum(x3, x4) + LINE_INTERSECT_TOL) & \
        #     (intersect_y >= np.minimum(y3, y4) - LINE_INTERSECT_TOL) & (intersect_y <= np.maximum(y3, y4) + LINE_INTERSECT_TOL) & \
        #     flag_int_seg_mask & self.agent_int_seg_mask

        # intersect_x = np.where(mask, intersect_x, -self.env_diag)  #a coordinate out of bounds and far away
        # intersect_y = np.where(mask, intersect_y, -self.env_diag)  #a coordinate out of bounds and far away
        # intersections = np.stack((intersect_x.flatten(), intersect_y.flatten()), axis=-1).reshape(intersect_x.shape + (2,))

        # # determine lidar ray readings
        # ray_origins = np.expand_dims(ray_origins, axis=1)
        # intersection_dists = np.linalg.norm(intersections - ray_origins, axis=-1)
        # ray_int_inds = np.argmin(intersection_dists, axis=-1)

        # ray_int_labels = self.ray_int_seg_labels[ray_int_inds]
        # ray_intersections = intersections[np.arange(self.num_agents).reshape(-1, 1), np.arange(self.num_lidar_rays), ray_int_inds]
        # ray_int_dists = intersection_dists[np.arange(self.num_agents).reshape(-1, 1), np.arange(self.num_lidar_rays), ray_int_inds]

        # # correct lidar ray readings for which nothing was detected
        # invalid_ray_ints = np.where(np.all(np.logical_not(mask), axis=-1))
        # ray_int_labels[invalid_ray_ints] = self.ray_int_label_map["nothing"]
        # ray_intersections[invalid_ray_ints] = ray_ends[invalid_ray_ints]
        # ray_int_dists[invalid_ray_ints] = self.lidar_range

        # # save lidar readings
        # for i, agent_id in enumerate(self.players):
        #     self.state["lidar_labels"][agent_id] = ray_int_labels[i]
        #     self.state["lidar_ends"][agent_id] = ray_intersections[i]
        #     self.state["lidar_distances"][agent_id] = ray_int_dists[i]

    def _check_oob(self, env_idxs):
        """Checks if players are out of bounds and updates their states (and any flags in their possesion) accordingly."""
        prev_agent_oob = self.state['agent_oob'][env_idxs]
        
        # Set out-of-bounds
        agent_poses = self.state['agent_position'][env_idxs]
        agent_oob = np.any((agent_poses <= 0) | (self.env_size <= agent_poses), axis=-1)
        self.state['agent_oob'][env_idxs] = agent_oob

        new_oob = agent_oob & ~prev_agent_oob
        if not np.any(new_oob):
            return

        agent_oob_idxs = np.where(new_oob)
        agent_oob_idxs[0] = env_idxs[agent_oob_idxs[0]] 

        # Set tag (if applicable)
        if self.tag_on_oob:
            self.state['agent_is_tagged'][agent_oob_idxs] = True

        # Reset picked-up flag (if applicable)
        agent_has_flag_oob = self.state['agent_has_flag'][env_idxs] & new_oob
        if np.any(agent_has_flag_oob):
            # update agent
            self.state['agent_has_flag'][agent_oob_idxs] = False
            # update flag
            for team, agent_idxs in self.agent_inds_of_team.items():
                team_has_flag_oob = np.any(agent_has_flag_oob[:, agent_idxs], axis=-1)
                if np.any(team_has_flag_oob):
                    #note: assumes two teams, and one flag per team
                    other_team_idx = int(not int(team))

                    flag_oob_env_idxs = env_idxs[np.where(team_has_flag_oob)[0]]
                    self.flag[other_team_idx].reset(flag_oob_env_idxs)
                    self.state['flag_position'][flag_oob_env_idxs, other_team_idx] = self.flags[other_team_idx].pos[flag_oob_env_idxs]
                    self.state['flag_taken'][flag_oob_env_idxs, other_team_idx] = False

    def _check_untag_and_flag_keepout(self, env_idxs):
        """
        Untags the player if they return to their own flag.
        Updates "agent_in_flag_keepout in state.
        """
        for team, agent_idxs in self.agent_inds_of_team.items():
            env_agent_ixgrid = np.ix_(env_idxs, agent_idxs)
            agent_poses = self.state['agent_position'][env_agent_ixgrid]
            flag_home = self.flags[int(team)].home

            flag_distances = np.linalg.norm(flag_home - agent_poses, axis=-1)
            self.state['agent_is_tagged'][env_agent_ixgrid] &= flag_distances >= self.catch_radius
            self.state['agent_in_flag_keepout'][env_agent_ixgrid] = flag_distances <= self.flag_keepout_radius

    def _check_agent_made_tag(self, env_idxs):
        """
        Updates player states if they tagged another player.
        Note 1: assumes one tag allowed per tagging cooldown recharge.
        Note 2: assumes two teams, and one flag per team.
        """
        self.state["agent_made_tag"][env_idxs] = -1

        # Check tagging cooldown, on-sides, tagged status, and out-of-bounds status
        cantag = (
            (self.state["agent_tagging_cooldown"][env_idxs] == self.tagging_cooldown) &
            self.state["agent_on_sides"][env_idxs] &
            ~self.state["agent_is_tagged"][env_idxs] & 
            ~self.state["agent_oob"][env_idxs]
        )
        if not np.any(cantag):
            return

        # Check on-sides, and tagged status
        taggable = ~self.state["agent_on_sides"][env_idxs] & ~self.state["agent_is_tagged"][env_idxs]
        if not np.any(taggable):
            return

        # Update tags (if any)
        for team, agent_idxs in self.agent_inds_of_team.items():
            team_idx = int(team)
            other_team_idx = int(not team_idx)
            other_agent_idxs = np.setdiff1d(np.arange(self.num_agents), agent_idxs)

            env_agent_ixgrid = np.ix_(env_idxs, agent_idxs)
            env_other_agent_ixgrid = np.ix_(env_idxs, other_agent_idxs)

            a2a_dists = np.linalg.norm(
                self.state["agent_position"][env_agent_ixgrid][:, :, None, :] - self.state["agent_position"][env_other_agent_ixgrid][:, None, :, :],
                axis=-1
            )
            a2a_dists_in_range = a2a_dists < self.catch_radius
            if not np.any(a2a_dists_in_range):
                continue

            tag_matrix = (cantag[:, agent_idxs][:, :, None] & taggable[:, other_agent_idxs][:, None, :]) & a2a_dists_in_range
            for i, j in enumerate(agent_idxs):
                made_tag = np.any(tag_matrix[:, i], axis=-1)
                if np.any(made_tag):
                    made_tag_idxs = np.where(made_tag)[0]
                    target = np.argmin(np.where(tag_matrix[made_tag_idxs, i], a2a_dists_in_range[made_tag_idxs, i], np.inf), axis=-1)

                    tag_matrix[made_tag_idxs, i+1:, target] = False
                    
                    #update tagger agent
                    made_tag_env_idxs = env_idxs[made_tag_idxs]
                    target_idx = other_agent_idxs[target]

                    self.state['agent_made_tag'][made_tag_env_idxs, j] = target_idx
                    self.state['agent_tagging_cooldown'][made_tag_env_idxs, j] = 0.0
                    self.state['tags'][made_tag_env_idxs, team_idx] += 1
                    self.game_events[team]['tags'][made_tag_env_idxs] += 1

                    #update target agent
                    self.state['agent_is_tagged'][made_tag_env_idxs, target_idx] = True

                    #update flag and has_flag
                    target_has_flag = self.state['agent_has_flag'][made_tag_env_idxs, target_idx]
                    if np.any(target_has_flag):
                        target_has_flag_env_idxs = made_tag_env_idxs[np.where(target_has_flag)[0]]

                        self.flags[other_team_idx].reset(target_has_flag_env_idxs)
                        self.state['flag_position'][target_has_flag_env_idxs, other_team_idx] = self.flags[other_team_idx].pos[target_has_flag_env_idxs]
                        self.state['flag_taken'][target_has_flag_env_idxs, other_team_idx] = False
                        self.state['agent_has_flag'][made_tag_env_idxs, target_idx] = False

    def _check_flag_pickups(self, env_idxs):
        """
        Updates player states if they picked up the flag.
        Note: assumes two teams, and one flag per team.
        """
        for team, agent_idxs in self.agent_idxs_of_team.items():
            team_idx = int(team)
            other_team_idx = int(not team_idx)

            other_team_flag_not_taken = ~self.state['flag_taken'][env_idxs, other_team_idx]
            if np.any(other_team_flag_not_taken):
                flag_not_taken_env_idxs = env_idxs[np.where(other_team_flag_not_taken)[0]]
                env_agent_ixgrid = np.ix_(flag_not_taken_env_idxs, agent_idxs)

                flag_distances = np.linalg.norm(
                 j   self.flags[other_team_idx].home - self.state['agent_position'][env_agent_ixgrid],
                    axis=-1
                )
                agent_on_sides = self.state['agent_on_sides'][env_agent_ixgrid]
                agent_oob = self.state['agent_oob'][env_agent_ixgrid]
                agent_has_flag = self.state['agent_has_flag'][env_agent_ixgrid]
                agent_is_tagged = self.state['agent_is_tagged'][env_agent_ixgrid]

                agent_flag_pickups = (
                    (flag_distances < self.catch_radius) & ~agent_on_sides &
                    ~agent_oob & ~agent_has_flag & ~agent_is_tagged
                )
                if np.any(agent_flag_pickups):
                    #NOTE: we choose the agent for the flag pickup based on index priority (not distance)
                    flag_pickup_idxs = np.where(np.any(agent_flag_pickups, axis=-1))[0]
                    flag_pickup_env_idxs = flag_not_taken_env_idxs[flag_pickup_idxs]
                    flag_pickup_agent_idxs = agent_idxs[np.argmax(agent_flag_pickups[flag_pickup_idxs], axis=-1)]

                 j   # update agent
                    self.state['agent_has_flag'][flag_pickup_env_idxs, flag_pickup_agent_idxs] = True

                    # update flags
                    self.flags[other_team_idx].pos[flag_pickup_env_idxs] = np.array(
                        self.state['agent_position'][flag_pickup_env_idxs, flag_pickup_agent_idxs]
                    )
                    self.state['flag_position'][flag_pickup_env_idxs, other_team_idx] = self.flags[other_team_idx].pos[flag_pickup_env_idxs]
                    self.state['flag_taken'][flag_pickup_env_idxs, other_team_idx] = True

                    # update grabs
                    self.state['grabs'][flag_pickup_env_idxs, team_idx] += 1
                    self.game_events[team]['grabs'][flag_pickup_env_idxs] += 1

    def _check_flag_captures(self, env_idxs):
        """
        Updates states if a player captured a flag.
        Note: assumes two teams, and one flag per team.
        """
        for team, agent_idxs in self.agent_idxs_of_team.items():
            team_idx = int(team)
            other_team_idx = int(not team_idx)
            env_agent_ixgrid = np.ix_(env_idxs, agent_idxs)

            captures = self.state['agent_on_sides'][env_agent_ixgrid] & self.state['agent_has_flag'][env_agent_ixgrid]
            if np.any(captures):
                captures_idxs = np.where(np.any(captures, axis=-1))[0]
                captures_env_idxs = env_idxs[captures_idxs]
                captures_agent_idxs = agent_idxs[np.argmax(captures[captures_idxs], axis=-1)]

                # Update agent
                self.state['agent_has_flag'][captures_env_idxs, captures_agent_idxs] = False

                # Update flag
                self.flags[other_team_idx].reset(captures_env_idxs)
                self.state['flag_position'][captures_env_idxs, other_team_idx] = self.flags[other_team_idx].pos[captures_env_idxs]
                self.state['flag_taken'][captures_env_idxs, other_team_idx] = False

                # Update captures
                new_team_captures = np.sum(captures[captures_idxs], axis=-1)
                self.state['captures'][captures_env_idxs, team_idx] += new_team_captures
                self.game_events[player.team]['captures'][captures_env_idxs] += new_team_captures

    def set_config_values(self, config_dict):
        """
        Sets initial configuration parameters for the environment.

        Args:
            config_dict: The provided configuration. If a key is missing, it is replaced
            with the standard configuration value.
        """
        ### Set Variables from the Configuration Dictionary ###
        # Check for unrecognized variables
        for k in config_dict:
            if k not in config_dict_std:
                print(f"Warning! Config variable '{k}' not recognized (it will have no effect).")
                print("Please consult config.py for variable names.")
                print()

        # Geometry parameters
        self.gps_env = config_dict.get("gps_env", config_dict_std["gps_env"])
        self.topo_contour_eps = config_dict.get("topo_contour_eps", config_dict_std["topo_contour_eps"])

        # Dynamics parameters
        self.dynamics = config_dict.get("dynamics", config_dict_std["dynamics"])

        if isinstance(self.dynamics, (list, tuple, np.ndarray)):
            if len(self.dynamics) != 2*self.team_size:
                raise Exception(f"Length of dynamics config list must match total number of agents ({2*self.team_size})")
        elif isinstance(self.dynamics, str):
            self.dynamics = [self.dynamics for i in range(2*self.team_size)]
        else:
            raise Exception(f"Dynamics config improperly specified. Please see config.py for instructions.")

        for dynamics in self.dynamics:
            if dynamics not in dynamics_registry:
                raise Exception(
                    f"{dynamics} is not a valid dynamics class. Please check dynamics_registry.py"
                )

        # Dynamics parameters
        self.oob_speed_frac = config_dict.get("oob_speed_frac", config_dict_std["oob_speed_frac"])

        # Simulation parameters
        self.dt = config_dict.get("tau", config_dict_std["tau"])
        self.sim_speedup_factor = config_dict.get("sim_speedup_factor", config_dict_std["sim_speedup_factor"])

        # Game parameters
        self.max_score = config_dict.get("max_score", config_dict_std["max_score"])
        self.max_time = config_dict.get("max_time", config_dict_std["max_time"])
        self.max_cycles = ceil(self.max_time / (self.sim_speedup_factor * self.dt))
        self.tagging_cooldown = config_dict.get("tagging_cooldown", config_dict_std["tagging_cooldown"])
        self.tag_on_collision = config_dict.get("tag_on_collision", config_dict_std["tag_on_collision"])
        self.tag_on_oob = config_dict.get("tag_on_oob", config_dict_std["tag_on_oob"])

        # Observation and state parameters
        self.normalize_obs = config_dict.get("normalize_obs", config_dict_std["normalize_obs"])
        self.short_obs_hist_length = config_dict.get("short_obs_hist_length", config_dict_std["short_obs_hist_length"])
        self.short_obs_hist_interval = config_dict.get("short_obs_hist_interval", config_dict_std["short_obs_hist_interval"])
        self.long_obs_hist_length = config_dict.get("long_obs_hist_length", config_dict_std["long_obs_hist_length"])
        self.long_obs_hist_interval = config_dict.get("long_obs_hist_interval", config_dict_std["long_obs_hist_interval"])

        # Lidar-specific observation parameters
        self.lidar_obs = config_dict.get("lidar_obs", config_dict_std["lidar_obs"])
        self.num_lidar_rays = config_dict.get("num_lidar_rays", config_dict_std["num_lidar_rays"])

        # Global state parameters
        self.normalize_state = config_dict.get("normalize_state", config_dict_std["normalize_state"])
        self.short_state_hist_length = config_dict.get("short_state_hist_length", config_dict_std["short_state_hist_length"])
        self.short_state_hist_interval = config_dict.get("short_state_hist_interval", config_dict_std["short_state_hist_interval"])
        self.long_state_hist_length = config_dict.get("long_state_hist_length", config_dict_std["long_state_hist_length"])
        self.long_state_hist_interval = config_dict.get("long_state_hist_interval", config_dict_std["long_state_hist_interval"])

        # Rendering parameters
        self.screen_frac = config_dict.get("screen_frac", config_dict_std["screen_frac"])
        self.arena_buffer_frac = config_dict.get("arena_buffer_frac", config_dict_std["arena_buffer_frac"])
        self.render_ids = config_dict.get("render_agent_ids", config_dict_std["render_agent_ids"])
        self.render_field_points = config_dict.get("render_field_points", config_dict_std["render_field_points"])
        self.render_traj_mode = config_dict.get("render_traj_mode", config_dict_std["render_traj_mode"])
        self.render_traj_freq = config_dict.get("render_traj_freq", config_dict_std["render_traj_freq"])
        self.render_traj_cutoff = config_dict.get("render_traj_cutoff", config_dict_std["render_traj_cutoff"])
        self.render_lidar_mode = config_dict.get("render_lidar_mode", config_dict_std["render_lidar_mode"])
        self.render_saving = config_dict.get("render_saving", config_dict_std["render_saving"])
        self.render_transparency_alpha = config_dict.get("render_transparency_alpha", config_dict_std["render_transparency_alpha"])

        # Agent spawn parameters
        self.on_sides_init = config_dict.get("on_sides_init", config_dict_std["on_sides_init"])

        # Miscellaneous parameters
        if config_dict.get("suppress_numpy_warnings", config_dict_std["suppress_numpy_warnings"]):
            # Suppress numpy warnings to avoid printing out extra stuff to the console
            np.seterr(all="ignore")

        ### Environment History ###
        # Observations
        short_obs_hist_buffer_inds = np.arange(0, self.short_obs_hist_length * self.short_obs_hist_interval, self.short_obs_hist_interval)
        long_obs_hist_buffer_inds = np.arange(0, self.long_obs_hist_length * self.long_obs_hist_interval, self.long_obs_hist_interval)
        self.obs_hist_buffer_inds = np.unique(
            np.concatenate((short_obs_hist_buffer_inds, long_obs_hist_buffer_inds))
        )  # indices of history buffer corresponding to history entries

        self.obs_hist_len = len(self.obs_hist_buffer_inds)
        self.obs_hist_buffer_len = self.obs_hist_buffer_inds[-1] + 1

        short_obs_hist_oldest_timestep = self.short_obs_hist_length * self.short_obs_hist_interval - self.short_obs_hist_interval
        long_obs_hist_oldest_timestep = self.long_obs_hist_length * self.long_obs_hist_interval - self.long_obs_hist_interval
        if (
            short_obs_hist_oldest_timestep > long_obs_hist_oldest_timestep
            and self.long_obs_hist_length != 1
        ):
            print(f"Warning! The short term obs history contains older timestep (-{short_obs_hist_oldest_timestep}) than the long term obs history (-{long_obs_hist_oldest_timestep}).")
        
        # Global State
        short_state_hist_buffer_inds = np.arange(0, self.short_state_hist_length * self.short_state_hist_interval, self.short_state_hist_interval)
        long_state_hist_buffer_inds = np.arange(0, self.long_state_hist_length * self.long_state_hist_interval, self.long_state_hist_interval)
        self.state_hist_buffer_inds = np.unique(
            np.concatenate((short_state_hist_buffer_inds, long_state_hist_buffer_inds))
        )  # indices of history buffer corresponding to history entries

        self.state_hist_len = len(self.state_hist_buffer_inds)
        self.state_hist_buffer_len = self.state_hist_buffer_inds[-1] + 1

        short_state_hist_oldest_timestep = self.short_state_hist_length * self.short_state_hist_interval - self.short_state_hist_interval
        long_state_hist_oldest_timestep = self.long_state_hist_length * self.long_state_hist_interval - self.long_state_hist_interval
        if (
            short_state_hist_oldest_timestep > long_state_hist_oldest_timestep
            and self.long_state_hist_length != 1

        ):
            print(f"Warning! The short term state history contains older timestep (-{short_state_hist_oldest_timestep}) than the long term state history (-{long_state_hist_oldest_timestep}).")

        ### Environment Geometry Construction ###
        # Basic environment features
        env_bounds = config_dict.get("env_bounds", config_dict_std["env_bounds"])
        env_bounds_unit = config_dict.get("env_bounds_unit", config_dict_std["env_bounds_unit"])

        flag_homes = {
            Team.BLUE_TEAM: config_dict.get("blue_flag_home", config_dict_std["blue_flag_home"]),
            Team.RED_TEAM: config_dict.get("red_flag_home", config_dict_std["red_flag_home"])
        }
        flag_homes_unit = config_dict.get("flag_homes_unit", config_dict_std["flag_homes_unit"])

        scrimmage_coords = config_dict.get("scrimmage_coords", config_dict_std["scrimmage_coords"])
        scrimmage_coords_unit = config_dict.get("scrimmage_coords_unit", config_dict_std["scrimmage_coords_unit"])

        agent_radius = config_dict.get("agent_radius", config_dict_std["agent_radius"])
        agent_radius = self.multiagent_var(agent_radius, float, "agent_radius")

        flag_radius = config_dict.get("flag_radius", config_dict_std["flag_radius"])
        flag_keepout_radius = config_dict.get("flag_keepout", config_dict_std["flag_keepout"])
        catch_radius = config_dict.get("catch_radius", config_dict_std["catch_radius"])
        if flag_keepout_radius >= (catch_radius - max(agent_radius)):
            print(f"Warning! Flag keepout radius is >= than the catch radius")

        slip_radius = config_dict.get("slip_radius", config_dict_std["catch_radius"])
        slip_radius = self.multiagent_var(slip_radius, float, "slip_radius")

        lidar_range = config_dict.get("lidar_range", config_dict_std["lidar_range"])

        self._build_base_env_geom(
            env_bounds=env_bounds,
            flag_homes=flag_homes,
            scrimmage_coords=scrimmage_coords,
            env_bounds_unit=env_bounds_unit,
            flag_homes_unit=flag_homes_unit,
            scrimmage_coords_unit=scrimmage_coords_unit,
            agent_radius=agent_radius,
            flag_radius=flag_radius,
            flag_keepout_radius=flag_keepout_radius,
            catch_radius=catch_radius,
            slip_radius=slip_radius,
            lidar_range=lidar_range
        )

        # Scale the aquaticus point field by env size 
        if not self.gps_env:
            self.aquaticus_field_points = get_afp()

            if not np.all(np.isclose(self.scrimmage_coords[:, 0], 0.5*self.env_size[0])):
                print("Warning! Aquaticus field points are not side/team agnostic when environment is not symmetric.")
                print(f"Environment dimensions: {self.env_size}")
                print(f"Scrimmage line coordinates: {self.scrimmage_coords}")
                self.afp_sym = False
            else:
                self.afp_sym = True
                #TODO: pre-compute blue/ red symmetric field points so not necessary to be done every time _afp_to_speed_relheading() is called 

            for k, v in self.aquaticus_field_points.items():
                pt = self.env_rot_matrix @ np.asarray(v)
                pt += self.env_ll
                pt *= self.env_size
                self.aquaticus_field_points[k] = pt

        ### Environment Rendering ###
        if self.render_mode:
            self.render_fps = round(1 / self.dt)

            # pygame orientation vector
            self.PYGAME_UP = Vector2((0.0, 1.0))

            # pygame screen size
            arena_buffer = np.full((2,2), self.arena_buffer_frac * np.sqrt(np.prod(self.env_size))) #matches self.env_bounds [(left, bottom), (right, top)]

            if self.gps_env:
                #clip horizontal buffers if necessary
                render_area_width = self.env_size[0] + np.sum(arena_buffer[:, 0])
                if render_area_width > 2*EPSG_3857_EXT_X:
                    arena_buffer[:, 0] = (2*EPSG_3857_EXT_X - self.env_size[0]) / 2

                #clip vertical buffers if necessary
                arena_buffer[0][1] = min(arena_buffer[0][1], np.abs(-EPSG_3857_EXT_Y - self.env_bounds[0][1]))
                arena_buffer[1][1] = min(arena_buffer[1][1], np.abs(EPSG_3857_EXT_Y - self.env_bounds[1][1]))

            max_screen_size = get_screen_res()
            arena_aspect_ratio = (self.env_size[0] + np.sum(arena_buffer[:, 0])) / (self.env_size[1] + np.sum(arena_buffer[:, 1]))
            width_based_height = max_screen_size[0] / arena_aspect_ratio

            if width_based_height <= max_screen_size[1]:
                max_pygame_screen_width = max_screen_size[0]
            else:
                height_based_width = max_screen_size[1] * arena_aspect_ratio
                max_pygame_screen_width = int(height_based_width)

            self.pixel_size = (self.screen_frac * max_pygame_screen_width) / (self.env_size[0] + np.sum(arena_buffer[:, 0]))
            self.screen_width = round((self.env_size[0] + np.sum(arena_buffer[:, 0])) * self.pixel_size)
            self.screen_height = round((self.env_size[1] + np.sum(arena_buffer[:, 1])) * self.pixel_size)

            # environemnt element sizes in pixels
            self.arena_width, self.arena_height = self.pixel_size * self.env_size
            self.arena_buffer = self.pixel_size * arena_buffer
            self.boundary_width = 2  #pixels
            self.a2a_line_width = 5  #pixels
            self.flag_render_radius = np.clip(self.pixel_size * self.flag_radius, 10, None)  #pixels
            self.agent_render_radius = np.clip(self.pixel_size * self.agent_radius, 15, None)  #pixels

            # check that time between frames (1/render_fps) is not larger than timestep (self.dt)
            frame_rate_err_msg = (
                "Frame rate ({}) creates time intervals between frames larger"
                " than specified timestep ({})".format(self.render_fps, self.dt)
            )
            assert 1 / self.render_fps <= self.dt, frame_rate_err_msg

            # check that time warp is an integer >= 1
            if self.sim_speedup_factor < 1:
                print("Warning! sim_speedup_factor must be an integer >= 1! Defaulting to 1.")
                self.sim_speedup_factor = 1

            if type(self.sim_speedup_factor) != int:
                self.sim_speedup_factor = int(np.round(self.sim_speedup_factor))
                print(f"Warning! Converted sim_speedup_factor to integer: {self.sim_speedup_factor}")

            # check that render_saving is only True if environment is being rendered
            if self.render_saving:
                assert self.render_mode is not None, "Render_mode cannot be None to record video or take screenshot."

    def set_geom_config(self, config_dict):
        self.n_circle_segments = config_dict.get("n_circle_segments", config_dict_std["n_circle_segments"])
        n_quad_segs = round(self.n_circle_segments / 4)

        # Obstacles
        obstacle_params = config_dict.get("obstacles", config_dict_std["obstacles"])

        if self.gps_env:
            border_contour, island_contours, land_mask = self._get_topo_geom()

            #border
            border_obstacles = self._border_contour_to_border_obstacles(border_contour)
            if len(border_obstacles) > 0:
                if obstacle_params is None:
                    obstacle_params = {"polygon": []}
                obstacle_params["polygon"].extend(border_obstacles)

            #islands
            if len(island_contours) > 0:
                if obstacle_params is None:
                    obstacle_params = {"polygon": []}
                obstacle_params["polygon"].extend(island_contours)

        self.obstacles = []
        self.obstacle_geoms = (
            dict()
        )  # arrays with geometric info for obstacles to be used for vectorized calculations
        if obstacle_params is not None and isinstance(obstacle_params, dict):
            #circle obstacles
            circle_obstacles = obstacle_params.get("circle", None)
            if circle_obstacles is not None and isinstance(circle_obstacles, list):
                self.obstacle_geoms["circle"] = []
                for param in circle_obstacles:
                    self.obstacles.append(CircleObstacle(param[0], (param[1][0], param[1][1])))
                    self.obstacle_geoms["circle"].append([param[0], param[1][0], param[1][1]])
                self.obstacle_geoms["circle"] = np.asarray(self.obstacle_geoms["circle"])
            elif circle_obstacles is not None:
                raise TypeError(f"Expected circle obstacle parameters to be a list of tuples, not {type(circle_obstacles)}")

            #polygon obstacle
            poly_obstacle = obstacle_params.get("polygon", None)
            if poly_obstacle is not None and isinstance(poly_obstacle, list):
                self.obstacle_geoms["polygon"] = []
                for param in poly_obstacle:
                    converted_param = [(p[0], p[1]) for p in param]
                    self.obstacles.append(PolygonObstacle(converted_param))
                    self.obstacle_geoms["polygon"].extend([(p, param[(i + 1) % len(param)]) for i, p in enumerate(param)])
                self.obstacle_geoms["polygon"] = np.asarray(self.obstacle_geoms["polygon"])
            elif poly_obstacle is not None:
                raise TypeError(f"Expected polygon obstacle parameters to be a list of tuples, not {type(poly_obstacle)}")

        elif obstacle_params is not None:
            raise TypeError(f"Expected obstacle_params to be None or a dict, not {type(obstacle_params)}")

        # Adjust scrimmage line
        if self.gps_env:
            scrim_int_segs = [(p, border_contour[(i+1) % len(border_contour)]) for i, p in enumerate(border_contour)]
        else:
            scrim_int_segs = self.env_edges

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
            ray_int_label_names.extend([f"agent_{i}" for i, _ in enumerate(self.agents)])
            self.ray_int_label_map = {label_name: i for i, label_name in enumerate(ray_int_label_names)}

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
                        agent_ind = int(label_name[6:])
                        if agent_ind in self.agent_inds_of_team[team]:
                            detection_class = LIDAR_DETECTION_CLASS_MAP["teammate"]
                        else:
                            detection_class = LIDAR_DETECTION_CLASS_MAP["opponent"]
                    else:
                        raise Exception("Unknown lidar detection class.")

                    self.obj_ray_detection_states[team].append(detection_class)
                self.obj_ray_detection_states[team] = np.asarray(self.obj_ray_detection_states[team])

            ray_int_segments = []
            ray_int_seg_labels = []
            self.seg_label_type_to_inds = {(label[: label.find("_")] if label[-1].isnumeric() else label): [] for label in ray_int_label_names}

            # boundaries
            ray_int_segments.extend(
                [[*self.env_ll, *self.env_lr],
                 [*self.env_lr, *self.env_ur],
                 [*self.env_ur, *self.env_ul],
                 [*self.env_ul, *self.env_ll]]
            )
            ray_int_seg_labels.extend(4 * [self.ray_int_label_map["obstacle"]])
            self.seg_label_type_to_inds["obstacle"].extend(np.arange(4))

            # obstacles
            obstacle_segments = [
                segment.flatten() for obstacle in self.obstacles for segment in self._generate_segments_from_obstacles(obstacle, n_quad_segs) if not self._segment_on_border(segment)
            ]
            ray_int_seg_labels.extend(len(obstacle_segments) * [self.ray_int_label_map["obstacle"]])
            self.seg_label_type_to_inds["obstacle"].extend(np.arange(len(ray_int_segments), len(ray_int_segments) + len(obstacle_segments)))
            ray_int_segments.extend(obstacle_segments)

            # flags
            for i, _ in enumerate(self.flags):
                vertices = list(Point(0.0, 0.0).buffer(self.flag_radius, quad_segs=n_quad_segs).exterior.coords)[:-1]
                segments = [[*vertex, *vertices[(j + 1) % len(vertices)]] for j, vertex in enumerate(vertices)]
                ray_int_seg_labels.extend(len(segments) * [self.ray_int_label_map[f"flag_{i}"]])
                self.seg_label_type_to_inds["flag"].extend(np.arange(len(ray_int_segments), len(ray_int_segments) + len(segments)))
                ray_int_segments.extend(segments)

            # agents
            for i, agent_id in enumerate(self.agents):
                vertices = list(Point(0.0, 0.0).buffer(self.agent_radius[i], quad_segs=n_quad_segs).exterior.coords)[:-1]
                segments = [[*vertex, *vertices[(j + 1) % len(vertices)]] for j, vertex in enumerate(vertices)]
                ray_int_seg_labels.extend(len(segments) * [self.ray_int_label_map[agent_id]])
                self.seg_label_type_to_inds["agent"].extend(np.arange(len(ray_int_segments), len(ray_int_segments) + len(segments)))
                ray_int_segments.extend(segments)

            # arrays
            self.ray_int_segments = np.array(ray_int_segments)
            self.ray_int_seg_labels = np.array(ray_int_seg_labels)

            # agent ray self intersection mask
            agent_int_seg_mask = np.ones((self.num_agents, len(self.ray_int_seg_labels)), dtype=bool)
            agent_seg_inds = self.seg_label_type_to_inds["agent"]

            for i in range(self.num_agents):
                seg_inds_start = i * self.n_circle_segments
                agent_int_seg_mask[i, agent_seg_inds[seg_inds_start : seg_inds_start + self.n_circle_segments]] = False

            self.agent_int_seg_mask = np.expand_dims(agent_int_seg_mask, axis=1)

        # Occupancy map
        if self.gps_env:
            self._generate_valid_start_poses(land_mask)

    def create_background_image(self):
        """ "Creates pygame surface with static objects for faster rendering."""
        pygame.font.init()  # needed to import pygame fonts

        if self.gps_env:
            pygame_background_img = pygame.surfarray.make_surface(np.transpose(self.background_img, (1, 0, 2))) #pygame assumes images are (h, w, 3)
            self.pygame_background_img = pygame.transform.scale(pygame_background_img, (self.screen_width, self.screen_height))

            # add attribution text
            img_attribution_font_size = max(8, round(0.35 * np.max(self.arena_buffer)))
            img_attribution_font = pygame.font.SysFont(None, img_attribution_font_size)
            img_attribution_text = img_attribution_font.render(self.background_img_attribution, True, "black")
            img_attribution_text_rect = img_attribution_text.get_rect()

            center_x = self.screen_width - (self.arena_buffer[1][0] + 2*self.boundary_width + 0.5*img_attribution_text_rect.w)

            if img_attribution_text_rect.h < self.arena_buffer[0][1]:
                center_y = self.screen_height - 0.5*self.arena_buffer[0][1]
            elif img_attribution_text_rect.h < self.arena_buffer[1][1]:
                center_y = 0.5*self.arena_buffer[1][1]
            else:
                center_y = self.screen_height - (self.arena_buffer[0][1] + 2*self.boundary_width + 0.5*img_attribution_text_rect.h)

            img_attribution_text_rect.center = [center_x, center_y]

            self.pygame_background_img.blit(img_attribution_text, img_attribution_text_rect)
        else:
            self.pygame_background_img = pygame.Surface((self.screen_width, self.screen_height))
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
            else: # polygon obstacle
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
            for v in self.aquaticus_field_points:
                draw.circle(
                    self.pygame_background_img,
                    (128, 0, 128),
                    self.env_to_screen(self.aquaticus_field_points[v]),
                    radius=5,
                )

    def get_distance_between_2_points(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        Convenience method for returning distance between two points.

        Args:
            start: Starting position to measure from
            end: Point to measure to
        Returns:
            The distance between `start` and `end`
        """
        return np.linalg.norm(np.asarray(start) - np.asarray(end))

    def _set_dones(self, env_idxs):
        """Check all of the end game conditions."""
        blue_scores = self.game_events[Team.BLUE_TEAM]['captures'][env_idxs]
        red_scores = self.game_events[Team.RED_TEAM]['captures'][env_idxs]

        self.dones["blue"][env_idxs] = (blue_scores == self.max_score) & (red_scores != self.max_score)
        self.dones["red"][env_idxs] = (red_scores == self.max_score) & (bluescores != self.max_score)
        self.dones["__all__"][env_idxs] = (
            (blue_scores == self.max_score) | (red_scores == self.max_score) |
            (self.current_time[env_idxs] >= self.max_time) | np.isclose(self.current_time[env_idxs], self.max_time)
        )

        # CLI Message
        cli_message = ""
        if self.render_mode == "human" and np.any(self.dones["__all__"][env_idxs]):
            done_env_idxs = env_idxs[np.where(self.dones["__all__"][env_idxs])[0]]
            cli_message = (
                f"Envs {done_env_idxs} are done with final scores: "
                f"{blue_scores[done_env_idx]}\u2013{red_scores[done_env_idx]} (Blue\u2013Red). "
            )
        
        return cli_message

    def compute_rewards(self, agent_id, team):
        if self.reward_config[agent_id] is None:
            return 0.0

        # Get reward based on the passed in reward function
        return self.reward_config[agent_id](
            agent_id=agent_id,
            team=team,
            agents=self.agents,
            agent_inds_of_team=self.agent_inds_of_team,
            state=self.state,
            prev_state=self.prev_state,
            env_size=self.env_size,
            agent_radius=self.agent_radius,
            catch_radius=self.catch_radius,
            scrimmage_coords=self.scrimmage_coords,
            max_speeds=self.max_speeds,
            tagging_cooldown=self.tagging_cooldown
        )

    def _reset_dones(self, env_idxs):
        """Resets the environments done indicators."""
        self.dones["blue"][env_idxs] = False
        self.dones["red"][env_idxs] = False
        self.dones["__all__"][env_idxs] = False

    def reset(self, env_idxs=None, seed=None, options: Optional[dict] = None):
        """
        Resets the environment so that it is ready to be used.

        Args:
            env_idxs (optional): Which environments to reset.
            seed (optional): Starting seed.
            options (optional): Additonal options for resetting the environment:
                -"normalize_obs": whether or not to normalize observations (sets self.normalize_obs)
                -"normalize_state": whether or not to normalize the global state (sets self.normalize_state)
                    *note: will be overwritten and set to False if self.normalize_obs is False
                -"sync_start": identical initial conditions across vector environments
                -"state_dict": self.state dictionary from a previous episode

                    note: state_dict should be produced by the same (or an equivalently configured) instance of
                          the Pyquaticus class, otherwise the obervations and global state may be inconsistent

                -"init_dict": partial state dictionary for initializing the environment with the following optional keys:
                    -'agent_position'*, 'agent_pos_unit'**,
                    -'agent_speed'*, 'agent_heading'*,
                    -'agent_has_flag'*, 'agent_is_tagged'*, 'agent_tagging_cooldown'*,
                    -'captures'***, 'tags'***, 'grabs'***, 'agent_collisions'***

                      *Note 1: These variables can either be specified as a dict with agent id's as keys, in which case it is not
                               required to specify variable-specific information for each agent and _generate_agent_starts() will
                               be used to generate unspecified information, or as an array, which must be be of length self.num_agents
                               and the indices of the entries must match the indices of the corresponding agents' ids in self.agents

                     **Note 2: 'agent_pos_unit' can be either "m" (meters relative to origin), "wm_xy" (web mercator xy),
                               or "ll" (lat-lon) when self.gps_env is True, and can only be "m" (meters relative to origin)
                               for default (non-gps) environment. If not specified, 'agent_position' will be assumed to be
                               relative to the environment origin (bottom left) and in the default environment units
                               (these can be found by checking self.env_bounds_unit after initializing the environment)

                    ***Note 3: These variables can either be specified as a dict with teams (from the Team class in structs.py) as keys,
                               in which case it is not required to specify variable-specific information for each team and variables will
                               be set to 0 for unspecified teams, or as an array, which must be of length self.agents_of_team and the indides
                               of the entries must match the indices of the corresponding teams' ids in self.agents_of_team
        """
        self._seed(seed=seed)

        if env_idxs is None:
            env_idxs = np.arange(self.n_envs)
        else:
            env_idxs = np.unique(env_idxs)

        self.current_time[env_idxs] = 0.
        self._reset_dones(env_idxs)
        self.active_collisions[env_idxs] = False

        if options is not None:
            self.normalize_obs = options.get("normalize_obs", self.normalize_obs)
            self.normalize_state = options.get("normalize_state", self.normalize_state)
            sync_start = options.get("sync_start", False)

            state_dict = options.get("state_dict", None)
            init_dict = options.get("init_dict", None)
            if state_dict != None and init_dict != None:
                raise Exception("Cannot reset environment with both state_dict and init_dict. Choose either/or.")
        else:
            sync_start = False
            state_dict = None
            init_dict = None

        # Reset env from state_dict 
        if state_dict != None:
            if np.any(self.reset_count[env_idxs] == 0):
                raise Exception(
                    "Resetting from state_dict should only be done for an environment that has been previously reset."
                )
            sync_start_warning = True
            for k in self.state:
                if sync_start and len(state_dict['agent_position'].shape) > 2:
                    if sync_start_warning:
                        print("Warning! Only information from the first environment in state_dict will be used for sync_start.")
                    self.state[k][env_idxs] = copy.deepcopy(state_dict[k][0])
                else:
                    self.state[k][env_idxs] = copy.deepcopy(state_dict[k])
            self._set_player_attributes_from_state()
            self._set_flag_attributes_from_state()
            self._set_game_events_from_state()

        # Reset env from init_dict or standard init
        else:
            if init_dict != None:
                self._set_state_from_init_dict(init_dict, env_idxs, sync_start)
            else:
                flag_homes = np.array([flag.home for flag in self.flags], dtype=float)

                if np.all(self.reset_count == 0):
                    self.state = {
                        "agent_position":            np.full((self.n_envs, self.num_agents, 2), np.nan),
                        "prev_agent_position":       np.full((self.n_envs, self.num_agents, 2), np.nan),
                        "agent_speed":               np.zeros((self.n_envs, self.num_agents), dtype=bool),
                        "agent_heading":             np.zeros((self.n_envs, self.num_agents), dtype=bool),
                        "agent_on_sides":            np.zeros((self.n_envs, self.num_agents), dtype=bool),
                        "agent_oob":                 np.zeros((self.n_envs, self.num_agents), dtype=bool),
                        "agent_in_flag_keepout":     np.zeros((self.n_envs, self.num_agents), dtype=bool),
                        "agent_has_flag":            np.zeros((self.n_envs, self.num_agents), dtype=bool),
                        "agent_is_tagged":           np.zeros((self.n_envs, self.num_agents), dtype=bool),
                        "agent_made_tag":            -np.ones((self.n_envs, self.num_agents), dtype=int),
                        "agent_tagging_cooldown":    np.full((self.n_envs, self.num_agents), self.tagging_cooldown)
                        "flag_position":             np.full((self.n_envs, *flag_homes.shape), flag_homes),
                        "flag_taken":                np.zeros((self.n_envs, len(self.flags)), dtype=bool),
                        "captures":                  np.zeros((self.n_envs, len(self.agents_of_team)), dtype=int),
                        "tags":                      np.zeros((self.n_envs, len(self.agents_of_team)), dtype=int),
                        "grabs":                     np.zeros((self.n_envs, len(self.agents_of_team)), dtype=int),
                        "agent_collisions":          np.zeros((self.n_envs, self.num_agents), dtype=int)
                    }
                else:
                    self.state["agent_oob"][env_idxs] = False
                    self.state["agent_in_flag_keepout"][env_idxs] = False
                    self.state["agent_has_flag"][env_idxs] = False
                    self.state["agent_is_tagged"][env_idxs] = False
                    self.state["agent_made_tag"][env_idxs] = -1
                    self.state["agent_tagging_cooldown"][env_idxs] = self.tagging_cooldown
                    self.state["flag_position"][env_idxs] = flag_homes
                    self.state["flag_taken"][env_idxs] = False
                    self.state["captures"][env_idxs] = 0
                    self.state["tags"][env_idxs] = 0
                    self.state["grabs"][env_idxs] = 0
                    self.state["agent_collisions"][env_idxs] = 0

                agent_poses, agent_speeds, agent_headings, agent_on_sides = self._generate_agent_starts(
                    env_idxs,
                    sync_start,
                    flag_homes_not_picked_up=np.full((env_idxs.shape[0], *flag_homes.shape), flag_homes)
                )
                self.state['agent_position'][env_idxs] = agent_poses
                self.state['prev_agent_position'][env_idxs] = copy.deepcopy(agent_poses)
                self.state['agent_speed'][env_idxs] = agent_speeds
                self.state['agent_heading'][env_idxs] = agent_headings
                self.state['agent_on_sides'][env_idxs] = agent_on_sides

            # set player and flag attributes and self.game_events
            self._set_player_attributes_from_state()
            self._set_flag_attributes_from_state()
            self._set_game_events_from_state()
            for player in self.players.values():
                player.reset(env_idxs) #reset agent-specific dynamics

            self.state['agent_dynamics'][env_idxs] = np.array([[player.state[i] for player in self.players.values()]] for i in env_idxs)

            # run event checks
            self._check_oob(env_idxs)
            self._check_untag_and_flag_keepout(env_idxs)
            self._check_agent_made_tag(env_idxs)
            self._check_flag_pickups(env_idxs)
            #NOTE: _check_flag_captures is not currently necessary b/c initialization does not allow
            #for starting with flag on-sides and state_dict initialization would not start with capture
            #(it would have been detected in the step function checks).

            # obstacles
            self._update_dist_bearing_to_obstacles(env_idxs)

            # lidar
            if self.lidar_obs:
                raise NotImplementedError("Vector environment with Lidar not implemented.")
                # # reset lidar readings
                # self.state["lidar_labels"] = dict()
                # self.state["lidar_ends"] = dict()
                # self.state["lidar_distances"] = dict()

                # for agent_id in self.players:
                #     self.state["lidar_labels"][agent_id] = np.zeros(self.num_lidar_rays, dtype=int)
                #     self.state["lidar_ends"][agent_id] = np.zeros((self.num_lidar_rays, 2))
                #     self.state["lidar_distances"][agent_id] = np.zeros(self.num_lidar_rays)
                #     self._update_lidar()

                # for team in self.agents_of_team:
                #     for label_name, label_idx in self.ray_int_label_map.items():
                #         if label_name.startswith("agent"):
                #             #reset agent lidar detection states (not tagged and do not have flag)
                #             if int(label_name[6:]) in self.agent_inds_of_team[team]:
                #                 self.obj_ray_detection_states[team][label_idx] = LIDAR_DETECTION_CLASS_MAP["teammate"]
                #             else:
                #                 self.obj_ray_detection_states[team][label_idx] = LIDAR_DETECTION_CLASS_MAP["opponent"]

        # Rendering
        if self.render_mode:
            self.render_ctr[env_idxs] = 0
            for i in env_idxs:
                if self.render_saving:
                    self.render_buffer[i] = []
                if self.render_traj_mode:
                    for agent_id in self.players:
                        self.traj_render_buffer[agent_id]["traj"][i] = []
                        self.traj_render_buffer[agent_id]["agent"][i] = []
                        self.traj_render_buffer[agent_id]["history"][i] = []

        # Observations and Info
        obs = {}
        info = {}
        global_state = self.state_to_global_state(self.normalize_state, env_idxs)

        for agent_id in self.agents:
            info[agent_id] = {
                "env_idxs": env_idxs,
                "global_state": global_state
            }
            if self.normalize_obs:
                obs[agent_id], info[agent_id]["unnorm_obs"] = self.state_to_obs(env_idxs, agent_id, self.normalize_obs)
            else:
                obs[agent_id], _ = self.state_to_obs(env_idxs, agent_id, self.normalize_obs)

        # Reset Count
        self.reset_count[env_idxs] += 1

        return obs, info

    def _set_state_from_init_dict(self, init_dict: dict, env_idxs: list, sync_start: bool):
        """
        Args:
            "init_dict": partial state dictionary for initializing the environment with the following optional keys:
                -'agent_position'*, 'agent_pos_unit'**,
                -'agent_speed'*, 'agent_heading'*,
                -'agent_has_flag'*, 'agent_is_tagged'*, 'agent_tagging_cooldown'*,
                -'captures'***, 'tags'***, 'grabs'***, 'agent_collisions'***

                  *Note 1: These variables can either be specified as a dict with agent id's as keys, in which case it is not
                           required to specify variable-specific information for all agents and _generate_agent_starts() will
                           be used to generate unspecified information, or as an array, which must be be of shape (self.n_envs, self.num_agents)
                           and the indices of the entries must match the indices of the corresponding agents' ids in self.agents

                 **Note 2: 'agent_pos_unit' can be either "m" (meters relative to origin), "wm_xy" (web mercator xy),
                           or "ll" (lat-lon) when self.gps_env is True, and can only be "m" (meters relative to origin)
                           for default (non-gps) environment. If not specified, 'agent_position' will be assumed to be
                           relative to the environment origin (bottom left) and in the default environment units
                           (these can be found by checking self.env_bounds_unit after initializing the environment)

                ***Note 3: These variables can either be specified as a dict with teams (from the Team class in structs.py) as keys,
                           in which case it is not required to specify variable-specific information for each team and variables will
                           be set to 0 for unspecified teams, or as an array, which must be of length self.agents_of_team and the indides
                           of the entries must match the indices of the corresponding teams' ids in self.agents_of_team

        Note 4: assumes two teams, and one flag per team.
        """
        ### Setup Order of State Dictionary ###
        flag_homes = np.array([flag.home for flag in self.flags], dtype=float)

        if np.all(self.reset_count == 0):
            self.state = {
                "agent_position":            np.full((self.n_envs, self.num_agents, 2), np.nan),
                "prev_agent_position":       np.full((self.n_envs, self.num_agents, 2), np.nan),
                "agent_speed":               np.zeros((self.n_envs, self.num_agents), dtype=bool),
                "agent_heading":             np.zeros((self.n_envs, self.num_agents), dtype=bool),
                "agent_on_sides":            np.zeros((self.n_envs, self.num_agents), dtype=bool),
                "agent_oob":                 np.zeros((self.n_envs, self.num_agents), dtype=bool),
                "agent_in_flag_keepout":     np.zeros((self.n_envs, self.num_agents), dtype=bool),
                "agent_has_flag":            np.zeros((self.n_envs, self.num_agents), dtype=bool),
                "agent_is_tagged":           np.zeros((self.n_envs, self.num_agents), dtype=bool),
                "agent_made_tag":            -np.ones((self.n_envs, self.num_agents), dtype=int),
                "agent_tagging_cooldown":    np.full((self.n_envs, self.num_agents), self.tagging_cooldown)
                "flag_position":             np.full((self.n_envs, *flag_homes.shape), flag_homes),
                "flag_taken":                np.zeros((self.n_envs, len(self.flags)), dtype=bool),
                "captures":                  np.zeros((self.n_envs, len(self.agents_of_team)), dtype=int),
                "tags":                      np.zeros((self.n_envs, len(self.agents_of_team)), dtype=int),
                "grabs":                     np.zeros((self.n_envs, len(self.agents_of_team)), dtype=int),
                "agent_collisions":          np.zeros((self.n_envs, self.num_agents), dtype=int)
            }
        else:
            self.state["agent_oob"][env_idxs] = False
            self.state["agent_in_flag_keepout"][env_idxs] = False
            self.state["agent_has_flag"][env_idxs] = False
            self.state["agent_is_tagged"][env_idxs] = False
            self.state["agent_made_tag"][env_idxs] = -1
            self.state["agent_tagging_cooldown"][env_idxs] = self.tagging_cooldown
            self.state["flag_position"][env_idxs] = flag_homes
            self.state["flag_taken"][env_idxs] = False
            self.state["captures"][env_idxs] = 0
            self.state["tags"][env_idxs] = 0
            self.state["grabs"][env_idxs] = 0
            self.state["agent_collisions"][env_idxs] = 0

        ### Set Agents from init_dict ###
        ## setup agent pos unit ##
        if "agent_position" in init_dict:
            agent_pos_unit = init_dict.get('agent_pos_unit', None)
            if agent_pos_unit is None:   
                agent_pos_unit = self.env_bounds_unit
            else:
                if self.gps_env:
                    if not (agent_pos_unit == "ll" or agent_pos_unit == "wm_xy" or agent_pos_unit == "m"):
                        raise Exception(
                            f"Unrecognized agent_pos_unit: '{agent_pos_unit}'. Choose from 'll', 'wm_xy', or 'm' when self.gps_env is True"
                        )
                else:
                    if agent_pos_unit != "m":
                        raise Exception(
                            "Agent poses must be specified in meters relative to the origin ('m') when self.gps_env is False"
                        )

        ## position (float variables with 2 values) ##
        agent_pos_dict = {}
        for state_var in ["agent_position"]:
            if state_var in init_dict:
                if isinstance(init_dict[state_var], (list, tuple, np.ndarray)):
                    try:
                        val_flat = [p for v in flatten_generic(init_dict[state_var]) for p in ([np.nan, np.nan] if (v is None or v == np.nan) else [v])]
                        val = np.array(val_flat, dtype=float).reshape(-1, self.num_agents, 2)
                        assert val.shape[0] <= env_idxs.shape[0]
                    except:
                        raise Exception(
                            f"{state_var} {str(type(init_dict[state_var]))[8:-2]} must be be of shape (<={env_idxs.shape[0]}, {self.num_agents}, 2) with entries matching order of self.agents"
                        )
                else:
                    val = {}
                    for agent_id, av in init_dict[state_var]:
                        try:
                            agent_val_flat = [p for v in flatten_generic(av) for p in ([np.nan, np.nan] if (v is None or v == np.nan) else [v])]
                            val[agent_id] = np.array(agent_val_flat, dtype=float).reshape(-1, 2)
                            assert val[agent_id].shape[0] <= env_idxs.shape[0]
                        except:
                            raise Exception(
                                f"{state_var} {str(type(init_dict[state_var]))[8:-2]} values must be be of shape (<={env_idxs.shape[0]}, 2)"
                            )

                for i, agent_id in enumerate(self.agents):
                    if isinstance(val, np.ndarray):
                        agent_val = val[:, i]
                    else: 
                        agent_val = val.get(agent_id, None)
                        if agent_val is None:
                            continue

                    if sync_start and agent_val.shape[0] > 1:
                        agent_val[:] = agent_val[0]
                        print(f"Warning! Only first item from {state_var} will be used for sync_start.")

                    if self.gps_env:
                        if agent_pos_unit == "ll":
                            for i in range(agent_val.shape[0]):
                                agent_val[i] = np.asarray(mt.xy(agent_val[i][1], agent_val[i][0]))
                            agent_val = wrap_mercator_x_dist(agent_val - self.env_bounds[0])
                        elif agent_pos_unit == "wm_xy":
                            agent_val = wrap_mercator_x_dist(agent_val - self.env_bounds[0])
                        else:
                            agent_val /= self.meters_per_mercator_xy
                    agent_pos_dict[agent_id] = agent_val

        ## speed, heading, and tagging_cooldown (float variables with a single value) ##
        agent_spd_dict = {}
        agent_hdg_dict = {}
        for state_var in ["agent_speed", "agent_heading", "agent_tagging_cooldown"]:
            if state_var in init_dict:
                if isinstance(init_dict[state_var], (list, tuple, np.ndarray)):
                    try:
                        val_flat = [np.nan if v is None else v for v in flatten_generic(init_dict[state_var])]
                        val = np.array(val_flat, dtype=float).reshape(-1, self.num_agents)
                        assert val.shape[0] <= env_idxs.shape[0]
                    except:
                        raise Exception(
                            f"{state_var} {str(type(init_dict[state_var]))[8:-2]} must be be of shape (<={env_idxs.shape[0]}, {self.num_agents}) with entries matching order of self.agents"
                        )
                else:
                    val = {}
                    for agent_id, av in init_dict[state_var]:
                        try:
                            agent_val_flat = [np.nan if v is None else v for v in flatten_generic(av)]
                            val[agent_id] = np.array(agent_val_flat, dtype=float).reshape(-1)
                            assert val[agent_id].shape[0] <= env_idxs.shape[0]
                        except:
                            raise Exception(
                                f"{state_var} {str(type(init_dict[state_var]))[8:-2]} values must be be of shape (<={env_idxs.shape[0]},)"
                            )

                for i, agent_id in enumerate(self.agents):
                    if isinstance(val, np.ndarray):
                        agent_val = val[:, i]
                    else: 
                        agent_val = val.get(agent_id, None)
                        if agent_val is None:
                            continue

                    if sync_start and agent_val.shape[0] > 1:
                        agent_val[:] = agent_val[0]
                        print(f"Warning! Only first item from {state_var} will be used for sync_start.")

                    #speed
                    if state_var == "agent_speed":
                        agent_spd_dict[agent_id] = agent_val
                    #heading
                    elif state_var == "agent_heading":
                        agent_hdg_dict[agent_id] = agent_val
                    #tagging cooldown
                    elif state_var == "agent_tagging_cooldown":
                        if sync_start:
                            self.state['agent_tagging_cooldown'][:, i] = (
                                self.state['agent_tagging_cooldown'][0, i] if np.isnan(agent_val[0])
                                else agent_val[0]
                            )
                        else:
                            self.state['agent_tagging_cooldown'][:agent_val.shape[0], i] = np.where(
                                ~np.isnan(agent_val),
                                agent_val, 
                                self.state['agent_tagging_cooldown'][:agent_val.shape[0], i]
                            )

        ## has_flag and is_tagged (boolean variables with a single value) ##
        for state_var in ["agent_has_flag", "agent_is_tagged"]:
            if state_var in init_dict:
                if isinstance(init_dict[state_var], (list, tuple, np.ndarray)):
                    try:
                        val_flat = [False if (v is None or v == np.nan) else v for v in flatten_generic(init_dict[state_var])]
                        val = np.array(val_flat, dtype=bool).reshape(-1, self.num_agents)
                        assert val.shape[0] <= env_idxs.shape[0]
                    except:
                        raise Exception(
                            f"{state_var} {str(type(init_dict[state_var]))[8:-2]} "
                            f"must be be of shape (<={env_idxs.shape[0]}, {self.num_agents}) "
                            f"with entries matching order of self.agents"
                        )
                else:
                    val = {}
                    for agent_id, av in init_dict[state_var]:
                        try:
                            agent_val_flat = [False if (v is None or v == np.nan) else v for v in flatten_generic(av)]
                            val[agent_id] = np.array(agent_val_flat, dtype=bool).reshape(-1)
                            assert val[agent_id].shape[0] <= env_idxs.shape[0]
                        except:
                            raise Exception(
                                f"{state_var} {str(type(init_dict[state_var]))[8:-2]} "
                                f"values must be be of shape (<={env_idxs.shape[0]},)"
                            )

                for i, agent_id in enumerate(self.agents):
                    if isinstance(val, np.ndarray):
                        agent_val = val[:, i]
                    else: 
                        agent_val = val.get(agent_id, None)
                        if agent_val is None:
                            continue

                    if sync_start:
                        if agent_val.shape[0] > 1:
                            print(f"Warning! Only first item from {state_var} will be used for sync_start.")
                        self.state[state_var][env_idxs, i] = agent_val[0]
                    else:
                        self.state[state_var][env_idxs[:len(agent_val)], i] = agent_val

                    #has_flag cannot be True if is_tagged
                    if state_var == "agent_is_tagged":
                        self.state['agent_has_flag'][env_idxs] &= ~self.state['agent_is_tagged'][env_idxs] 
                        agent_spd_dict[agent_id] = agent_val

        ## set flag_taken (note: assumes two teams, and one flag per team) ##
        for team, agent_idxs in self.agent_inds_of_team.items():
            #check for contradiction with number of flags
            n_agents_have_flag = np.sum(self.state["agent_has_flag"][np.ix_(env_idxs, agent_idxs)], axis=-1)
            if np.any(n_agents_have_flag > (len(self.agents_of_team) - 1)):
                raise Exception(
                    f"Team {team} has {n_agents_have_flag} agents with a flag in the {env_idxs.shape[0]} "
                    f"envs and there should not be more than {len(self.agents_of_team) - 1} per env."
                )
            other_team_idx = int(not int(player.team))
            self.state['flag_taken'][env_idxs, other_team_idx] = n_agents_have_flag.astype(bool)

        ## set/generate agent positions and flag positions now that flag pickups have been initialized ##
        flag_homes_not_picked_up = np.array(
            [[2*[np.nan] if self.state['flag_taken'][i, j] else flag_home for j, flag_home in enumerate(flag_homes)]
            for i in env_idxs]
        )
        agent_poses, agent_speeds, agent_headings, agent_on_sides = self._generate_agent_starts(
            env_idxs,
            sync_start,
            flag_homes_not_picked_up=flag_homes_not_picked_up,
            agent_pos_dict=agent_pos_dict,
            agent_spd_dict=agent_spd_dict,
            agent_hdg_dict=agent_hdg_dict,
            agent_has_flag=self.state['agent_has_flag'][env_idxs]
        )
        self.state['agent_position'] = agent_poses
        self.state['prev_agent_position'] = copy.deepcopy(agent_poses)

        ## set agent_speed, agent_heading, and agent_on_sides ##
        self.state['agent_speed'] = agent_speeds
        self.state['agent_heading'] = agent_headings
        self.state['agent_on_sides'] = agent_on_sides

        ### Set Score and Game Events ###
        ## captures, tags, grabs ##
        #TODO: vectorize for n environments (as done above)
        # for state_var in ["captures", "tags", "grabs"]:
        #     if state_var in init_dict:
        #         if isinstance(init_dict[state_var], (list, tuple, np.ndarray)):
        #             num_teams = len(self.agents_of_team)
        #             if len(init_dict[state_var]) == num_teams:
        #                 self.state[state_var] = np.array(init_dict[state_var], dtype=int)
        #             else:
        #                 raise Exception(
        #                     f"{state_var} array must be be of length f{num_teams} with entries matching order of self.agents_of_team"
        #                 )
        #         else:
        #             for i, team in enumerate(init_dict[state_var]):
        #                 self.state[state_var][i] = init_dict[state_var][team]

        ## agent_collisions ##
        #TODO: vectorize for n environments (as done above)
        # if "agent_collisions" in init_dict:
        #     if len(init_dict["agent_collisions"]) == self.num_agents:
        #         self.state[state_var] = np.array(init_dict[state_var], dtype=int)
        #     else:
        #         raise Exception(
        #             f"agent_collisions array must be be of length f{self.num_agents} with entries matching order of self.agents"
        #         )

    def _set_player_attributes_from_state(self):
        for i, player in enumerate(self.players.values()):
            player.pos = self.state['agent_position'][:, i]
            player.prev_pos = self.state['prev_agent_position'][:, i]
            player.speed = self.state['agent_speed'][:, i]
            player.heading = self.state['agent_heading'][:, i]
            player.state = self.state['agent_dynamics'][:, i]

    def _set_flag_attributes_from_state(self):
        for flag in self.flags:
            team_idx = int(flag.team)
            flag.pos = self.state['flag_position'][:, team_idx]
            
            #NOTE: we do not set flag.home because this should already be set in __init__()

    def _generate_agent_starts(
        self,
        env_idxs: Union[list, np.ndarray],
        sync_start: bool,
        flag_homes_not_picked_up: np.ndarray,
        agent_pos_dict: Optional[dict] = None,
        agent_spd_dict: Optional[dict] = None,
        agent_hdg_dict: Optional[dict] = None,
        agent_has_flag: Optional[np.ndarray] = None
    ):
        """
        Generates starting positions, speeds, and headings for all players.
        Note: assumes two teams, and one flag per team.

        Args:
            flag_homes_not_picked_up: The home location of all flags that are not picked up
            agent_pos_dict: positions of a subset of the agents with id's as keys
            agent_spd_dict: speeds of a subset of the agents with id's as keys
            agent_hdg_dict: headings of a subset of the agents with id's as keys

        Returns
        -------
            Initial player positions
            Initial player orientations
            Initial player velocities
            Initial player on_sides bools
        """
        if agent_pos_dict is None:
            agent_pos_dict = {}
        if agent_spd_dict is None:
            agent_spd_dict = {}
        if agent_hdg_dict is None:
            agent_hdg_dict = {}
        if agent_has_flag is None:
            agent_has_flag = np.zeros((len(env_idxs), self.num_agents), dtype=bool)

        agent_poses = np.full((env_idxs.shape[0], self.num_agents, 2), np.nan)
        agent_speeds = np.zeros((env_idxs.shape[0], self.num_agents))
        agent_on_sides = np.ones((env_idxs.shape[0], self.num_agents), dtype=bool)
        if sync_start:
            agent_headings = np.tile(360 * np.random.rand(self.num_agents) - 180, (env_idxs.shape[0], 1))
        else:
            agent_headings = 360 * np.random.rand(env_idxs.shape[0], self.num_agents) - 180

        ### prep valid start poses tracker for gps environment ###
        if self.gps_env:
            if self.on_sides_init:
                valid_init_pos_inds = {
                    team: [np.arange(len(self.valid_team_init_poses[int(team)])) for _ in range(env_idxs.shape[0])]
                    for team in self.agents_of_team
                }
            else:
                vipi = np.arange(len(self.valid_init_poses))
                valid_init_pos_inds = [vipi for _ in range(env_idxs.shape[0])]

        ### initialize agents ###
        for i, player in enumerate(self.players.values()):
            team_idx = int(player.team)
            other_team_idx = int(not team_idx)

            ## position ##
            agent_pos = np.full((env_idxs.shape[0], 2), np.nan)

            # check preset poses
            if player.id in agent_pos_dict:
                agent_pos_preset = agent_pos_dict[player.id]
                agent_pos_preset_idxs = np.where(np.all(~np.isnan(agent_pos_preset), axis=-1))[0]

                if agent_pos_preset_idxs.shape[0] > 0:
                    #check if specified initial pos is in collision
                    valid_preset_pos, collision_types = self._check_valid_pos(
                        new_pos=agent_pos_preset[agent_pos_preset_idxs],
                        agent_idx=i,
                        agent_poses=agent_poses[agent_pos_preset_idxs],
                        flag_homes=flag_homes_not_picked_up[agent_pos_preset_idxs]
                    )
                    if not np.all(valid_preset_pos):
                        preset_pos_idxs_in_collision = np.where(~valid_preset_pos)[0]
                        collision_types_by_env = [
                            [k for k in collision_type if collision_type[k][j]]
                            for j in range(agent_pos_preset_idxs.shape[0]) if j in preset_pos_idxs_in_collision
                        ]
                        raise Exception(
                            f"Specified initial pos(es) ({agent_pos_preset[agent_pos_preset_idxs][preset_pos_idxs_in_collision]}) "
                            f"for agent {player.id} in env(s) {env_idxs[agent_pos_preset_idxs[preset_pos_idxs_in_collision]]} "
                            f"are in collision with environment object types '{collision_types_by_env}'"
                        )

                    #if applicable, check that agents with flag are not on-sides
                    preset_pos_idxs_on_sides_with_flag = np.where(
                        agent_has_flag[agent_pos_preset_idxs, i] &
                        self._check_on_sides(agent_pos_preset[agent_pos_preset_idxs], player.team)
                    )[0]
                    if preset_pos_idxs_on_sides_with_flag.shape[0] > 0:
                        raise Exception(
                            f"Agent {player.id} was specified as having a flag in env(s) "
                            f"{env_idxs[agent_pos_preset_idxs[preset_pos_idxs_on_sides_with_flag]]}, "
                            f"but its specified initial pos(es) ({agent_pos_preset[agent_pos_preset_idxs][preset_pos_idxs_on_sides_with_flag]})"
                            "are on-sides. This combination is not allowed."
                        )
                if sync_start:
                    agent_pos[:] = agent_pos_preset[0]
                else:
                    agent_pos[ :agent_pos_preset.shape[0]] = agent_pos_preset

            # generate poses where not already set
            agent_pos_to_init = np.isnan(agent_pos)
            valid_pos = np.ones(env_idxs.shape[0], dtype=bool)
            valid_pos[np.where(np.any(agent_pos_to_init, axis=-1))[0]] = False

            if not np.all(valid_pos):
                if self.gps_env: #TODO: vectorize, and allow partial initialization
                    assert np.all(np.all(agent_pos_to_init, axis=-1) | np.all(~agent_pos_to_init, axis=-1)), (
                        "For gps mode, partial pos specification (x or y) is not allowed"
                    )
                    while not np.all(valid_pos):
                        if sync_start:
                            if agent_has_flag[0, i]:
                                #initialize agent off-sides in environments where it has flag
                                agent_pos[:] = random.choice(self.valid_team_init_poses[other_team_idx])
                            else:
                                #initialize position in other environments
                                if self.on_sides_init:
                                    start_pos_idx = np.random.choice(valid_init_pos_inds[player.team][0])
                                    valid_init_pos_inds[player.team][0] = np.delete(valid_init_pos_inds[player.team][0], start_pos_idx)
                                    agent_pos[:] = self.valid_team_init_poses[team_idx][start_pos_idx]
                                else:
                                    start_pos_idx = np.random.choice(valid_init_pos_inds[0])
                                    valid_init_pos_inds[0] = np.delete(valid_init_pos_inds[0], start_pos_idx)
                                    agent_pos[:] = self.valid_init_poses[start_pos_idx]
                        else:
                            #initialize agent off-sides in environments where it has flag
                            envs_to_init_has_flag = np.where(agent_has_flag[:, i] & ~valid_pos)[0]
                            for j in envs_to_init_has_flag:
                                agent_pos[j] = random.choice(self.valid_team_init_poses[other_team_idx])

                            #initialize position in other environments
                            other_envs_to_init = np.where(~agent_has_flag[:, i] & ~valid_pos)[0]
                            for j in other_envs_to_init:
                                if self.on_sides_init:
                                    start_pos_idx = np.random.choice(valid_init_pos_inds[player.team][j])
                                    valid_init_pos_inds[player.team][j] = np.delete(valid_init_pos_inds[player.team][j], start_pos_idx)
                                    agent_pos[j] = self.valid_team_init_poses[team_idx][start_pos_idx]
                                else:
                                    start_pos_idx = np.random.choice(valid_init_pos_inds[j])
                                    valid_init_pos_inds[j] = np.delete(valid_init_pos_inds[j], start_pos_idx)
                                    agent_pos[j] = self.valid_init_poses[start_pos_idx]

                        #check if valid pos
                        valid_pos[np.where(~valid_pos)[0]], _ = self._check_valid_pos(agent_pos, i, agent_poses, flag_homes_not_picked_up)
                else:
                    while not np.all(valid_pos):
                        if sync_start:
                            agent_pos[:] = np.random.choice((-1,1), size=2) * np.random.rand(2) * (self.env_size/2 - self.agent_radius[i]) + self.env_size/2
                        else:
                            envs_to_init_x = np.where(~valid_pos & agent_pos_to_init[:, 0])[0]
                            envs_to_init_y = np.where(~valid_pos & agent_pos_to_init[:, 1])[0]

                            agent_pos[envs_to_init_x, 0] = (
                                np.random.choice([-1, 1], size=envs_to_init_x.shape[0]) *
                                np.random.rand(envs_to_init_x.shape[0]) *
                                (self.env_size[0]/2 - self.agent_radius[i]) + self.env_size[0]/2
                            )
                            agent_pos[envs_to_init_y, 1] = (
                                np.random.choice([-1, 1], size=envs_to_init_y.shape[0]) *
                                np.random.rand(envs_to_init_y.shape[0]) *
                                (self.env_size[1]/2 - self.agent_radius[i]) + self.env_size[1]/2
                            )

                        #check if valid pos
                        collision_free = self._check_valid_pos(agent_pos, i, agent_poses, flag_homes_not_picked_up)[0]
                        on_sides = self._check_on_sides(agent_pos, player.team)

                        #check agent off-sides in environments where it has flag
                        envs_to_init_has_flag = np.where(~valid_pos & agent_has_flag[:, i])[0]
                        valid_pos[envs_to_init_has_flag] = collision_free[envs_to_init_has_flag] & ~on_sides[envs_to_init_has_flag]

                        #check position in other environments
                        other_envs_to_init = np.where(~valid_pos & ~agent_has_flag[:, i])[0]
                        if self.on_sides_init:
                            valid_pos[other_envs_to_init] = collision_free[other_envs_to_init] & on_sides[other_envs_to_init]
                        else:
                            valid_pos[other_envs_to_init] = collision_free[other_envs_to_init]

            # save agent_pos
            agent_poses[:, i] = agent_pos

            ## on-sides ##
            agent_on_sides[:, i] = self._check_on_sides(agent_pos, player.team)

            ## picked up flag (if any) ##
            env_has_flag_idxs = np.where(agent_has_flag[:, i])[0]
            self.state['flag_position'][env_has_flag_idxs, other_team_idx] = copy.deepcopy(agent_pos[env_has_flag_idxs])

            ## speed ##
            if player.id in agent_spd_dict:
                agent_spd_preset = np.where(np.isnan(agent_spd_dict[player.id]), 0., agent_spd_dict[player.id])
                if sync_start:
                    agent_speeds[:, i] = agent_spd_preset[0]
                else:
                    agent_speeds[ :agent_spd_preset.shape[0], i] = agent_spd_preset

            ## heading ##
            if player.id in agent_hdg_dict:
                agent_hdg_preset = agent_hdg_dict[player.id]
                if sync_start:
                    agent_headings[:, i] = agent_hdg_preset[0] if not np.isnan(agent_hdg_preset[0]) else agent_headings[0, i]
                else:
                    agent_hdg_preset_idxs = np.where(~np.isnan(agent_hdg_preset))[0]
                    if np.any((agent_hdg_preset < -180) | (agent_hdg_preset > 180)):
                        raise Exception(
                            f"At least one initial heading specified for {player.id} ({agent_hdg_preset}) does not fall between -180 and 180 degrees."
                        )
                    agent_headings[agent_hdg_preset_idxs, i] = agent_hdg_preset[agent_hdg_preset_idxs]

        return agent_poses, agent_speeds, agent_headings, agent_on_sides

    def _check_valid_pos(self, new_pos, agent_idx, agent_poses, flag_homes):
        """
        Returns
        -------
            collision type: string or None
            collision bool
        """
        new_pos = np.asarray(new_pos).reshape(-1, 2)
        agent_poses = np.asarray(agent_poses)
        flag_homes = np.asarray(flag_homes)

        valid_pos = np.ones(new_pos.shape[0])
        collision_types = {
            "agent": np.zeros(new_pos.shape[0], dtype=bool),
            "flag": np.zeros(new_pos.shape[0], dtype=bool),
            "obstacle": np.zeros(new_pos.shape[0], dtype=bool)
        }

        #agents
        new_pos_by_env = new_pos[:, None, :]
        agent_dists = np.linalg.norm(new_pos_by_env - agent_poses, axis=-1)
        radii = np.tile(self.agent_radius, (self.agent_radius.shape[0], 1))
        envs_with_agent_collision = np.where(np.any(agent_dists <= self.agent_radius[agent_idx] + radii, axis=-1))[0]
        valid_pos[envs_with_agent_collision] = False
        collision_types['agent'][envs_with_agent_collision] = True

        #flags
        flag_dists = np.linalg.norm(new_pos_by_env - flag_homes, axis=-1)
        envs_with_flag_collision = np.where(np.any(flag_dists <= self.flag_keepout_radius, axis=-1))[0]
        valid_pos[envs_with_flag_collision] = False
        collision_types['flag'][envs_with_flag_collision] = True

        #obstacles
        ## TODO: add check to make sure agent isn't spawned inside an obstacle
        envs_with_obstacle_collision = np.where(detect_collision(new_pos, self.agent_radius[agent_idx], self.obstacle_geoms))[0]
        valid_pos[envs_with_obstacle_collision] = False
        collision_types['obstacle'][envs_with_obstacle_collision] = True

        return valid_pos, collision_types

    def _update_dist_bearing_to_obstacles(self, env_idxs):
        """Computes the distance and heading from each player to each obstacle"""
        # dist_bearing_to_obstacles = dict()
        # for i, player in enumerate(self.players.values()):
        #     player_pos = player.pos
        #     player_dists_to_obstacles = list()
        #     for obstacle in self.obstacles:
        #         # TODO: vectorize
        #         dist_to_obstacle = obstacle.distance_from(player_pos, radius=self.agent_radius[i], heading=player.heading)
        #         player_dists_to_obstacles.append(dist_to_obstacle)
        #     dist_bearing_to_obstacles[player.id] = player_dists_to_obstacles
        # self.dist_bearing_to_obstacles = dist_bearing_to_obstacles
        pass

    def _build_base_env_geom(
        self,
        env_bounds,
        flag_homes,
        scrimmage_coords,
        env_bounds_unit: str,
        flag_homes_unit: str,
        scrimmage_coords_unit: str,
        agent_radius: np.ndarray,
        flag_radius: float,
        flag_keepout_radius: float,
        catch_radius: float,
        slip_radius: np.ndarray,
        lidar_range: float,
    ):
        #### Basic Checks ####
        ### auto env_bounds and flag_homes ###
        if self._is_auto_string(env_bounds) and (
            self._is_auto_string(flag_homes[Team.BLUE_TEAM])
            or self._is_auto_string(flag_homes[Team.RED_TEAM])
        ):
            raise Exception(
                "Either env_bounds or blue AND red flag homes must be set in config_dict (cannot both be 'auto')"
            )

        if self.gps_env:
            global cx
            global mt
            global _sm2ll
            global Geodesic
            import contextily as cx
            import mercantile as mt
            from contextily.tile import _sm2ll
            from geographiclib.geodesic import Geodesic

            ### environment bounds ###
            if self._is_auto_string(env_bounds):
                flag_home_blue = np.asarray(flag_homes[Team.BLUE_TEAM])
                flag_home_red = np.asarray(flag_homes[Team.RED_TEAM])

                if flag_homes_unit == "m":
                    raise Exception(
                        "Flag homes must be specified in absolute coordinates (lat/long or web mercator xy) to auto-generate gps environment bounds."
                    )
                elif flag_homes_unit == "ll":
                    # convert flag poses to web mercator xy
                    flag_home_blue = np.asarray(mt.xy(*flag_home_blue[-1::-1]))
                    flag_home_red = np.asarray(mt.xy(*flag_home_red[-1::-1]))

                flag_vec = wrap_mercator_x(flag_home_blue - flag_home_red)
                flag_distance = np.linalg.norm(flag_vec)

                if flag_distance == 0:
                    raise Exception(
                        "Flag homes of opposite teams cannot be in the same location."
                    )

                flag_unit_vec = flag_vec / flag_distance
                flag_perp_vec = np.array([-flag_unit_vec[1], flag_unit_vec[0]])

                # check if bounds will wrap more than once around the world
                env_width = (
                    np.abs(flag_vec[0]) +
                    (flag_distance/6) * np.abs(flag_unit_vec[0]) +
                    (flag_distance/3) * np.abs(flag_perp_vec[0])
                )
                if env_width > 2*EPSG_3857_EXT_X:
                    raise Exception(
                        "Automatic construction of environment bounds based on flag poses failed. \
Desired environment width is greater than earth's equatorial diameter."
                    )

                # assuming default aquaticus field size ratio drawn on web mercator, these bounds will contain it
                bounds_pt1 = flag_home_blue + (flag_distance/6) * flag_unit_vec + (flag_distance/3) * flag_perp_vec
                bounds_pt2 = flag_home_blue + (flag_distance/6) * flag_unit_vec + (flag_distance/3) * -flag_perp_vec
                bounds_pt3 = flag_home_red + (flag_distance/6) * -flag_unit_vec + (flag_distance/3) * flag_perp_vec
                bounds_pt4 = flag_home_red + (flag_distance/6) * -flag_unit_vec + (flag_distance/3) * -flag_perp_vec
                bounds_points = wrap_mercator_x([bounds_pt1, bounds_pt2, bounds_pt3, bounds_pt4])

                # determine bounds
                if np.sign(flag_vec[0]) == 1:
                    # blue flag on right
                    if np.sign(flag_vec[1]) == 1:
                        xmin = bounds_points[2][0]
                        xmax = bounds_points[1][0]
                    elif np.sign(flag_vec[1]) == -1:
                        xmin = bounds_points[3][0]
                        xmax = bounds_points[0][0]
                    else:
                        xmin = bounds_points[2][0]
                        xmax = bounds_points[0][0]
                elif np.sign(flag_vec[0]) == -1:
                    # red flag on right
                    if np.sign(flag_vec[1]) == 1:
                        xmin = bounds_points[0][0]
                        xmax = bounds_points[3][0]
                    elif np.sign(flag_vec[1]) == -1:
                        xmin = bounds_points[1][0]
                        xmax = bounds_points[2][0]
                    else:
                        xmin = bounds_points[0][0]
                        xmax = bounds_points[2][0]
                else:
                    # blue flag x == red flag x
                    if np.sign(flag_vec[1]) == 1:
                        #blue flag on top
                        xmin = bounds_points[0][0]
                        xmax = bounds_points[1][0]
                    else:
                        #red flag on top
                        xmin = bounds_points[1][0]
                        xmax = bounds_points[0][0]

                env_bounds = np.zeros((2, 2))
                env_bounds[0][0] = xmin #left x bound
                env_bounds[1][0] = xmax #right y bound
                env_bounds[0][1] = np.min(bounds_points[:, 1]) #lower y bound
                env_bounds[1][1] = np.max(bounds_points[:, 1]) #upper y bound
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
                            "Flag locations must be specified in absolute coordinates (lat/long or web mercator xy) \
when gps environment bounds are specified in meters"
                        )

                    if len(env_bounds.shape) == 1:
                        env_bounds = np.array([
                            (0.0, 0.0),
                            env_bounds
                        ])
                    else:
                        raise Exception(
                            "Environment bounds in meters should be given as [xmax, ymax]"
                        )

                    if np.any(env_bounds[1] == 0.0):
                        raise Exception(
                            "Environment max bounds must be > 0 when specified in meters"
                        )

                    # get flag midpoint
                    flag_home_blue = np.asarray(flag_homes[Team.BLUE_TEAM])
                    flag_home_red = np.asarray(flag_homes[Team.RED_TEAM])

                    if flag_homes_unit == "wm_xy":
                        flag_home_blue = np.flip(_sm2ll(*flag_home_blue))
                        flag_home_red = np.flip(_sm2ll(*flag_home_red))

                    geodict_flags = Geodesic.WGS84.Inverse(
                        lat1=flag_home_blue[0],
                        lon1=flag_home_blue[1],
                        lat2=flag_home_red[0],
                        lon2=flag_home_red[1]
                    )
                    geodict_flag_midpoint = Geodesic.WGS84.Direct(
                        lat1=flag_home_blue[0],
                        lon1=flag_home_blue[1],
                        azi1=geodict_flags["azi1"],
                        s12=geodict_flags["s12"]/2
                    )
                    flag_midpoint = (
                        geodict_flag_midpoint["lat2"],
                        geodict_flag_midpoint["lon2"]
                    )

                    # vertical bounds
                    env_top = Geodesic.WGS84.Direct(
                        lat1=flag_midpoint[0],
                        lon1=flag_midpoint[1],
                        azi1=0,  # degrees
                        s12=env_bounds[1][1]/2
                    )["lat2"]
                    env_bottom = Geodesic.WGS84.Direct(
                        lat1=flag_midpoint[0],
                        lon1=flag_midpoint[1],
                        azi1=180,  # degrees
                        s12=env_bounds[1][1]/2
                    )["lat2"]

                    # horizontal bounds
                    geoc_lat = np.arctan(
                        (POLAR_RADIUS / EQUATORIAL_RADIUS)**2 * np.tan(np.deg2rad(flag_midpoint[0]))
                    )
                    small_circle_circum = np.pi * 2*EQUATORIAL_RADIUS * np.cos(geoc_lat)
                    env_left = flag_midpoint[1] - 360*(0.5*env_bounds[1][0] / small_circle_circum)
                    env_right = flag_midpoint[1] + 360*(0.5*env_bounds[1][0] / small_circle_circum)

                    env_left = angle180(env_left)
                    env_right = angle180(env_right)

                    # convert bounds to web mercator xy
                    env_bounds = np.array([
                        mt.xy(env_left, env_bottom),
                        mt.xy(env_right, env_top)
                    ])
                else:
                    # reshape to group by min and max bounds
                    env_bounds = env_bounds.reshape((2,2))

                    # check for exceptions
                    if np.any(env_bounds[0] == env_bounds[1]):
                        raise Exception(
                            f"The specified environment bounds {env_bounds} form an environment of zero area."
                        )

                    # convert bounds to web mercator xy
                    if env_bounds_unit == "ll":
                        env_bounds = np.array([
                            mt.xy(*env_bounds[0]),
                            mt.xy(*env_bounds[1])
                        ])

            # unit
            env_bounds_unit = "wm_xy"

            # match x bound sign (if applicable)
            if np.abs(env_bounds[0, 0]) == EPSG_3857_EXT_X:
                env_bounds[0, 0] = -EPSG_3857_EXT_X
            if np.abs(env_bounds[1, 0]) == EPSG_3857_EXT_X:
                env_bounds[1, 0] = EPSG_3857_EXT_X

            # clip y bounds
            if np.any(np.abs(env_bounds[:, 1]) > EPSG_3857_EXT_Y):
                print(f"Warning! Clipping environment latitude bounds {env_bounds[:, 1]} to fall between {[-EPSG_3857_EXT_Y, EPSG_3857_EXT_Y]}")
                env_bounds[:, 1] = np.clip(env_bounds[:, 1], -EPSG_3857_EXT_Y, EPSG_3857_EXT_Y)

            # environment size, diagonal, and corners
            self.env_size = wrap_mercator_x_dist(np.diff(env_bounds, axis=0)[0])
            self.env_diag = np.linalg.norm(self.env_size)

            self.env_ll = np.array([0.0, 0.0])              #ll = lower left
            self.env_lr = np.array([self.env_size[0], 0.0]) #lr = lower right
            self.env_ur = np.array(self.env_size)           #ur = upper right
            self.env_ul = np.array([0.0, self.env_size[1]]) #ul = upper left

            self.env_corners = np.array([
                self.env_ll,
                self.env_lr,
                self.env_ur,
                self.env_ul
            ])

            ### flags home ###
            # auto home
            if self._is_auto_string(flag_homes[Team.BLUE_TEAM]) and self._is_auto_string(flag_homes[Team.RED_TEAM]):
                flag_homes[Team.BLUE_TEAM] = wrap_mercator_x(
                    env_bounds[0] + np.array([1/8 * self.env_size[0], 0.5 * self.env_size[1]])
                )
                flag_homes[Team.RED_TEAM] = wrap_mercator_x(
                    env_bounds[0] + np.array([7/8 * self.env_size[0], 0.5 * self.env_size[1]])
                )
            elif self._is_auto_string(flag_homes[Team.BLUE_TEAM]) or self._is_auto_string(flag_homes[Team.RED_TEAM]):
                raise Exception("Flag homes should be either all 'auto', or all specified")
            else:
                if flag_homes_unit == "m":
                    raise Exception("'m' (meters) should only be used to specify flag homes when gps_env is False")

                flag_homes[Team.BLUE_TEAM] = np.asarray(flag_homes[Team.BLUE_TEAM])
                flag_homes[Team.RED_TEAM] = np.asarray(flag_homes[Team.RED_TEAM])

                if flag_homes_unit == "ll":
                    # convert flag poses to web mercator xy
                    flag_homes[Team.BLUE_TEAM] = mt.xy(*flag_homes[Team.BLUE_TEAM][-1::-1])
                    flag_homes[Team.RED_TEAM] = mt.xy(*flag_homes[Team.RED_TEAM][-1::-1])

            # normalize relative to environment bounds
            flag_homes[Team.BLUE_TEAM] = wrap_mercator_x_dist(flag_homes[Team.BLUE_TEAM] - env_bounds[0]) 
            flag_homes[Team.RED_TEAM] = wrap_mercator_x_dist(flag_homes[Team.RED_TEAM] - env_bounds[0])

            # blue flag
            if (
                np.any(flag_homes[Team.BLUE_TEAM] <= 0) or
                np.any(flag_homes[Team.BLUE_TEAM] >= self.env_size)
            ):
                raise Exception(
                    f"Blue flag home {flag_homes[Team.BLUE_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}"
                )

            #red flag
            if (
                np.any(flag_homes[Team.RED_TEAM] <= 0) or
                np.any(flag_homes[Team.RED_TEAM] >= self.env_size)
            ):
                raise Exception(
                    f"Red flag home {flag_homes[Team.RED_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}"
                )

            # unit
            flag_homes_unit = "wm_xy"

            ### scrimmage line ###
            if self._is_auto_string(scrimmage_coords):
                flags_vec = flag_homes[Team.BLUE_TEAM] - flag_homes[Team.RED_TEAM]

                scrim_vec1 = np.array([-flags_vec[1], flags_vec[0]])
                scrim_vec2 = np.array([flags_vec[1], -flags_vec[0]])
                flags_midpoint = 0.5 * (flag_homes[Team.BLUE_TEAM] + flag_homes[Team.RED_TEAM])

                scrimmage_coord1 = self._get_polygon_intersection(flags_midpoint, scrim_vec1, self.env_corners)[1]
                scrimmage_coord2 = self._get_polygon_intersection(flags_midpoint, scrim_vec2, self.env_corners)[1]
                scrimmage_coords = np.asarray([scrimmage_coord1, scrimmage_coord2])
            else:
                # check and convert units if necessary
                if scrimmage_coords_unit == "m":
                    raise Exception(
                        "'m' (meters) should only be used to specify flag homes when gps_env is False"
                    )
                elif scrimmage_coords_unit == "wm_xy":
                    pass
                elif scrimmage_coords_unit == "ll":
                    scrimmage_coords_1 = mt.xy(*scrimmage_coords[0][-1::-1])
                    scrimmage_coords_2 = mt.xy(*scrimmage_coords[1][-1::-1])
                    scrimmage_coords = np.array([scrimmage_coords_1, scrimmage_coords_2])
                else:
                    raise Exception(
                        f"Unit '{scrimmage_coords_unit}' not recognized. Please choose from 'll' or 'wm_xy' for gps environments."
                    )

                # check for exceptions
                if np.all(scrimmage_coords[0] == scrimmage_coords[1]):
                    raise Exception(
                        "Scrimmage line must be specified with two DIFFERENT coordinates"
                    )
                if scrimmage_coords[0][0] == scrimmage_coords[1][0] and (
                    scrimmage_coords[0][0] == env_bounds[0][0] or
                    scrimmage_coords[0][0] == env_bounds[1][0]
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} lie on the same vertical edge of the env boundary."
                    )
                if scrimmage_coords[0][1] == scrimmage_coords[1][1] and (
                    scrimmage_coords[0][1] == env_bounds[0][1] or
                    scrimmage_coords[0][1] == env_bounds[1][1]
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} lie on the same horizontal edge of the env boundary."
                    )

                # normalize relative to environment bounds
                scrimmage_coords_wm_xy = copy.deepcopy(scrimmage_coords)
                scrimmage_coords = wrap_mercator_x_dist(scrimmage_coords - env_bounds[0])

                # extend scrimmage line if necessary to fully divide environment
                if scrimmage_coords[1][1] == scrimmage_coords[0][1]:  # horizontal line
                    extended_point_1 = [-scrimmage_coords[0][0], scrimmage_coords[0][1]]
                    extended_point_2 = [self.env_size[0] + (self.env_size[0] - scrimmage_coords[1][0]), scrimmage_coords[1][1]]
                elif scrimmage_coords[1][0] == scrimmage_coords[0][0]:  # vertical line
                    extended_point_1 = [scrimmage_coords[0][0], -scrimmage_coords[0][1]]
                    extended_point_2 = [scrimmage_coords[1][0], self.env_size[1] + (self.env_size[1] - scrimmage_coords[1][1])]
                else:
                    scrimmage_slope = (scrimmage_coords[1][1] - scrimmage_coords[0][1]) / (scrimmage_coords[1][0] - scrimmage_coords[0][0])

                    # compute intersection point for the first scrimmage_coord
                    t_env_bound_x1 = -scrimmage_coords[0][0] * scrimmage_slope
                    t_env_bound_y1 = (-scrimmage_coords[0][1] / scrimmage_slope if scrimmage_slope != 0 else 0)
                    t_env_bound_x2 = (self.env_size[0] - scrimmage_coords[0][0]) * scrimmage_slope
                    t_env_bound_y2 = ((self.env_size[1] - scrimmage_coords[0][1]) / scrimmage_slope if scrimmage_slope != 0 else 0)
                    t_env_bounds = [t_env_bound_x1, t_env_bound_y1, t_env_bound_x2, t_env_bound_y2]
                    max_t = max(t_env_bounds) * 10
                    min_t = min(t_env_bounds) * 10

                    extended_point_1 = (scrimmage_coords[0] + np.array([max_t, max_t * scrimmage_slope]) if max_t > 0 else scrimmage_coords[0])
                    extended_point_2 = (scrimmage_coords[0] + np.array([min_t, min_t * scrimmage_slope]) if min_t < 0 else scrimmage_coords[0])

                extended_scrimmage_coords = np.array([extended_point_1, extended_point_2])
                full_scrim_line = LineString(extended_scrimmage_coords)
                scrim_line_env_intersection = intersection(full_scrim_line, Polygon(self.env_corners))

                if (
                    scrim_line_env_intersection.is_empty or
                    len(scrim_line_env_intersection.coords) == 1 #only intersects a vertex
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords_wm_xy} create a line that does not bisect the environment of bounds {env_bounds}"
                    )
                else:
                    scrim_line_env_intersection = np.array(scrim_line_env_intersection.coords)

                    # intersection points should lie on boundary (if they don't then the line doesn't bisect the env)
                    if not (
                        (
                            (0 <= scrim_line_env_intersection[0][0] <= self.env_size[0]) and
                            ((scrim_line_env_intersection[0][1] == 0) or (scrim_line_env_intersection[0][1] == self.env_size[1]))
                            ) or
                        (
                            (0 <= scrim_line_env_intersection[0][1] <= self.env_size[1]) and
                            ((scrim_line_env_intersection[0][0] == 0) or (scrim_line_env_intersection[0][0] == self.env_size[0]))
                            )
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords_wm_xy} create a line that does not bisect the environment of bounds {env_bounds}"
                        )
                    if not (
                        (
                            (0 <= scrim_line_env_intersection[1][0] <= self.env_size[0]) and
                            ((scrim_line_env_intersection[1][1] == 0) or (scrim_line_env_intersection[1][1] == self.env_size[1]))
                            ) or
                        (
                            (0 <= scrim_line_env_intersection[1][1] <= self.env_size[1]) and
                            ((scrim_line_env_intersection[1][0] == 0) or (scrim_line_env_intersection[1][0] == self.env_size[0]))
                            )
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords_wm_xy} create a line that does not bisect the environment of bounds {env_bounds}"
                        )

                    scrimmage_coords = scrim_line_env_intersection

            # unit
            scrimmage_coords_unit = "wm_xy"

            ### agent and flag geometries ###
            lon1, lat1 = _sm2ll(*env_bounds[0])
            lon2, lat2 = _sm2ll(*env_bounds[1])
            lon_diff = longitude_diff_west2east(lon1, lon2)

            if np.abs(lat1) > np.abs(lat2):
                lat = lat1
            else:
                lat = lat2

            geoc_lat = np.arctan((POLAR_RADIUS / EQUATORIAL_RADIUS) ** 2 * np.tan(np.deg2rad(lat)))
            small_circle_circum = np.pi * 2 * EQUATORIAL_RADIUS * np.cos(geoc_lat)

            # use most warped (squished) horizontal environment border to underestimate the number of
            # meters per mercator xy, therefore overestimate how close objects are to one another
            self.meters_per_mercator_xy = (small_circle_circum * (lon_diff / 360) / self.env_size[0])
            agent_radius /= self.meters_per_mercator_xy
            flag_radius /= self.meters_per_mercator_xy
            catch_radius /= self.meters_per_mercator_xy
            flag_keepout_radius /= self.meters_per_mercator_xy
            slip_radius /= self.meters_per_mercator_xy
            lidar_range /= self.meters_per_mercator_xy

        else:
            ### environment bounds ###
            if env_bounds_unit != "m":
                raise Exception(
                    "Environment bounds unit must be meters ('m') when gps_env is False"
                )
            if self._is_auto_string(env_bounds):
                if np.any(np.sign([flag_homes[Team.BLUE_TEAM], flag_homes[Team.RED_TEAM]]) == -1):
                    raise Exception(
                        "Flag coordinates must be in the positive quadrant when gps_env is False"
                    )
                if np.any(np.sign([flag_homes[Team.BLUE_TEAM], flag_homes[Team.RED_TEAM]]) == 0):
                    raise Exception(
                        "Flag coordinates must not lie on the axes of the positive quadrant when gps_env is False"
                    )
                #environment size
                flag_xmin = min(flag_homes[Team.BLUE_TEAM][0], flag_homes[Team.RED_TEAM][0])
                flag_ymin = min(flag_homes[Team.BLUE_TEAM][1], flag_homes[Team.RED_TEAM][1])

                flag_xmax = max(flag_homes[Team.BLUE_TEAM][0], flag_homes[Team.RED_TEAM][0])
                flag_ymax = max(flag_homes[Team.BLUE_TEAM][1], flag_homes[Team.RED_TEAM][1])

                self.env_size = np.array([flag_xmax + flag_xmin, flag_ymax + flag_ymin])
                env_bounds = np.array([(0.0, 0.0), self.env_size])
            else:
                env_bounds = np.asarray(env_bounds)

                if len(env_bounds.shape) == 1:
                    if np.any(env_bounds == 0.0):
                        raise Exception("Environment max bounds must be > 0 when specified in meters")

                    # environment size
                    self.env_size = env_bounds
                    env_bounds = np.array([
                        (0.0, 0.0),
                        env_bounds
                    ])
                else:
                    raise Exception(
                            "Environment bounds in meters should be given as [xmax, ymax]"
                        )

            # environment diagonal and corners
            self.env_diag = np.linalg.norm(self.env_size)

            self.env_ll = np.array([0.0, 0.0])              #ll = lower left
            self.env_lr = np.array([self.env_size[0], 0.0]) #lr = lower right
            self.env_ur = np.array(self.env_size)           #ur = upper right
            self.env_ul = np.array([0.0, self.env_size[1]]) #ul = upper left

            self.env_corners = np.array([
                self.env_ll,
                self.env_lr,
                self.env_ur,
                self.env_ul
            ])

            ### flags home ###
            # auto home
            if self._is_auto_string(flag_homes[Team.BLUE_TEAM]) and self._is_auto_string(flag_homes[Team.RED_TEAM]):
                if flag_homes_unit == "ll" or flag_homes_unit == "wm_xy":
                    raise Exception(
                        "'ll' (Lat/Long) and 'wm_xy' (web mercator xy) units should only be used when gps_env is True"
                    )
                flag_homes[Team.BLUE_TEAM] = np.array([1/8*self.env_size[0], 0.5*self.env_size[1]])
                flag_homes[Team.RED_TEAM] = np.array([7/8*self.env_size[0], 0.5*self.env_size[1]])
            elif self._is_auto_string(flag_homes[Team.BLUE_TEAM]) or self._is_auto_string(flag_homes[Team.RED_TEAM]):
                raise Exception(
                    "Flag homes are either all 'auto', or all specified"
                )
            else:
                flag_homes[Team.BLUE_TEAM] = np.asarray(flag_homes[Team.BLUE_TEAM])
                flag_homes[Team.RED_TEAM] = np.asarray(flag_homes[Team.RED_TEAM])

            #blue flag
            if (
                np.any(flag_homes[Team.BLUE_TEAM] <= env_bounds[0]) or
                np.any(flag_homes[Team.BLUE_TEAM] >= env_bounds[1])
            ):
                raise Exception(
                    f"Blue flag home {flag_homes[Team.BLUE_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}"
                )

            #red flag
            if (
                np.any(flag_homes[Team.RED_TEAM] <= env_bounds[0]) or
                np.any(flag_homes[Team.RED_TEAM] >= env_bounds[1])
            ):
                raise Exception(f"Red flag home {flag_homes[Team.RED_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}")

            ### scrimmage line ###
            if self._is_auto_string(scrimmage_coords):
                flags_vec = flag_homes[Team.BLUE_TEAM] - flag_homes[Team.RED_TEAM]

                scrim_vec1 = np.array([-flags_vec[1], flags_vec[0]])
                scrim_vec2 = np.array([flags_vec[1], -flags_vec[0]])
                flags_midpoint = 0.5 * (flag_homes[Team.BLUE_TEAM] + flag_homes[Team.RED_TEAM])

                scrimmage_coord1 = self._get_polygon_intersection(flags_midpoint, scrim_vec1, self.env_corners)[1]
                scrimmage_coord2 = self._get_polygon_intersection(flags_midpoint, scrim_vec2, self.env_corners)[1]
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
                    scrimmage_coords[0][0] == env_bounds[0][0] or
                    scrimmage_coords[0][0] == env_bounds[1][0]
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} cannot lie on the same edge of the env boundary"
                    )
                if scrimmage_coords[0][1] == scrimmage_coords[1][1] and (
                    scrimmage_coords[0][1] == env_bounds[0][1] or
                    scrimmage_coords[0][1] == env_bounds[1][1]
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} cannot lie on the same edge of the env boundary"
                    )

                # env biseciton check
                full_scrim_line = LineString(scrimmage_coords)
                scrim_line_env_intersection = intersection(full_scrim_line, Polygon(self.env_corners))

                if (
                    scrim_line_env_intersection.is_empty or
                    len(scrim_line_env_intersection.coords) == 1 #only intersects a vertex
                ):
                    raise Exception(
                        f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                    )
                else:
                    scrim_line_env_intersection = np.array(scrim_line_env_intersection.coords)

                    # intersection points should lie on boundary (if they don't then the line doesn't bisect the env)
                    if not (
                        (
                            (env_bounds[0][0] <= scrim_line_env_intersection[0][0] <= env_bounds[1][0]) and
                            ((scrim_line_env_intersection[0][1] == env_bounds[0][1]) or (scrim_line_env_intersection[0][1] == env_bounds[1][1]))
                            ) or
                        (
                            (env_bounds[0][1] <= scrim_line_env_intersection[0][1] <= env_bounds[1][1]) and
                            ((scrim_line_env_intersection[0][0] == env_bounds[0][0]) or (scrim_line_env_intersection[0][0] == env_bounds[1][0]))
                            )
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                        )
                    if not (
                        (
                            (env_bounds[0][0] <= scrim_line_env_intersection[1][0] <= env_bounds[1][0]) and
                            ((scrim_line_env_intersection[1][1] == env_bounds[0][1]) or (scrim_line_env_intersection[1][1] == env_bounds[1][1]))
                            ) or
                        (
                            (env_bounds[0][1] <= scrim_line_env_intersection[1][1] <= env_bounds[1][1]) and
                            ((scrim_line_env_intersection[1][0] == env_bounds[0][0]) or (scrim_line_env_intersection[1][0] == env_bounds[1][0]))
                            )
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                        )

        ### Set Attributes ###
        # bounds, edges, and scrimmage line
        self.env_bounds = env_bounds
        self.env_bounds_unit = env_bounds_unit

        self.env_edges = np.array([
            [self.env_ll, self.env_lr],
            [self.env_lr, self.env_ur],
            [self.env_ur, self.env_ul],
            [self.env_ul, self.env_ll]
        ])

        self.scrimmage_coords = scrimmage_coords
        self.scrimmage_coords_unit = scrimmage_coords_unit
        self.scrimmage_vec = scrimmage_coords[1] - scrimmage_coords[0]

        # environment angle (rotation)
        rot_vec = self.env_lr - self.env_ll
        self.env_rot_angle = np.arctan2(rot_vec[1], rot_vec[0])
        s, c = np.sin(self.env_rot_angle), np.cos(self.env_rot_angle)
        self.env_rot_matrix = np.array([[c, -s], [s, c]])

        # agent and flag geometries
        self.flag_homes = flag_homes
        self.flag_homes_unit = flag_homes_unit
        self.agent_radius = agent_radius
        self.flag_radius = flag_radius
        self.flag_keepout_radius = flag_keepout_radius
        self.catch_radius = catch_radius
        self.slip_radius = slip_radius

        if self.lidar_obs:
            self.lidar_range = lidar_range

        # on sides
        scrim2blue = self.flag_homes[Team.BLUE_TEAM] - scrimmage_coords[0]
        scrim2red = self.flag_homes[Team.RED_TEAM] - scrimmage_coords[0]

        self._on_sides_sign = {
            Team.BLUE_TEAM: np.sign(np.cross(self.scrimmage_vec, scrim2blue)),
            Team.RED_TEAM: np.sign(np.cross(self.scrimmage_vec, scrim2red))
        }

        # flag bisection check
        if self._on_sides_sign[Team.BLUE_TEAM] == self._on_sides_sign[Team.RED_TEAM]:
            raise Exception(
                "The specified flag locations and scrimmage line coordinates are not valid because the scrimmage line does not divide the flag locations"
            )

        closest_point_blue_flag_to_scrimmage_line = closest_point_on_line(
            self.scrimmage_coords[0],
            self.scrimmage_coords[1],
            self.flag_homes[Team.BLUE_TEAM]
        )
        closest_point_red_flag_to_scrimmage_line = closest_point_on_line(
            self.scrimmage_coords[0],
            self.scrimmage_coords[1],
            self.flag_homes[Team.RED_TEAM]
        )

        dist_blue_flag_to_scrimmage_line = np.linalg.norm(closest_point_blue_flag_to_scrimmage_line - self.flag_homes[Team.BLUE_TEAM])
        dist_red_flag_to_scrimmage_line = np.linalg.norm(closest_point_red_flag_to_scrimmage_line - self.flag_homes[Team.RED_TEAM])

        if dist_blue_flag_to_scrimmage_line < LINE_INTERSECT_TOL:
            raise Exception("The blue flag is too close to the scrimmage line.")
        elif dist_red_flag_to_scrimmage_line < LINE_INTERSECT_TOL:
            raise Exception("The red flag is too close to the scrimmage line.")

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

    def _get_topo_geom(self):
        ### Environment Map Retrieval and Caching ###
        map_caching_dir = str(pathlib.Path(__file__).resolve().parents[1] / "__mapcache__")
        if not os.path.isdir(map_caching_dir):
            os.mkdir(map_caching_dir)

        lon1, lat1 = np.round(_sm2ll(*self.env_bounds[0]), 7)
        lon2, lat2 = np.round(_sm2ll(*self.env_bounds[1]), 7)

        map_cache_path = os.path.join(map_caching_dir, f"tile@(({lat1},{lon1}), ({lat2},{lon2})).pkl")

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

            # topographical tile (for building geometries)
            if self.env_bounds[0][0] < self.env_bounds[1][0]:
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
                topo_img = crop_tiles(
                    topo_tile[:, :, :-1],
                    topo_ext, *self.env_bounds.flatten(),
                    ll=False
                )
            else:
                #tile 1
                topo_tile_1, topo_ext_1 = cx.bounds2img(
                    w=self.env_bounds[0][0],
                    s=self.env_bounds[0][1],
                    e=EPSG_3857_EXT_X, #180 longitude from the west
                    n=self.env_bounds[1][1],
                    zoom="auto",
                    source=topo_tile_source,
                    ll=False,
                    wait=0,
                    max_retries=2,
                    n_connections=1,
                    use_cache=False,
                    zoom_adjust=None,
                )
                topo_img_1 = crop_tiles(
                    topo_tile_1[:, :, :-1],
                    topo_ext_1,
                    w=self.env_bounds[0][0],
                    s=self.env_bounds[0][1],
                    e=EPSG_3857_EXT_X, #180 longitude from the west
                    n=self.env_bounds[1][1],
                    ll=False
                )

                #tile 2
                topo_tile_2, topo_ext_2 = cx.bounds2img(
                    w=-EPSG_3857_EXT_X, #180 longitude from the east
                    s=self.env_bounds[0][1],
                    e=self.env_bounds[1][0],
                    n=self.env_bounds[1][1],
                    zoom="auto",
                    source=topo_tile_source,
                    ll=False,
                    wait=0,
                    max_retries=2,
                    n_connections=1,
                    use_cache=False,
                    zoom_adjust=None,
                )
                topo_img_2 = crop_tiles(
                    topo_tile_2[:, :, :-1],
                    topo_ext_2,
                    w=-EPSG_3857_EXT_X, #180 longitude from the east
                    s=self.env_bounds[0][1],
                    e=self.env_bounds[1][0],
                    n=self.env_bounds[1][1],
                    ll=False
                )

                #combine tiles to cross 180 longitude
                topo_img = np.hstack((topo_img_1, topo_img_2))

            # rendering tile (for pygame background)
            render_tile_bounds = wrap_mercator_x(
                self.env_bounds + (self.arena_buffer / self.pixel_size) * np.array([[-1], [1]])
            )

            if render_tile_bounds[0][0] < render_tile_bounds[1][0]:
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
                self.background_img = crop_tiles(
                    render_tile[:, :, :-1],
                    render_ext,
                    *render_tile_bounds.flatten(),
                    ll=False,
                )
            else:
                #tile 1
                render_tile_1, render_ext_1 = cx.bounds2img(
                    w=render_tile_bounds[0][0],
                    s=render_tile_bounds[0][1],
                    e=EPSG_3857_EXT_X, #180 longitude from the west
                    n=render_tile_bounds[1][1],
                    zoom="auto",
                    source=render_tile_source,
                    ll=False,
                    wait=0,
                    max_retries=2,
                    n_connections=1,
                    use_cache=False,
                    zoom_adjust=None,
                )
                background_img_1 = crop_tiles(
                    render_tile_1[:, :, :-1],
                    render_ext_1,
                    w=render_tile_bounds[0][0],
                    s=render_tile_bounds[0][1],
                    e=EPSG_3857_EXT_X, #180 longitude from the west
                    n=render_tile_bounds[1][1],
                    ll=False
                )

                #tile 2
                render_tile_2, render_ext_2 = cx.bounds2img(
                    w=-EPSG_3857_EXT_X, #180 longitude from the east
                    s=render_tile_bounds[0][1],
                    e=render_tile_bounds[1][0],
                    n=render_tile_bounds[1][1],
                    zoom="auto",
                    source=render_tile_source,
                    ll=False,
                    wait=0,
                    max_retries=2,
                    n_connections=1,
                    use_cache=False,
                    zoom_adjust=None,
                )
                background_img_2 = crop_tiles(
                    render_tile_2[:, :, :-1],
                    render_ext_2,
                    w=-EPSG_3857_EXT_X, #180 longitude from the east
                    s=render_tile_bounds[0][1],
                    e=render_tile_bounds[1][0],
                    n=render_tile_bounds[1][1],
                    ll=False
                )

                #combine tiles to cross 180 longitude
                self.background_img = np.hstack((background_img_1, background_img_2))

            # cache maps
            map_cache = {
                "topographical_image": topo_img,
                "render_image": self.background_img,
                "attribution": self.background_img_attribution,
            }
            with open(map_cache_path, "wb") as f:
                pickle.dump(map_cache, f)

        ### Topology Construction ###
        #TODO: remove start poses that happen to be trapped (maybe do this by using contours with buffer pixel size rounded up based on agent radius)
        #mask by water color on topo image
        flag_homes = np.asarray([flag.home for flag in self.flags])
        flag_water_xs, flag_water_ys = flag_homes[:, 0], flag_homes[:, 1]

        flag_water_xs = np.clip(
            np.floor(topo_img.shape[1] * (flag_water_xs / self.env_size[0])),
            None,
            topo_img.shape[1] - 1
        ).astype(int)

        flag_water_ys = np.clip(
            np.floor(topo_img.shape[0] * (1 - flag_water_ys / self.env_size[1])),
            None,
            topo_img.shape[0] - 1
        ).astype(int)

        flag_water_pixel_colors = topo_img[flag_water_ys, flag_water_xs]
        for flag_water_pixel_color in flag_water_pixel_colors: 
            if not (
                np.all(flag_water_pixel_color == 38) or #DO NOT CHANGE (specific to CartoDB.DarkMatterNoLabels)!
                np.all(flag_water_pixel_color == 39) or #DO NOT CHANGE (specific to CartoDB.DarkMatterNoLabels)!
                np.all(flag_water_pixel_color == 40)    #DO NOT CHANGE (specific to CartoDB.DarkMatterNoLabels)! 
            ):
                raise Exception(
                    f"One of the flags ({flag_homes}) is not in the the water."
                )

        mask = (
            np.all(topo_img == 38, axis=-1) | #DO NOT CHANGE (specific to CartoDB.DarkMatterNoLabels)!
            np.all(topo_img == 39, axis=-1) | #DO NOT CHANGE (specific to CartoDB.DarkMatterNoLabels)!
            np.all(topo_img == 40, axis=-1)   #DO NOT CHANGE (specific to CartoDB.DarkMatterNoLabels)!
        ) 
        water_connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled_mask, _ = label(mask, structure=water_connectivity)

        water_pixel_x, water_pixel_y  = flag_water_xs[0], flag_water_ys[0] #assume flag is in correct body of water
        target_label = labeled_mask[water_pixel_y, water_pixel_x]
        land_mask = labeled_mask == target_label

        # water contours
        land_mask_binary = 255 * land_mask.astype(np.uint8)
        water_contours, _ = cv2.findContours(land_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # https://docs.opencv.org/4.10.0/d4/d73/tutorial_py_contours_begin.html
        # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html

        border_contour = max(water_contours, key=cv2.contourArea)
        # TODO: check if this is just the environment bounds, then non-convex approximation will go to the largest island
        border_land_mask = cv2.drawContours(np.zeros_like(land_mask_binary), [border_contour], -1, 255, -1)

        # island contours
        water_mask = np.logical_not(land_mask)
        island_binary = 255 * (border_land_mask * water_mask).astype(np.uint8)
        island_contours, _ = cv2.findContours(island_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # approximate outer contour (border land)
        eps = self.topo_contour_eps * cv2.arcLength(border_contour, True)
        border_cnt_approx = cv2.approxPolyDP(border_contour, eps, True)

        border_land_mask_approx = cv2.drawContours(np.zeros_like(land_mask_binary), [border_cnt_approx], -1, 255, -1)
        border_land_mask_approx = cv2.drawContours(border_land_mask_approx, [border_cnt_approx], -1, 0, 0)

        labeled_border_land_mask_approx, _ = label(border_land_mask_approx, structure=water_connectivity)
        target_water_label = labeled_border_land_mask_approx[water_pixel_y, water_pixel_x]
        border_land_mask_approx = labeled_border_land_mask_approx == target_water_label

        # approximate island contours
        island_cnts_approx = []
        for i, cnt in enumerate(island_contours):
            if cnt.shape[0] != 1:
                eps = self.topo_contour_eps * cv2.arcLength(cnt, True)
                cnt_approx = cv2.approxPolyDP(cnt, eps, True)
                cvx_hull = cv2.convexHull(cnt_approx)
                island_cnts_approx.append(cvx_hull)

        island_mask_approx = cv2.drawContours(255 * np.ones_like(island_binary), island_cnts_approx, -1, 0, -1)  # convex island masks

        # final approximate land mask
        land_mask_approx = border_land_mask_approx * island_mask_approx / 255

        # squeeze contours
        border_cnt = self._img2env_coords(border_cnt_approx.squeeze(), topo_img.shape)
        island_cnts = [self._img2env_coords(cnt.squeeze(), topo_img.shape) for cnt in island_cnts_approx]

        return border_cnt, island_cnts, land_mask_approx

    def _img2env_coords(self, cnt, image_shape):
        cnt = cnt.astype(float) # convert contour array to float64 so as not to lose precision
        cnt[:, 0] = self.env_size[0] * cnt[:, 0] / (image_shape[1] - 1)
        cnt[:, 1] = self.env_size[1] * (1 - cnt[:, 1] / (image_shape[0] - 1))

        return cnt

    def _border_contour_to_border_obstacles(self, border_cnt):
        border_pt_inds = np.where(self._point_on_border(border_cnt))[0]

        if len(border_pt_inds) == 0:
            return [border_cnt]
        else:
            border_cnt = np.roll(border_cnt, -(border_pt_inds[0] + 1), axis=0)

            obstacles = []
            current_obstacle = [border_cnt[-1]]
            n_obstacle_border_pts = 1
            for i, p in enumerate(border_cnt):
                current_obstacle.append(p)

                n_obstacle_border_pts += self._point_on_border(p)
                if n_obstacle_border_pts == 2:
                    if len(current_obstacle) > 2:
                        # contour start wall
                        cnt_start_borders = self._point_on_which_border(current_obstacle[0])
                        if len(cnt_start_borders) == 2:
                            if 3 in cnt_start_borders and 0 in cnt_start_borders:
                                #wall 0 comes after wall 3 (moving counterclockwise)
                                #(see _point_on_which_border() doc string)
                                cnt_start_border = 0
                            else:
                                cnt_start_border = max(cnt_start_borders)
                        else:
                            cnt_start_border = cnt_start_borders[0]

                        # contour end wall
                        cnt_end_borders = self._point_on_which_border(current_obstacle[-1])
                        if len(cnt_end_borders) == 2:
                            if 3 in cnt_end_borders and 0 in cnt_end_borders:
                                #wall 0 comes after wall 3 (moving counterclockwise)
                                #(see _point_on_which_border() doc string)
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
                                current_obstacle.insert(0, self.env_corners[j])

                        obstacles.append(np.array(current_obstacle))

                    #next border contour
                    current_obstacle = [p]
                    n_obstacle_border_pts = 1

            return obstacles

    def _point_on_border(self, p):
        """p can be a single point or multiple points"""
        p = self._standard_pos(p)
        return np.any(p == 0, axis=-1) | np.any(p == self.env_size, axis=-1)

    def _segment_on_border(self, segment):
        p1_borders, p2_borders = self._point_on_which_border(segment)
        same_border = (len(np.intersect1d(p1_borders, p2_borders, assume_unique=True)) > 0)

        return same_border

    def _generate_segments_from_obstacles(self, obstacle, n_quad_segs):
        if isinstance(obstacle, PolygonObstacle):
            vertices = obstacle.anchor_points
        else:  # CircleObstacle
            radius = obstacle.radius
            center = obstacle.center_point
            vertices = list(Point(*center).buffer(radius, quad_segs=n_quad_segs).exterior.coords)[:-1]

        segments = [[vertex, vertices[(i + 1) % len(vertices)]] for i, vertex in enumerate(vertices)]

        return np.asarray(segments)

    def _generate_valid_start_poses(self, land_mask):
        # Conversion factor from mask pixels to environment coordinates
        x_scale = self.env_size[0] / land_mask.shape[1]
        y_scale = self.env_size[1] / land_mask.shape[0]

        # Get coordinates of water pixels in environment units
        water_coords = np.flip(np.column_stack(np.where(land_mask)), axis=-1)
        water_coords[:, 1] = (land_mask.shape[0] - water_coords[:, 1])  # adjust y-coords to go from bottom to top
        water_coords_env = (water_coords + 0.5) * [x_scale, y_scale]

        # Create a list of valid positions
        poses_in_collision = detect_collision(
            water_coords_env,
            max(self.agent_radius),
            self.obstacle_geoms
        )
        valid_init_poses = water_coords_env[np.where(np.logical_not(poses_in_collision))[0]]
        self.valid_init_poses = valid_init_poses[np.where(
            np.all(
                (max(self.agent_radius) < valid_init_poses) & (valid_init_poses < (self.env_size - max(self.agent_radius))), #on-sides
                axis=-1
            )
        )[0]]

        # Create lists of team-specific on-side init positions
        self.valid_team_init_poses = []
        for team in self.agents_of_team:
            self.valid_team_init_poses.append(
                self.valid_init_poses[
                    np.where(self._check_on_sides(self.valid_init_poses, team))[0]
                ]
            )

    def render(self):
        """
        Overridden method inherited from `Gym`.

        Draws all players/flags/etc on the pygame screen.
        """
        # Create screen
        if self.screen is None:
            pygame.init()

            if self.render_mode:
                self.agent_font = pygame.font.SysFont(None, int(2 * min(self.agent_render_radius)))

                if self.render_mode == "human":
                    pygame.display.set_caption("Capture The Flag")
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                elif self.render_mode == "rgb_array":
                    self.screen = pygame.Surface((self.screen_width, self.screen_height))
                else:
                    raise ValueError(
                        f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
                    )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return

        # Background
        self.screen.blit(self.pygame_background_img, (0, 0))

        # Flags
        for team in self.agents_of_team:
            team_idx = int(team)
            flag = self.flags[team_idx]
            color = "blue" if team == Team.BLUE_TEAM else "red"

            # team flag (not picked up)
            if not self.flags[team_idx].taken:
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
                    radius=(self.flag_keepout_radius) * self.pixel_size,
                    width=self.boundary_width,
                )

            # team home region
            home_center_screen = self.env_to_screen(self.flags[team_idx].home)
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
                for i in range(1, len(self.traj_render_buffer[self.agents[0]]["traj"]) + 1):
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
                for i in range(1, len(self.traj_render_buffer[self.agents[0]]["agent"]) + 1):
                    for agent_id in self.players:
                        prev_rot_blit_pos, prev_agent_surf = self.traj_render_buffer[agent_id]["agent"][-i]
                        prev_agent_surf.set_alpha(self.render_transparency_alpha)
                        self.screen.blit(prev_agent_surf, prev_rot_blit_pos)
            # history
            elif self.render_traj_mode.endswith("history"):
                for i in reversed(self.obs_hist_buffer_inds[1:] - 1): #current state of agent is not included in history buffer
                    for agent_id in self.players:
                        if i < len(self.traj_render_buffer[agent_id]["history"]):
                            prev_rot_blit_pos, prev_agent_surf = self.traj_render_buffer[agent_id]["history"][i]
                            render_tranparency = 255 - ((255 - self.render_transparency_alpha) * (i+1) / (self.obs_hist_buffer_len-1))
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
                    ray_headings_global = np.deg2rad((heading_angle_conversion(player.heading) + self.lidar_ray_headings) % 360)
                    ray_vecs = np.array([np.cos(ray_headings_global), np.sin(ray_headings_global)]).T
                    lidar_starts = player.pos + self.agent_radius[player.idx] * ray_vecs
                    for i in range(self.num_lidar_rays):
                        if (
                            self.render_lidar_mode == "full" or
                            (
                                self.render_lidar_mode == "detection" and
                                self.state["lidar_labels"][player.id][i] != self.ray_int_label_map["nothing"]
                            )
                        ):
                            draw.line(
                                self.screen,
                                color,
                                self.env_to_screen(lidar_starts[i]),
                                self.env_to_screen(self.state["lidar_ends"][player.id][i]),
                                width=1,
                            )
                #tagging
                player.render_tagging_oob(self.tagging_cooldown)

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
                        radius=0.55 * self.agent_render_radius[player.idx],
                    )

                # agent id
                if self.render_ids:
                    if self.gps_env:
                        font_color = "white"
                    else:
                        font_color = "white" if team == Team.BLUE_TEAM else "black"

                    player_number_label = self.agent_font.render(str(player.idx), True, font_color)
                    player_number_label_rect = player_number_label.get_rect()
                    player_number_label_rect.center = (0.5 * rotated_surface_size[0], 0.52 * rotated_surface_size[1]) # Using 0.52 for the y-coordinate because it looks nicer than 0.5
                    rotated_surface.blit(player_number_label, player_number_label_rect)

                # blit agent onto screen
                self.screen.blit(rotated_surface, rotated_blit_pos)

                # save agent surface for trajectory rendering
                if self.render_traj_mode:
                    # add traj/ agent render data
                    if self.render_traj_mode.startswith("traj"):
                        self.traj_render_buffer[player.id]["traj"].insert(0, blit_pos)

                    if (
                        self.render_traj_mode.endswith("agent") and
                        (self.render_ctr-1) % self.render_traj_freq == 0
                    ):
                        self.traj_render_buffer[player.id]['agent'].insert(0, (rotated_blit_pos, rotated_surface))

                    elif self.render_traj_mode.endswith("history"):
                        self.traj_render_buffer[player.id]["history"].insert(0, (rotated_blit_pos, rotated_surface))

                    # truncate traj
                    if self.render_traj_cutoff is not None:
                        if self.render_traj_mode.startswith("traj"):
                            self.traj_render_buffer[player.id]["traj"] = self.traj_render_buffer[player.id]["traj"][: self.render_traj_cutoff]

                        if self.render_traj_mode.endswith("agent"):
                            agent_render_cutoff = (
                                floor(self.render_traj_cutoff / self.render_traj_freq) +
                                (
                                    (
                                        (self.render_ctr-1) % self.render_traj_freq +
                                        self.render_traj_freq * floor(self.render_traj_cutoff / self.render_traj_freq)
                                    ) <= self.render_traj_cutoff
                                )
                            )
                            self.traj_render_buffer[player.id]["agent"] = self.traj_render_buffer[player.id]["agent"][: agent_render_cutoff]

                    self.traj_render_buffer[player.id]["history"] = self.traj_render_buffer[player.id]["history"][: self.obs_hist_buffer_len]

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
                            hsv_hue = (a2a_dis - self.catch_radius) / (2*self.catch_radius - self.catch_radius)
                            hsv_hue = 0.33 * np.clip(hsv_hue, 0, 1)
                            line_color = tuple(255 * np.asarray(colorsys.hsv_to_rgb(hsv_hue, 0.9, 0.9)))

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
            self.render_buffer.append(
                np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.screen), dtype=np.uint8),
                    axes=(1, 0, 2)
                )
            )

        # Update counter
        self.render_ctr += 1

    def env_to_screen(self, pos):
        screen_pos = self.pixel_size * np.asarray(pos)
        screen_pos[0] += self.arena_buffer[0][0]
        screen_pos[1] = self.arena_height - screen_pos[1] + self.arena_buffer[1][1]

        return screen_pos

    def buffer_to_video(self, recording_compression=False):
        """Convert and save current render buffer as a video"""
        if not self.render_saving:
            print(f"Warning! Environment rendering is disabled. Cannot save the video. See the render_saving option in the config dictionary.")
            print()
        elif self.render_mode is not None:
            if self.render_ctr > 1:
                video_file_dir = str(pathlib.Path(__file__).resolve().parents[1] / 'videos')
                if not os.path.isdir(video_file_dir):
                    os.mkdir(video_file_dir)

                now = datetime.now() #get date and time
                video_id = now.strftime("%m-%d-%Y_%H-%M-%S")
                video_file_name = f"pyquaticus_{video_id}.mp4"
                video_file_path = os.path.join(video_file_dir, video_file_name)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_file_path, fourcc, self.render_fps, (self.screen_width, self.screen_height))
                for img in self.render_buffer:
                    out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                out.release()

                if recording_compression:
                    compressed_video_file_name = f"pyquaticus_{video_id}_compressed.mp4"

                    compressed_video_file_path = os.path.join(video_file_dir, compressed_video_file_name)
                    subprocess.run([
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-i",
                        video_file_path,
                        "-c:v",
                        "libx264",
                        compressed_video_file_path
                    ])
            else:
                print("Attempted to save video but render_buffer is empty!")
                print()
        else:
            raise Exception("Envrionment was not rendered. See the render_mode option in the config dictionary.")

    def save_screenshot(self):
        """ "Save an image of the most recently rendered env."""

        if not self.render_saving:
            print(f"Warning! Environment rendering is disabled. Cannot save the screenshot. See the render_saving option in the config dictionary.")
            print()
        elif self.render_mode is not None:
            image_file_dir = str(pathlib.Path(__file__).resolve().parents[1] / 'screenshots')
            if not os.path.isdir(image_file_dir):
                os.mkdir(image_file_dir)

            now = datetime.now()  # get date and time
            image_id = now.strftime("%m-%d-%Y_%H-%M-%S")
            image_file_name = f"pyquaticus_{image_id}.png"
            image_file_path = os.path.join(image_file_dir, image_file_name)

            cv2.imwrite(image_file_path, cv2.cvtColor(self.render_buffer[self.render_ctr - 1], cv2.COLOR_RGB2BGR))
        else:
            raise Exception("Envrionment was not rendered. See the render_mode option in the config dictionary.")

    def close(self):
        """Close the pygame window, if open."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _min(self, a, b):
        """Convenience method for determining a minimum value. The standard `min()` takes much longer to run."""
        if a < b:
            return a
        else:
            return b

    def state_to_obs(self, env_idxs, agent_id, normalize=True):
        """
        Modified method to convert the state to agent observations. In addition to the
        logic performed in the superclass state_to_obs, this method adds the distance
        and bearing to obstacles into the observation and then performs the
        normalization.

        Args:
            agent_id: The agent who's observation is being generated
            normalize: Flag to normalize the values in the observation
        Returns
            The agent's observation
        """
        obs, _ = super().state_to_obs(env_idxs, agent_id, normalize=False)

        #TODO: fix obstacle obs
        # if not self.lidar_obs:
        #     # Obstacle Distance/Bearing
        #     for i, obstacle in enumerate(
        #         self.state["dist_bearing_to_obstacles"][agent_id]
        #     ):
        #         obs[f"obstacle_{i}_distance"] = obstacle[0]
        #         obs[f"obstacle_{i}_bearing"] = obstacle[1]

        if normalize:
            return self.agent_obs_normalizer.normalized(obs), obs
        else:
            return obs, None

    def multiagent_var(self, val, dtype, name:str):
        """
        val: variable value
        dtype: built-in data type to create multiagent variable with
        name: variable name
        """
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) != len(self.agents):
                raise Exception(f"{name} list incorrect length")
        elif isinstance(val, dict):
            if dtype is dict:
                for agent_id in self.players:
                    if agent_id not in val:
                        raise Exception(f"No value for {agent_id} in {name} {str(type(val))[8:-2]}")
                return val
            else:
                temp_val = [None for _ in self.agents]
                for agent_id, player in self.players.items():
                    if agent_id in val:
                        temp_val[player.idx] = val[agent_id]
                    else:
                        raise Exception(f"No value for {agent_id} in {name} {str(type(val))[8:-2]}")
                val = temp_val
        else:
            if dtype is dict:
                val = {agent_id: val for agent_id in self.players}
                return val
            else:
                val = [val for _ in range(len(self.agents))]

        return np.array(val, dtype=dtype)
