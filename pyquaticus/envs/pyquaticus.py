#DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
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
    POLAR_RADIUS
)
from pyquaticus.structs import CircleObstacle, Flag, PolygonObstacle, RenderingPlayer, Team
from pyquaticus.utils.obs_utils import ObsNormalizer
from pyquaticus.utils.pid import PID
from pyquaticus.utils.utils import (
    angle180,
    clip,
    closest_point_on_line,
    get_rot_angle,
    get_screen_res,
    heading_angle_conversion,
    mag_bearing_to,
    mag_heading_to_vec,
    rc_intersection,
    reflect_vector,
    rot2d,
    vec_to_mag_heading
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
                    speed, heading = self._discrete_action_to_speed_relheading(action_dict[player.id])
                else:
                    #Make point system the same on both blue and red side
                    if player.team == Team.BLUE_TEAM:
                        if 'P' in action_dict[player.id]:
                            action_dict[player.id] = 'S' + action_dict[player.id][1:]
                        elif 'S' in action_dict[player.id]:
                            action_dict[player.id] = 'P' + action_dict[player.id][1:]
                        if 'X' not in action_dict[player.id] and action_dict[player.id] not in ['SC', 'CC', 'PC']:
                            action_dict[player.id] += 'X'
                        elif action_dict[player.id] not in ['SC', 'CC', 'PC']:
                            action_dict[player.id] = action_dict[player.id][:-1]

                    _, heading = mag_bearing_to(player.pos, self.config_dict["aquaticus_field_points"][action_dict[player.id]], player.heading)
                    if -0.3 <= self.get_distance_between_2_points(player.pos, self.config_dict["aquaticus_field_points"][action_dict[player.id]]) <= 0.3: #
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
        return ACTION_MAP[action]

    def _relheading_to_global_heading(self, player_heading, relheading):
        return angle180((player_heading + relheading) % 360)

    def _register_state_elements(self, num_on_team, num_obstacles):
        """Initializes the normalizer."""
        agent_obs_normalizer = ObsNormalizer(False)
        if self.lidar_obs:
            max_bearing = [180]
            max_dist_scrimmage = [self.env_diag]
            max_lidar_dist = self.num_lidar_rays * [self.lidar_range]
            min_dist = [0.0]
            max_bool, min_bool = [1.0], [0.0]
            max_speed, min_speed = [MAX_SPEED], [0.0]
            max_score, min_score = [self.max_score], [0.0]

            agent_obs_normalizer.register("scrimmage_line_bearing", max_bearing)
            agent_obs_normalizer.register("scrimmage_line_distance", max_dist_scrimmage, min_dist)
            agent_obs_normalizer.register("speed", max_speed, min_speed)
            agent_obs_normalizer.register("has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("team_has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("opponent_has_flag", max_bool, min_bool)
            agent_obs_normalizer.register("on_side", max_bool, min_bool)
            agent_obs_normalizer.register("tagging_cooldown", [self.tagging_cooldown], [0.0])
            agent_obs_normalizer.register("is_tagged", max_bool, min_bool)
            agent_obs_normalizer.register("team_score", max_score, min_score)
            agent_obs_normalizer.register("opponent_score", max_score, min_score)
            agent_obs_normalizer.register("ray_distances", max_lidar_dist)
            agent_obs_normalizer.register("ray_labels", self.num_lidar_rays * [len(LIDAR_DETECTION_CLASS_MAP) - 1])
        else:
            max_bearing = [180]
            max_dist = [self.env_diag + 10]  # add a ten meter buffer #TODO: convert to web_mercator if gps_env
            max_dist_scrimmage = [self.env_diag]
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
            agent_obs_normalizer.register("scrimmage_line_distance", max_dist_scrimmage, min_dist)
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
                agent_obs_normalizer.register(
                    f"obstacle_{i}_bearing", max_bearing
                )

        self._state_elements_initialized = True
        return agent_obs_normalizer

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
        if not hasattr(self, '_state_elements_initialized') or not self._state_elements_initialized:
            raise RuntimeError("Have not registered state elements")

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
                obs["team_score"] = self.game_score["blue_captures"]
                obs["opponent_score"] = self.game_score["red_captures"]
            else:
                obs["team_score"] = self.game_score["red_captures"]
                obs["opponent_score"] = self.game_score["blue_captures"]

            # Lidar
            obs["ray_distances"] = self.state["lidar_distances"][agent_id]
            obs["ray_labels"] = self.obj_ray_detection_states[own_team][self.state["lidar_labels"][agent_id]]

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
                wall_closest_point = closest_point_on_line(
                    wall[0], wall[1], np_pos
                )
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
            #Is tagged
            obs["is_tagged"] = agent.is_tagged

            #Team score and Opponent score
            if agent.team == Team.BLUE_TEAM:
                obs["team_score"] = self.game_score["blue_captures"]
                obs["opponent_score"] = self.game_score["red_captures"]
            else:
                obs["team_score"] = self.game_score["red_captures"]
                obs["opponent_score"] = self.game_score["blue_captures"]

            # Relative observations to other agents
            # teammates first
            # TODO: consider sorting these by some metric
            #       in an attempt to get permutation invariance
            #       distance or maybe flag status (or some combination?)
            #       i.e. sorted by perceived relevance
            for team in [own_team, other_team]:
                dif_agents = filter(lambda a: a.id != agent.id, self.agents_of_team[team])
                for i, dif_agent in enumerate(dif_agents):
                    entry_name = f"teammate_{i}" if team == own_team else f"opponent_{i}"

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
        return Discrete(len(ACTION_MAP))

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

        Note that for the other team, the walls will be rotated such that the
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

        def dist_from_wall(flag_pos, wall):
            pt = closest_point_on_line(wall[0], wall[1], flag_pos)
            dist, _ = mag_bearing_to(flag_pos, pt)
            return dist

        # short walls are at index 1 and 3
        blue_flag = self.flags[int(Team.BLUE_TEAM)].home
        red_flag  = self.flags[int(Team.RED_TEAM)].home
        self._walls = {}
        if dist_from_wall(blue_flag, all_walls[1]) < dist_from_wall(blue_flag, all_walls[3]):
            self._walls[int(Team.BLUE_TEAM)] = all_walls
            self._walls[int(Team.RED_TEAM)] = rotate_walls(all_walls, 2)
        else:
            assert dist_from_wall(red_flag, all_walls[1]) < dist_from_wall(red_flag, all_walls[3])
            self._walls[int(Team.RED_TEAM)] = all_walls
            self._walls[int(Team.BLUE_TEAM)] = rotate_walls(all_walls, 2)




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
        render_mode: Optional[str] = None
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
        self.game_score = {'blue_captures':0, 'blue_tags':0, 'blue_grabs':0, 'red_captures':0, 'red_tags':0, 'red_grabs':0}
        #blue_captures: number of times the blue team has grabbed (picked up) red's flag and brought it back to the blue side
        #blue_tags: number of times the blue team successfully tagged an opponent
        #blue_grabs: number of times the blue team grabbed (picked up) the opponents flag
        #red_captures: number of times the red team has grabbed blue's flag and brought it back to the red side
        #red_tags: number of times the blue team successfully tagged an opponent
        #red_grabs: number of times the blue team grabbed (picked up) the opponents flag

        self.seed()

        # Set variables from config
        self.set_config_values(config_dict)

        # Create players, use IDs from [0, (2 * team size) - 1] so their IDs can also be used as indices.
        b_players = []
        r_players = []

        for i in range(0, self.num_blue):
            b_players.append(
                RenderingPlayer(i, Team.BLUE_TEAM, (self.agent_radius * self.pixel_size), render_mode)
            )
        for i in range(self.num_blue, self.num_blue + self.num_red):
            r_players.append(
                RenderingPlayer(i, Team.RED_TEAM, (self.agent_radius * self.pixel_size), render_mode)
            )

        self.players = {player.id: player for player in itertools.chain(b_players, r_players)} #maps player ids (or names) to player objects
        self.agents = [agent_id for agent_id in self.players]

        # Agents (player objects) of each team
        self.agents_of_team = {Team.BLUE_TEAM: b_players, Team.RED_TEAM: r_players}
        self.agent_ids_of_team = {team: [player.id for player in self.agents_of_team[team]] for team in Team}

        # Mappings from agent ids to team member ids and opponent ids
        self.agent_to_team_ids = {
            agent_id: [p.id for p in self.agents_of_team[player.team]] for agent_id, player in self.players.items()
        }
        self.agent_to_opp_ids = {
            agent_id: [p.id for p in self.agents_of_team[Team(not player.team.value)]] for agent_id, player in self.players.items()
        }

        # Create a PID controller for each agent
        if self.render_mode:
            dt = 1/self.render_fps
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
        self.action_spaces = {
            agent_id: self.get_agent_action_space() for agent_id in self.players
        }

        self.agent_obs_normalizer = self._register_state_elements(team_size, len(self.obstacles))
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

        for player in self.players.values():
            if player.tagging_cooldown != self.tagging_cooldown:
                # player is still under a cooldown from tagging, advance their cooldown timer, clip at the configured tagging cooldown
                player.tagging_cooldown = self._min(
                    (player.tagging_cooldown + self.sim_speedup_factor * self.tau), self.tagging_cooldown
                )

        self.flag_collision_bool = np.zeros(self.num_agents)

        action_dict = self._to_speed_heading(raw_action_dict)
        if self.render_mode:
            for _i in range(self.num_renders_per_step):
                for _j in range(self.sim_speedup_factor):
                    self._move_agents(action_dict, 1/self.render_fps)
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
        self.state["current_time"] = self.current_time

        # agent and flag capture checks and more
        self._check_pickup_flags()
        self._check_agent_captures()
        self._check_flag_captures()
        if not self.teleport_on_tag:
            self._check_untag()
        self._set_dones()
        self._get_dist_to_obstacles()

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
 
                    self.obj_ray_detection_states[team][self.ray_int_label_map[f"agent_{agent_id}"]] = LIDAR_DETECTION_CLASS_MAP[detection_class_name]

        if self.message and self.render_mode:
            print(self.message)

        rewards = {agent_id: self.compute_rewards(agent_id) for agent_id in self.players}
        obs = {agent_id: self.state_to_obs(agent_id, self.normalize) for agent_id in raw_action_dict}
        info = {}

        terminated = False
        truncated = False
        if self.dones["__all__"]:
            if self.dones["blue"] or self.dones["red"]:
                terminated = True
            else:
                truncated = True
        terminated = {agent: terminated for agent in raw_action_dict}
        truncated = {agent: truncated for agent in raw_action_dict}

        return obs, rewards, terminated, truncated, info

    def _move_agents(self, action_dict, dt):
        """Moves agents in the space according to the specified speed/heading in `action_dict`."""
        for i, player in enumerate(self.players.values()):
            pos_x = player.pos[0]
            pos_y = player.pos[1]
            flag_loc = self.flags[int(player.team)].home

            player.on_own_side = self._check_on_sides(player.pos, player.team)

            # convert desired_speed   and  desired_heading to
            #         desired_thrust  and  desired_rudder
            # requested heading is relative so it directly maps to the heading error
            
            if player.is_tagged and not self.teleport_on_tag:
                flag_home = self.flags[int(player.team)].home
                _, heading_error = mag_bearing_to(player.pos, flag_home, player.heading)
                desired_speed = self.config_dict["max_speed"]
            else:
                desired_speed, heading_error = action_dict[player.id]

            # desired heading is relative to current heading
            speed_error = desired_speed - player.speed
            desired_speed = self._pid_controllers[player.id]["speed"](speed_error)
            desired_rudder = self._pid_controllers[player.id]["heading"](heading_error)

            desired_thrust = player.thrust + self.speed_factor * desired_speed

            desired_thrust = clip(desired_thrust, -self.max_thrust, self.max_thrust)
            desired_rudder = clip(desired_rudder, -self.max_rudder, self.max_rudder)

            # propagate vehicle speed
            raw_speed = np.interp(
                desired_thrust, self.thrust_map[0, :], self.thrust_map[1, :]
            )
            new_speed = self._min(
                raw_speed * 1 - ((abs(desired_rudder) / 100) * self.turn_loss),
                self.max_speed,
            )
            if (new_speed - player.speed) / dt > self.max_acc:
                new_speed = new_speed + self.max_acc * dt
            elif (player.speed - new_speed) / dt > self.max_dec:
                new_speed = new_speed - self.max_dec * dt

            # propagate vehicle heading
            raw_d_hdg = desired_rudder * (self.turn_rate / 100) * dt
            thrust_d_hdg = raw_d_hdg * (1 + (abs(desired_thrust) - 50) / 50)
            if desired_thrust < 0:
                thrust_d_hdg = -thrust_d_hdg

            # if not moving, then can't turn
            if (new_speed + player.speed) / 2.0 < 0.5:
                thrust_d_hdg = 0.0
            new_heading = angle180(player.heading + thrust_d_hdg)

            vel = mag_heading_to_vec(new_speed, new_heading)

            # If the player hits a boundary, return them to their original starting position and skip
            # to the next agent.
            player_hit_obstacle = False
            for obstacle in self.obstacles:
                collision = obstacle.detect_collision((pos_x, pos_y), radius = self.agent_radius)
                if collision is True:
                    player_hit_obstacle = True
                    break
            if player_hit_obstacle is True or not (
                (self.agent_radius <= pos_x <= self.env_size[0] - self.agent_radius)
                and (
                    self.agent_radius <= pos_y <= self.env_size[1] - self.agent_radius
                )
            ):
                if player.team == Team.RED_TEAM:
                    self.game_score['blue_tags'] += 1
                else:
                    self.game_score['red_tags'] += 1
                if player.has_flag:
                    # If they have a flag, return the flag to it's home area
                    self.flags[int(not int(player.team))].reset()
                    self.state["flag_taken"][int(not int(player.team))] = 0
                    if player.team == Team.RED_TEAM:
                        self.red_team_flag_pickup = False
                    else:
                        self.blue_team_flag_pickup = False
                self.state["agent_oob"][player.id] = 1
                if self.teleport_on_tag:
                    player.reset()
                else:
                    if self.tag_on_collision:
                        self.state["agent_tagged"][player.id] = 1
                        player.is_tagged = True
                    player.rotate(copy.deepcopy(player.prev_pos))
                continue
            else:
                self.state["agent_oob"][player.id] = 0

            # check if agent is in keepout region
            ag_dis_2_flag = self.get_distance_between_2_points(
                np.asarray([pos_x, pos_y]), np.asarray(flag_loc)
            )
            if (
                ag_dis_2_flag < self.flag_keepout
                and not self.state["flag_taken"][int(player.team)]
                and self.flag_keepout > 0.
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
                vel_ref[1] = 0. # convention is that vector pointing from keepout intersection to flag center is y' axis in new reference frame

                vel = rot2d(vel_ref, crd_ref_angle)
                pos_x = ag_pos[0]
                pos_y = ag_pos[1]

            if new_speed > 0.1:
                # only rely on vel if speed is large enough to recover heading
                new_speed, new_heading = vec_to_mag_heading(vel)
            # propagate vehicle position
            hdg_rad = math.radians(player.heading)
            new_hdg_rad = math.radians(new_heading)
            avg_speed = (new_speed + player.speed) / 2.0
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

            player.prev_pos = player.pos
            player.pos = np.asarray(new_ag_pos)
            player.speed = clip(new_speed, 0.0, self.max_speed)
            player.heading = angle180(new_heading)
            player.thrust = desired_thrust

            self.state["agent_position"][i] = player.prev_pos
            self.state["prev_agent_position"][i] = player.pos
            self.state["agent_spd_hdg"][i] = [player.speed, player.heading]

    def _check_on_sides(self, pos, team):
        scrim2pos = np.asarray(pos) - self.scrimmage_coords[0]
        cp_sign = np.sign(self._cross_product(self.scrimmage_vec, scrim2pos))

        return cp_sign == self.on_sides_sign[team] or cp_sign == 0

    def _update_lidar(self):
        ray_int_segments = np.copy(self.ray_int_segments)

        # Valid flag intersection segments mask
        flag_int_seg_mask = np.ones(len(self.ray_int_seg_labels), dtype=bool)
        flag_seg_inds = self.seg_label_type_to_inds["flag"]
        flag_int_seg_mask[flag_seg_inds] = np.repeat(np.logical_not(self.state["flag_taken"]), self.n_circle_segments)

        # Translate non-static ray intersection geometries (flags and agents)
        ray_int_segments[flag_seg_inds] += np.repeat(
            np.tile(self.state["flag_home"], 2),
            self.n_circle_segments,
            axis=0
        )
        agent_seg_inds = self.seg_label_type_to_inds["agent"]
        ray_int_segments[agent_seg_inds] += np.repeat(
            np.tile(self.state["agent_position"], 2),
            self.n_circle_segments,
            axis=0
        )
        ray_int_segments = ray_int_segments.reshape(1, -1, 4)

        # Agent rays
        ray_origins = np.expand_dims(self.state["agent_position"], axis=1)
        ray_headings_global = np.deg2rad((heading_angle_conversion(self.state["agent_spd_hdg"][:, 1]).reshape(-1, 1) + self.lidar_ray_headings) % 360)
        ray_vecs = np.array([np.cos(ray_headings_global), np.sin(ray_headings_global)]).transpose(1, 2, 0)
        ray_ends = ray_origins + self.lidar_range * ray_vecs
        ray_segments = np.concatenate(
            (np.full(ray_ends.shape, ray_origins), ray_ends),
            axis=-1
        )
        ray_segments = ray_segments.reshape(self.num_agents, -1, 1, 4)

        #compute ray intersections
        x1, y1, x2, y2 = ray_segments[..., 0], ray_segments[..., 1], ray_segments[..., 2], ray_segments[..., 3]
        x3, y3, x4, y4 = ray_int_segments[..., 0], ray_int_segments[..., 1], ray_int_segments[..., 2], ray_int_segments[..., 3]
        
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        intersect_x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        intersect_y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        
        #mask invalid intersections (parallel lines, outside of segment bounds, picked up flags, own agent segments)
        mask = (denom != 0) & \
            (intersect_x >= np.minimum(x1, x2) - LINE_INTERSECT_TOL) & (intersect_x <= np.maximum(x1, x2) + LINE_INTERSECT_TOL) & \
            (intersect_y >= np.minimum(y1, y2) - LINE_INTERSECT_TOL) & (intersect_y <= np.maximum(y1, y2) + LINE_INTERSECT_TOL) & \
            (intersect_x >= np.minimum(x3, x4) - LINE_INTERSECT_TOL) & (intersect_x <= np.maximum(x3, x4) + LINE_INTERSECT_TOL) & \
            (intersect_y >= np.minimum(y3, y4) - LINE_INTERSECT_TOL) & (intersect_y <= np.maximum(y3, y4) + LINE_INTERSECT_TOL) & \
            flag_int_seg_mask & self.agent_int_seg_mask

        print(mask)
        # print(flag_int_seg_mask)
        # print(self.agent_int_seg_mask)
        import sys
        sys.exit()

        intersect_x = np.where(mask, intersect_x, -self.env_diag) #a coordinate out of bounds and far away
        intersect_y = np.where(mask, intersect_y, -self.env_diag) #a coordinate out of bounds and far away
        intersections = np.stack((intersect_x.flatten(), intersect_y.flatten()), axis=-1).reshape(intersect_x.shape + (2,))

        #determine lidar ray readings
        intersection_dists = np.linalg.norm(intersections - ray_origins, axis=-1)
        ray_int_inds = np.argmin(intersection_dists, axis=-1)

        ray_int_labels = self.ray_int_seg_labels[ray_int_inds]
        ray_intersections = intersections[np.arange(intersections.shape[0]), ray_int_inds]
        ray_int_dists = intersection_dists[np.arange(intersection_dists.shape[0]), ray_int_inds]

        #correct lidar ray readings for which nothing was detected
        invalid_ray_ints = np.where(np.all(np.logical_not(mask), axis=-1))[0]
        ray_int_labels[invalid_ray_ints] = self.ray_int_label_map["nothing"]
        ray_intersections[invalid_ray_ints] = ray_ends[invalid_ray_ints]
        ray_int_dists[invalid_ray_ints] = self.lidar_range

        #save lidar readings
        for i, agent_id in enumerate(self.players):
            self.state["lidar_labels"][agent_id] = ray_int_labels
            self.state["lidar_ends"][agent_id] = ray_intersections
            self.state["lidar_distances"][agent_id] = ray_int_dists

    def _check_pickup_flags(self):
        """Updates player states if they picked up the flag."""
        for player in self.players.values():
            team = int(player.team)
            other_team = int(not team)
            if not (player.has_flag or self.state["flag_taken"][other_team]) and (not player.is_tagged):
                flag_pos = self.flags[other_team].pos
                distance_to_flag = self.get_distance_between_2_points(
                    player.pos, flag_pos
                )

                if distance_to_flag < self.catch_radius:
                    player.has_flag = True
                    self.state["flag_taken"][other_team] = 1
                    if player.team == Team.BLUE_TEAM:
                        self.game_score['blue_grabs'] += 1
                        self.blue_team_flag_pickup = True
                    else:
                        self.game_score['red_grabs'] += 1
                        self.red_team_flag_pickup = True
                        break

    def _check_untag(self):
        """Untags the player if they return to their own flag."""
        for player in self.players.values():
            team = int(player.team)
            flag_home = self.flags[team].home
            distance_to_flag = self.get_distance_between_2_points(
                player.pos, flag_home
            )
            if distance_to_flag < self.catch_radius and player.is_tagged:
                self.state["agent_tagged"][player.id] = 0
                player.is_tagged = False

    def _check_agent_captures(self):
        """Updates player states if they tagged another player."""
        # Reset capture state if teleport_on_tag is true
        if self.teleport_on_tag:
            self.state["agent_tagged"] = [0] * self.num_agents
            for player in self.players.values():
                player.is_tagged = False

        self.state["agent_captures"] = [None] * self.num_agents
        for player in self.players.values():
            # Only continue logic check if player tagged someone if it's on its own side and is untagged.
            if player.on_own_side and (
                player.tagging_cooldown == self.tagging_cooldown
            ) and not player.is_tagged:
                for other_player in self.players.values():
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
                                self.game_score['blue_tags'] += 1
                            else:
                                self.game_score['red_tags'] += 1
                            self.state["agent_tagged"][other_player.id] = 1
                            other_player.is_tagged = True
                            self.state["agent_captures"][player.id] = other_player.id
                            # If we get here, then `player` tagged `other_player` and we need to reset `other_player`
                            # Only if config["teleport_on_capture"] == True
                            if self.teleport_on_tag:
                                buffer_sign = (
                                    1.0
                                    if self.flags[o_team].home[0] < self.scrimmage
                                    else -1.0
                                )
                                other_player.pos[0] = self.flags[o_team].home[
                                    0
                                ] + buffer_sign * (self.flag_keepout + 0.1)
                                other_player.pos[1] = self.flags[o_team].home[1]
                                other_player.speed = 0.0
                                other_player.on_own_side = True

                            if other_player.has_flag:
                                # If the player with the flag was tagged, the flag also needs to be reset.
                                if other_player.team == Team.BLUE_TEAM:
                                    self.blue_team_flag_pickup = False
                                else:
                                    self.red_team_flag_pickup = False
                                self.state["flag_taken"][int(player.team)] = 0
                                self.flags[int(player.team)].reset()
                                other_player.has_flag = False

                            # Set players tagging cooldown
                            player.tagging_cooldown = 0.0

                            # Break loop (should not be allowed to tag again during current timestep)
                            break

    def _check_flag_captures(self):
        """Updates states if a player captured a flag."""
        # these are false except at the instance that the flag is captured
        self.blue_team_flag_capture = False
        self.red_team_flag_capture = False
        for player in self.players.values():
            if player.on_own_side and player.has_flag:
                if player.team == Team.BLUE_TEAM:
                    self.blue_team_flag_capture = True
                    self.game_score['blue_captures'] += 1
                    self.blue_team_flag_pickup = False
                else:
                    self.red_team_flag_capture = True
                    self.game_score['red_captures'] += 1
                    self.red_team_flag_pickup = False
                player.has_flag = False
                scored_flag = self.flags[not int(player.team)]
                self.state["flag_taken"][int(scored_flag.team)] = False
                scored_flag.reset()
                break

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
        self.topo_contour_eps = config_dict.get("topo_contour_eps", config_dict_std["topo_contour_eps"])

        # MOOS dynamics parameters
        self.max_speed = config_dict.get("max_speed", config_dict_std["max_speed"])
        self.speed_factor = config_dict.get("speed_factor", config_dict_std["speed_factor"])
        self.thrust_map = config_dict.get("thrust_map", config_dict_std["thrust_map"])
        self.max_thrust = config_dict.get("max_thrust", config_dict_std["max_thrust"])
        self.max_rudder = config_dict.get("max_rudder", config_dict_std["max_rudder"])
        self.turn_loss = config_dict.get("turn_loss", config_dict_std["turn_loss"])
        self.turn_rate = config_dict.get("turn_rate", config_dict_std["turn_rate"])
        self.max_acc = config_dict.get("max_acc", config_dict_std["max_acc"])
        self.max_dec = config_dict.get("max_dec", config_dict_std["max_dec"])

        # Simulation parameters
        self.tau = config_dict.get("tau", config_dict_std["tau"])
        self.sim_speedup_factor = config_dict.get("sim_speedup_factor", config_dict_std["sim_speedup_factor"])

        # Game parameters
        self.max_score = config_dict.get("max_score", config_dict_std["max_score"])
        self.max_time = config_dict.get("max_time", config_dict_std["max_time"])
        self.tagging_cooldown = config_dict.get("tagging_cooldown", config_dict_std["tagging_cooldown"])
        self.teleport_on_tag = config_dict.get("teleport_on_tag", config_dict_std["teleport_on_tag"])
        self.tag_on_collision = config_dict.get("tag_on_collision", config_dict_std["tag_on_collision"])

        # Observation parameters
        self.normalize = config_dict.get("normalize", config_dict_std["normalize"])
        self.lidar_obs = config_dict.get("lidar_obs", config_dict_std["lidar_obs"])
        self.num_lidar_rays = config_dict.get("num_lidar_rays", config_dict_std["num_lidar_rays"])

        # Rendering parameters
        self.render_fps = config_dict.get("render_fps", config_dict_std["render_fps"])
        self.screen_frac = config_dict.get("screen_frac", config_dict_std["screen_frac"])
        self.render_ids = config_dict.get("render_agent_ids", config_dict_std["render_agent_ids"])
        self.render_field_points = config_dict.get("render_field_points", config_dict_std["render_field_points"])
        self.render_traj_mode = config_dict.get("render_traj_mode", config_dict_std["render_traj_mode"])
        self.render_traj_freq = config_dict.get("render_traj_freq", config_dict_std["render_traj_freq"])
        self.render_traj_cutoff = config_dict.get("render_traj_cutoff", config_dict_std["render_traj_cutoff"])
        self.render_lidar = config_dict.get("render_lidar", config_dict_std["render_lidar"])
        self.record_render = config_dict.get("record_render", config_dict_std["record_render"])
        self.recording_format = config_dict.get("recording_format", config_dict_std["recording_format"])

        # Miscellaneous parameters
        if config_dict.get("suppress_numpy_warnings", config_dict_std["suppress_numpy_warnings"]):
            # Suppress numpy warnings to avoid printing out extra stuff to the console
            np.seterr(all="ignore")


        ### Environment Geometry Construction ###
        # Basic environment features
        env_bounds = config_dict.get("env_bounds", config_dict_std["env_bounds"])
        env_bounds_unit = config_dict.get("env_bounds_unit", config_dict_std["env_bounds_unit"])

        flag_homes = {}
        flag_homes[Team.BLUE_TEAM] = config_dict.get("blue_flag_home", config_dict_std["blue_flag_home"])
        flag_homes[Team.RED_TEAM] = config_dict.get("red_flag_home", config_dict_std["red_flag_home"])
        flag_homes_unit = config_dict.get("flag_homes_unit", config_dict_std["flag_homes_unit"])

        scrimmage_coords = config_dict.get("scrimmage_coords", config_dict_std["scrimmage_coords"])
        scrimmage_coords_unit = config_dict.get("scrimmage_coords_unit", config_dict_std["scrimmage_coords_unit"])

        agent_radius = config_dict.get("agent_radius", config_dict_std["agent_radius"])
        flag_radius = config_dict.get("flag_radius", config_dict_std["flag_radius"])
        flag_keepout = config_dict.get("flag_keepout", config_dict_std["flag_keepout"])
        catch_radius = config_dict.get("catch_radius", config_dict_std["catch_radius"])
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
            lidar_range=lidar_range
        )

        # Aquaticus point field
        #TODO

        # Environment corners
        self.env_ll = np.array([0.0, 0.0], dtype=np.float32)
        self.env_lr = np.array([self.env_size[0], 0.0], dtype=np.float32)
        self.env_ul = np.array([0.0, self.env_size[1]], dtype=np.float32)
        self.env_ur = np.array(self.env_size, dtype=np.float32)
        # ll = lower left, lr = lower right
        # ul = upper left, ur = upper right

        ### Environment Rendering ###
        if self.render_mode:
            # pygame orientation vector
            self.PYGAME_UP = Vector2((0.0, 1.0))

            # pygame screen size
            self.arena_buffer_frac = 1/20
            arena_buffer = self.arena_buffer_frac * self.env_diag

            max_screen_size = get_screen_res()
            arena_aspect_ratio = (self.env_size[0] + 2*arena_buffer) / (self.env_size[1] + 2*arena_buffer)
            width_based_height = max_screen_size[0] / arena_aspect_ratio

            if width_based_height <= max_screen_size[1]:
                max_pygame_screen_width = max_screen_size[0]
            else:
                height_based_width = max_screen_size[1] * arena_aspect_ratio
                max_pygame_screen_width = int(height_based_width)

            self.pixel_size = (self.screen_frac * max_pygame_screen_width) / (self.env_size[0] + 2*arena_buffer)
            self.screen_width = round(
                (self.env_size[0] + 2*arena_buffer) * self.pixel_size
            )
            self.screen_height = round(
                (self.env_size[1] + 2*arena_buffer) * self.pixel_size
            )

            # environemnt element sizes in pixels
            self.arena_width, self.arena_height = self.pixel_size * self.env_size
            self.arena_buffer = self.pixel_size * arena_buffer
            self.boundary_width = 2  #pixels
            self.a2a_line_width = 3 #pixels

            # miscellaneous
            self.num_renders_per_step = int(self.render_fps * self.tau)
            self.render_boundary_rect = True #standard rectangular boundary

            # check that world size (pixels) does not exceed the screen dimensions
            world_screen_err_msg = (
                "Specified env_size with arena_buffer ({} pixels) exceeds the maximum size {} in at least one"
                " dimension".format(
                    [
                        round(2*self.arena_buffer + self.pixel_size * self.env_size[0]),
                        round(2*self.arena_buffer + self.pixel_size * self.env_size[1])
                    ], 
                    max_screen_size
                )
            )
            assert (
                2*self.arena_buffer + self.pixel_size * self.env_size[0] <= max_screen_size[0] and
                2*self.arena_buffer + self.pixel_size * self.env_size[1] <= max_screen_size[1]
            ), world_screen_err_msg

            # check that time between frames (1/render_fps) is not larger than timestep (tau)
            frame_rate_err_msg = (
                "Specified frame rate ({}) creates time intervals between frames larger"
                " than specified timestep ({})".format(self.render_fps, self.tau)
            )
            assert 1 / self.render_fps <= self.tau, frame_rate_err_msg

            # check that time warp is an integer >= 1
            if self.sim_speedup_factor < 1:
                print("Warning: sim_speedup_factor must be an integer >= 1! Defaulting to 1.")
                self.sim_speedup_factor = 1

            if type(self.sim_speedup_factor) != int:
                self.sim_speedup_factor = int(np.round(self.sim_speedup_factor))
                print(f"Warning: Converted sim_speedup_factor to integer: {self.sim_speedup_factor}")

    def set_geom_config(self, config_dict):
        self.n_circle_segments = config_dict.get("n_circle_segments", config_dict_std["n_circle_segments"])
        n_quad_segs = round(self.n_circle_segments/4)

        # Obstacles
        obstacle_params = config_dict.get("obstacles", config_dict_std["obstacles"])

        if self.gps_env:
            border_contour, island_contours, land_mask = self._get_topo_geom()

            if border_contour is not None:
                if obstacle_params is None:
                    obstacle_params = {"polygon": []}
                obstacle_params["polygon"].append(border_contour)
                self.render_boundary_rect = False

            if len(island_contours) > 0:
                if obstacle_params is None:
                    obstacle_params = {"polygon": []}
                obstacle_params["polygon"].extend(island_contours)

        self.obstacles = list()
        if obstacle_params is not None and isinstance(obstacle_params, dict):
            circle_obstacles = obstacle_params.get("circle", None)
            if circle_obstacles is not None and isinstance(circle_obstacles, list):
                for param in circle_obstacles:
                    self.obstacles.append(CircleObstacle(param[0], (param[1][0], param[1][1])))
            elif circle_obstacles is not None:
                raise TypeError(f"Expected circle obstacle parameters to be a list of tuples, not {type(circle_obstacles)}")
            poly_obstacle = obstacle_params.get("polygon", None)
            if poly_obstacle is not None and isinstance(poly_obstacle, list):
                for param in poly_obstacle:
                    converted_param = [(p[0], p[1]) for p in param]
                    self.obstacles.append(PolygonObstacle(converted_param))
            elif poly_obstacle is not None:
                raise TypeError(f"Expected polygon obstacle parameters to be a list of tuples, not {type(poly_obstacle)}")
        elif obstacle_params is not None:
            raise TypeError(f"Expected obstacle_params to be None or a dict, not {type(obstacle_params)}")

        # Adjust scrimmage line
        scrim_seg = LineString(self.scrimmage_coords)
        scrim_int_segs = [(p, param[(i+1) % len(param)]) for param in poly_obstacle for i, p in enumerate(param)]
        if border_contour is None:
            scrim_int_segs.extend([
                [self.env_ll, self.env_lr],
                [self.env_lr, self.env_ur],
                [self.env_ur, self.env_ul],
                [self.env_ul, self.env_ll]
            ])

        scrim_ints = []
        for seg in scrim_int_segs:
            seg_int = intersection(scrim_seg, LineString(seg))
            if not seg_int.is_empty:
                scrim_ints.append(seg_int.coords[0])

        scrim_ints = np.asarray(scrim_ints)
        scrim_int_dists = np.linalg.norm(scrim_ints.reshape(-1, 1, 2) - scrim_ints, axis=-1)
        scrim_end_inds = np.unravel_index(np.argmax(scrim_int_dists), scrim_int_dists.shape)
        self.scrimmage_coords = scrim_ints[scrim_end_inds, :]

        # Ray casting
        if self.lidar_obs:
            self.lidar_ray_headings = np.linspace(0, (self.num_lidar_rays - 1) * 360 / self.num_lidar_rays, self.num_lidar_rays)

            ray_int_label_names = ["nothing", "obstacle"]
            ray_int_label_names.extend([f"flag_{i}" for i, _ in enumerate(self.flags)])
            ray_int_label_names.extend([f"agent_{agent_id}" for agent_id in self.agents])
            self.ray_int_label_map = OrderedDict({label_name: i for i, label_name in enumerate(ray_int_label_names)})
            
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
                self.obj_ray_detection_states[team] = np.asarray(self.obj_ray_detection_states[team]) 

            ray_int_segments = []
            ray_int_seg_labels = []
            self.seg_label_type_to_inds = {
                (label[: label.find("_")] if label[-1].isnumeric() else label): [] for label in ray_int_label_names
            }

            #boundary
            if self.gps_env:
                if border_contour is None:
                    ray_int_segments.extend([
                        [*self.env_ll, *self.env_lr],
                        [*self.env_lr, *self.env_ur],
                        [*self.env_ur, *self.env_ul],
                        [*self.env_ul, *self.env_ll]
                    ])
                    ray_int_seg_labels.extend(4 * [self.ray_int_label_map["obstacle"]])
                    self.seg_label_type_to_inds["obstacle"].extend(np.arange(4))
            else:
                ray_int_segments.extend([
                    [*self.env_ll, *self.env_lr],
                    [*self.env_lr, *self.env_ur],
                    [*self.env_ur, *self.env_ul],
                    [*self.env_ul, *self.env_ll]
                ])
                ray_int_seg_labels.extend(4 * [self.ray_int_label_map["obstacle"]])
                self.seg_label_type_to_inds["obstacle"].extend(np.arange(4))

            #obstacles
            obstacle_segments = [segment for obstacle in self.obstacles for segment in self._generate_segments_from_obstacles(obstacle, n_quad_segs)]
            ray_int_seg_labels.extend(
                len(obstacle_segments) * [self.ray_int_label_map["obstacle"]]
            )
            self.seg_label_type_to_inds["obstacle"].extend(
                np.arange(len(ray_int_segments), len(ray_int_segments) + len(obstacle_segments))
            )
            ray_int_segments.extend(obstacle_segments)

            #flags
            for i, _ in enumerate(self.flags):
                vertices = list(Point(0., 0.).buffer(self.flag_radius, quad_segs=n_quad_segs).exterior.coords)[:-1] #approximate circle with an octagon
                segments = [[*vertex, *vertices[(i+1) % len(vertices)]] for i, vertex in enumerate(vertices)]
                ray_int_seg_labels.extend(
                    len(segments) * [self.ray_int_label_map[f"flag_{i}"]]
                )
                self.seg_label_type_to_inds["flag"].extend(
                    np.arange(len(ray_int_segments), len(ray_int_segments) + len(segments))
                )
                ray_int_segments.extend(segments)

            #agents
            for agent_id in self.agents:
                vertices = list(Point(0., 0.).buffer(self.agent_radius, quad_segs=n_quad_segs).exterior.coords)[:-1] #approximate circle with an octagon
                segments = [[*vertex, *vertices[(i+1) % len(vertices)]] for i, vertex in enumerate(vertices)]
                ray_int_seg_labels.extend(
                    len(segments) * [self.ray_int_label_map[f"agent_{agent_id}"]]
                )
                self.seg_label_type_to_inds["agent"].extend(
                    np.arange(len(ray_int_segments), len(ray_int_segments) + len(segments))
                )
                ray_int_segments.extend(segments)

            self.ray_int_segments = np.asarray(ray_int_segments)
            self.ray_int_seg_labels = np.asarray(ray_int_seg_labels)

            #agent ray self intersection mask
            self.agent_int_seg_mask = np.ones((self.num_agents, len(self.ray_int_seg_labels)), dtype=bool)
            agent_seg_inds = self.seg_label_type_to_inds["agent"]

            for i in range(self.num_agents):
                seg_inds_start = i * self.n_circle_segments 
                self.agent_int_seg_mask[i, agent_seg_inds[seg_inds_start: seg_inds_start + self.n_circle_segments]] = False

        # Occupancy map
        #TODO

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
        if self.game_score["red_captures"] == self.max_score:
            self.dones["red"] = True
            self.dones["__all__"] = True
            self.message = "Red Wins! Blue Loses"

        elif self.game_score["blue_captures"] == self.max_score:
            self.dones["blue"] = True
            self.dones["__all__"] = True
            self.message = "Blue Wins! Red Loses"

        elif self.state["current_time"] >= self.max_time:
            self.dones["__all__"] = True
            if self.game_score['blue_captures'] > self.game_score['red_captures']:
                self.message = "Blue Wins! Red Loses"
            elif self.game_score['blue_captures'] > self.game_score['red_captures']:
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
        for i, obstacle in enumerate(self.state["dist_to_obstacles"][agent.id]):
            self.params[agent.id][f"obstacle_{i}_distance"] = obstacle[0]
            self.params[agent.id][f"obstacle_{i}_bearing"] = obstacle[1]

        if agent.team == Team.RED_TEAM:
            # Game Events
            self.params[agent.id]["num_teammates"] = self.num_red
            self.params[agent.id]["num_opponents"] = self.num_blue
            self.params[agent.id]["team_flag_pickup"] = self.red_team_flag_pickup
            self.params[agent.id]["team_flag_capture"] = self.red_team_flag_capture
            self.params[agent.id]["opponent_flag_pickup"] = self.blue_team_flag_pickup
            self.params[agent.id]["opponent_flag_capture"] = self.blue_team_flag_capture
            # Elements
            self.params[agent.id]["team_flag_home"] = self.get_distance_between_2_points(
                    agent.pos, copy.deepcopy(self.state["flag_home"][1])
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
            self.params[agent.id]["team_flag_pickup"] = self.blue_team_flag_pickup
            self.params[agent.id]["team_flag_capture"] = self.blue_team_flag_capture
            self.params[agent.id]["opponent_flag_pickup"] = self.red_team_flag_pickup
            self.params[agent.id]["opponent_flag_capture"] = self.red_team_flag_capture
            # Elements
            self.params[agent.id]["team_flag_home"] = self.get_distance_between_2_points(
                    agent.pos, copy.deepcopy(self.state["flag_home"][0])
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
        self.params[agent.id]["tagging_cooldown"] = (
            not agent.tagging_cooldown >= 10.0
        )
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
        self.params[agent.id]["wall_distances"] =  self._get_dists_to_boundary()[agent.id]
        self.params[agent.id]["agent_captures"] = copy.deepcopy(self.state["agent_captures"])
        self.params[agent.id]["agent_tagged"] = copy.deepcopy(self.state["agent_tagged"])
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
            options: Additonal options for resetting the environment (for now it just contains normalize)
        """
        if seed is not None:
            self.seed(seed=seed)

        if options is not None:
            self.normalize = options.get("normalize", config_dict_std["normalize"])

        if return_info:
            raise DeprecationWarning("return_info has been deprecated by PettingZoo -- https://github.com/Farama-Foundation/PettingZoo/pull/890")

        flag_locations = np.asarray(list(self.flag_homes.values()))

        for flag in self.flags:
            flag.home = flag_locations[int(flag.team)]
            flag.reset()

        self.dones = self._reset_dones()

        agent_positions, agent_spd_hdg, agent_on_sides = self._generate_agent_starts(flag_locations)

        self.state = {
            "agent_position": agent_positions,
            "prev_agent_position": copy.deepcopy(agent_positions),
            "agent_spd_hdg": agent_spd_hdg,
            "agent_has_flag": np.zeros(self.num_agents), #TODO: update during game
            "agent_on_sides": agent_on_sides, #TODO: update during game
            "flag_home": copy.deepcopy(flag_locations),
            "flag_locations": flag_locations, #TODO: update during game
            "flag_taken": np.zeros(len(self.flags)),
            "current_time": 0.0,
            "agent_captures": [
                None
            ] * self.num_agents,  # whether this agent tagged something
            "agent_tagged": [0] * self.num_agents,  # if this agent was tagged
            "agent_oob": [0] * self.num_agents,  # if this agent went out of bounds
            "dist_to_obstacles": dict()
        }
        agent_ids = list(self.players.keys())
        for k in agent_ids:
            self.state["dist_to_obstacles"][k] = [(0, 0)] * len(self.obstacles)

        if self.lidar_obs:
            #reset lidar readings
            self.state["lidar_labels"] = {agent_id: np.zeros(self.num_lidar_rays) for agent_id in agent_ids}
            self.state["lidar_ends"] = {agent_id: np.zeros((self.num_lidar_rays, 2)) for agent_id in agent_ids}
            self.state["lidar_distances"] = {agent_id: np.zeros(self.num_lidar_rays) for agent_id in agent_ids}
            self._update_lidar()

            for team in self.agents_of_team:
                for label_name, label_idx in self.ray_int_label_map.items():
                    if label_name.startswith("agent"):
                        #reset agent lidar detection states
                        if int(label_name[6:]) in self.agent_ids_of_team[team]:
                            self.obj_ray_detection_states[team][label_idx] = LIDAR_DETECTION_CLASS_MAP["teammate"]
                        else:
                            self.obj_ray_detection_states[team][label_idx] = LIDAR_DETECTION_CLASS_MAP["opponent"]

        for k in self.game_score:
            self.game_score[k] = 0

        self._determine_team_wall_orient()

        self.blue_team_flag_pickup = False
        self.red_team_flag_pickup = False
        self.blue_team_score = 0
        self.blue_team_score_prev = 0
        self.red_team_score = 0
        self.red_team_score_prev = 0
        self.message = ""
        self.current_time = 0
        self.reset_count += 1
        reset_obs = {agent_id: self.state_to_obs(agent_id, self.normalize) for agent_id in self.players}

        if self.render_mode:
            self.render_ctr = 0

            if self.record_render:
                #TODO change this to step function
                if len(self.render_buffer) > 0:
                    self.buffer_to_video()
                    self.render_buffer = []
            if self.render_traj_mode:
                self.traj_render_buffer = {agent_id: {'traj': [], 'agent': []} for agent_id in self.players}

            self._render()
            self.render_ctr = 0

        return reset_obs

    def _generate_agent_starts(self, flag_locations):
        """
        Generates starting positions for all players based on flag locations.

        If `random_init` is `True`, then agent positons are generated randomly using their home
        flag and random offsets.

        If `random_init` is `False`, then agent positions are programmatically generated using the formula
        init_x = env_size[x] / 4
        init_y = (env_size[y] / (team_size + 1)) * ((player_id % team_size) + 1)

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

        # if self.random_init:
        if True:
            flags_separation = self.get_distance_between_2_points(
                flag_locations[0], flag_locations[1]
            )

        for player in self.players.values():
            player.is_tagged = False
            player.thrust = 0.0
            player.speed = 0.0
            player.has_flag = False
            player.on_own_side = True
            player.tagging_cooldown = self.tagging_cooldown
            # if self.random_init:
            if True:
                max_agent_separation = flags_separation - 2 * self.flag_keepout

                # starting center point between two agents
                agent_shift = np.random.choice((-1.0, 1.0)) * np.random.uniform(
                    0, 0.5 * (0.5 * max_agent_separation + r) - r
                )

                # adjust agent max and min separation ranges based on shifted center point
                max_agent_separation = 2 * (
                    max_agent_separation / 2 - np.abs(agent_shift)
                )
                min_agent_separation = max(4 * r, 2 * (np.abs(agent_shift) + r))

                # initial agent separation
                np.random.uniform(
                    min_agent_separation, max_agent_separation
                )
                player.orientation = (
                    flag_locations[int(not int(player.team))]
                    - flag_locations[int(player.team)]
                ) / flags_separation
                player.pos = flag_locations[int(player.team)] + player.orientation * (
                    flags_separation / 2
                )

                agent_shift = agent_shift * player.orientation
                player.pos += agent_shift
                player.prev_pos = copy.deepcopy(player.pos)
            else:
                if player.team == Team.RED_TEAM:
                    init_x_pos = self.env_size[0] / 4
                    player.heading = 90
                else:
                    init_x_pos = self.env_size[0] - self.env_size[0] / 4
                    player.heading = -90

                init_y_pos = (self.env_size[1] / (self.team_size + 1)) * (
                    (player.id % self.team_size) + 1
                )
                player.pos = [init_x_pos, init_y_pos]
                player.prev_pos = copy.deepcopy(player.pos)
            player.home = copy.deepcopy(player.pos)
            agent_locations.append(player.pos)
            agent_spd_hdg.append([player.speed, player.heading])
            agent_on_sides.append(True)

        return np.asarray(agent_locations), np.asarray(agent_spd_hdg), np.asarray(agent_on_sides)

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
    
    def _get_dist_to_obstacles(self):
        """Computes the distance from each player to each obstacle"""
        dist_to_obstacles = dict()
        for player in self.players.values():
            player_pos = player.pos
            player_dists_to_obstacles = list()
            for obstacle in self.obstacles:
                dist_to_obstacle = obstacle.distance_from(player_pos, radius = self.agent_radius, heading=player.heading)
                player_dists_to_obstacles.append(dist_to_obstacle)
            dist_to_obstacles[player.id] = player_dists_to_obstacles
        self.state["dist_to_obstacles"] = dist_to_obstacles

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
        lidar_range: float
    ):
        if (
            self._is_auto_string(env_bounds) and 
            (self._is_auto_string(flag_homes[Team.BLUE_TEAM]) or self._is_auto_string(flag_homes[Team.RED_TEAM]))
        ):
            raise Exception("Either env_bounds or blue AND red flag homes must be set in config_dict (cannot both be 'auto')")

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
                    #convert flag poses to web mercator xy
                    flag_home_blue = np.asarray(mt.xy(*flag_home_blue[-1::-1]))
                    flag_home_red = np.asarray(mt.xy(*flag_home_red[-1::-1]))
                    
                flag_vec = flag_home_blue - flag_home_red
                flag_distance = np.linalg.norm(flag_vec)
                flag_unit_vec = flag_vec / flag_distance
                flag_perp_vec = np.array([-flag_unit_vec[1], flag_unit_vec[0]]) 

                #assuming default aquaticus field size ratio drawn on web mercator, these bounds will contain it
                border_pt1 =  flag_home_blue + (flag_distance/6) * flag_unit_vec + (flag_distance/3) * flag_perp_vec
                border_pt2 =  flag_home_blue + (flag_distance/6) * flag_unit_vec + (flag_distance/3) * -flag_perp_vec
                border_pt3 =  flag_home_red + (flag_distance/6) * -flag_unit_vec + (flag_distance/3) * flag_perp_vec
                border_pt4 =  flag_home_red + (flag_distance/6) * -flag_unit_vec + (flag_distance/3) * -flag_perp_vec
                border_points = np.array([border_pt1, border_pt2, border_pt3, border_pt4])

                #environment bounds will be in web mercator xy
                env_bounds = np.zeros((2,2))
                env_bounds[0][0] = np.min(border_points[:, 0])
                env_bounds[0][1] = np.min(border_points[:, 1])
                env_bounds[1][0] = np.max(border_points[:, 0])
                env_bounds[1][1] = np.max(border_points[:, 1])
            else:
                env_bounds = np.asarray(env_bounds)

                if env_bounds_unit == "m":
                    #check for exceptions
                    if (
                        self._is_auto_string(flag_homes[Team.BLUE_TEAM]) or
                        self._is_auto_string(flag_homes[Team.RED_TEAM]) or
                        flag_homes_unit == "m"
                    ):
                        raise Exception(
                            "Flag locations must be specified in aboslute coordinates (lat/long or web mercator xy) \
when gps environment bounds are specified in meters")

                    if len(env_bounds.shape) == 1:
                        env_bounds = np.array([
                            (0., 0.),
                            env_bounds
                        ])
                    if np.any(env_bounds[1] == 0.):
                        raise Exception("Environment max bounds must be > 0 when specified in meters")

                    #get flag midpoint
                    if flag_homes_unit == "wm_xy":
                        flag_home_blue = np.flip(_sm2ll(*flag_homes[Team.BLUE_TEAM]))
                        flag_home_red = np.flip(_sm2ll(*flag_homes[Team.RED_TEAM]))

                    geodict_flags = Geodesic.WGS84.Inverse(
                        lat1=flag_home_blue[0],
                        lon1=flag_home_blue[1],
                        lat2=flag_home_red[0],
                        lon2=flag_home_red[1]
                    )
                    geodict_flag_midpoint = Geodesic.WGS84.Direct(
                        lat1=flag_home_blue[0],
                        lon1=flag_home_blue[1],
                        azi1=geodict_flags['azi1'],
                        s12=geodict_flags['s12']/2
                    )
                    flag_midpoint = (geodict_flag_midpoint['lat2'], geodict_flag_midpoint['lon2'])

                    #vertical bounds
                    env_top = Geodesic.WGS84.Direct(
                        lat1=flag_midpoint[0],
                        lon1=flag_midpoint[1],
                        azi1=0, #degrees
                        s12=0.5*env_bounds[1][1]
                    )['lat2']
                    env_bottom = Geodesic.WGS84.Direct(
                        lat1=flag_midpoint[0],
                        lon1=flag_midpoint[1],
                        azi1=180, #degrees
                        s12=0.5*env_bounds[1][1]
                    )['lat2']

                    #horizontal bounds
                    geoc_lat = np.arctan(
                        (POLAR_RADIUS / EQUATORIAL_RADIUS) * np.tan(flag_midpoint[0])
                    )
                    small_circle_circum = np.pi * 2 * EQUATORIAL_RADIUS * np.cos(geoc_lat)
                    env_left = flag_midpoint[1] - 360 * (0.5*env_bounds[1][0] / small_circle_circum)
                    env_right = flag_midpoint[1] + 360 * (0.5*env_bounds[1][0] / small_circle_circum)

                    env_left = angle180(env_left)
                    env_right = angle180(env_right)

                    #convert bounds to web mercator xy
                    env_bounds = np.array([
                        mt.xy(env_left, env_bottom),
                        mt.xy(env_right, env_top)
                    ])
                elif env_bounds_unit == "ll":
                    #convert bounds to web mercator xy
                    wm_xy_bounds = np.array([
                        mt.xy(*env_bounds[0][-1::-1]),
                        mt.xy(*env_bounds[1][-1::-1])
                    ])
                    left = np.min(wm_xy_bounds[:, 0])
                    bottom = np.min(wm_xy_bounds[:, 1])
                    right = np.max(wm_xy_bounds[:, 0])
                    top = np.max(wm_xy_bounds[:, 1])
                    env_bounds = np.array([
                        [left, bottom],
                        [right, top]
                    ])
                else: #web mercator xy
                    left = np.min(env_bounds[:, 0])
                    bottom = np.min(env_bounds[:, 1])
                    right = np.max(env_bounds[:, 0])
                    top = np.max(env_bounds[:, 1])
                    env_bounds = np.array([
                        [left, bottom],
                        [right, top]
                    ])
            #unit
            env_bounds_unit = "wm_xy"

            #environment size
            self.env_size = np.diff(env_bounds, axis=0)[0]
            self.env_diag = np.linalg.norm(self.env_size)

            #vertices
            self.env_bounds_vertices = np.array([
                env_bounds[0],
                (env_bounds[1][0], env_bounds[0][1]),
                env_bounds[1],
                (env_bounds[0][0], env_bounds[1][1])
            ])

            ### flags home ###
            #auto home
            if self._is_auto_string(flag_homes[Team.BLUE_TEAM]) and self._is_auto_string(flag_homes[Team.RED_TEAM]):
                flag_homes[Team.BLUE_TEAM] = env_bounds[0] + np.array([7/8*self.env_size[0], 0.5*self.env_size[0]])
                flag_homes[Team.RED_TEAM] = env_bounds[0] + np.array([1/8*self.env_size[0], 0.5*self.env_size[0]])
            elif self._is_auto_string(flag_homes[Team.BLUE_TEAM]) or self._is_auto_string(flag_homes[Team.RED_TEAM]):
                raise Exception("Flag homes should be either all 'auto', or all specified")
            else:
                if flag_homes_unit == "m":
                    raise Exception("'m' (meters) should only be used to specify flag homes when gps_env is False")

                flag_homes[Team.BLUE_TEAM] = np.asarray(flag_homes[Team.BLUE_TEAM])
                flag_homes[Team.RED_TEAM] = np.asarray(flag_homes[Team.RED_TEAM])

                if flag_homes_unit == "ll":
                    #convert flag poses to web mercator xy
                    flag_homes[Team.BLUE_TEAM] = mt.xy(*flag_homes[Team.BLUE_TEAM][-1::-1])
                    flag_homes[Team.RED_TEAM] = mt.xy(*flag_homes[Team.RED_TEAM][-1::-1])

            #blue flag
            if (
                np.any(flag_homes[Team.BLUE_TEAM] <= env_bounds[0]) or
                np.any(flag_homes[Team.BLUE_TEAM] >= env_bounds[1])
            ):
                raise Exception(f"Blue flag home {flag_homes[Team.BLUE_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}")

            #red flag
            if (
                np.any(flag_homes[Team.RED_TEAM] <= env_bounds[0]) or
                np.any(flag_homes[Team.RED_TEAM] >= env_bounds[1])
            ):
                raise Exception(f"Red flag home {flag_homes[Team.RED_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}")

            #normalize relative to environment bounds
            flag_homes[Team.BLUE_TEAM] -= env_bounds[0]
            flag_homes[Team.RED_TEAM] -= env_bounds[0]

            #unit
            flag_homes_unit = "wm_xy"

            ### scrimmage line ###
            if self._is_auto_string(scrimmage_coords):
                flags_vec = flag_homes[Team.BLUE_TEAM] - flag_homes[Team.RED_TEAM]

                scrim_vec1 = np.array([-flags_vec[1], flags_vec[0]])
                scrim_vec2 = np.array([flags_vec[1], -flags_vec[0]])
                flags_midpoint = 0.5 * (flag_homes[Team.BLUE_TEAM] + flag_homes[Team.RED_TEAM]) + env_bounds[0]

                scrimmage_coord1 = self._get_polygon_intersection(flags_midpoint, scrim_vec1, self.env_bounds_vertices)[1]
                scrimmage_coord2 = self._get_polygon_intersection(flags_midpoint, scrim_vec2, self.env_bounds_vertices)[1]
                scrimmage_coords = np.asarray([scrimmage_coord1, scrimmage_coord2]) - env_bounds[0]
            else:
                if scrimmage_coords_unit == "m":
                    raise Exception("'m' (meters) should only be used to specify flag homes when gps_env is False")
                raise NotImplementedError("TODO: non-auto scrimmage line for gps environment")
                #TODO

            #unit
            scrimmage_coords_unit = "wm_xy"

            #TODO, check that flags do not fall on scrimamge line

            ### agent and flag geometries ###
            lon1, lat1 = _sm2ll(*env_bounds[0])
            lon2, lat2 = _sm2ll(*env_bounds[1])
            lon_diff = self._longitude_diff_west2east(lon1, lon2)

            if np.abs(lat1) > np.abs(lat2):
                lat = lat1
            else:
                lat = lat2

            geoc_lat = np.arctan((POLAR_RADIUS / EQUATORIAL_RADIUS) * np.tan(lat))
            small_circle_circum = np.pi * 2 * EQUATORIAL_RADIUS * np.cos(geoc_lat)
            
            #use most warped (squished) horizontal border to underestimate the number of
            #meters per mercator xy, therefore overestimate how close objects are to one another
            self.meters_per_mercator_xy = small_circle_circum * (lon_diff/360) / self.env_size[0]
            agent_radius /= self.meters_per_mercator_xy
            flag_radius /= self.meters_per_mercator_xy
            catch_radius /= self.meters_per_mercator_xy
            flag_keepout /= self.meters_per_mercator_xy
            lidar_range /= self.meters_per_mercator_xy

        else:
            ### environment bounds ###
            if env_bounds_unit != "m":
                raise Exception("Environment bounds unit must be meters ('m') when gps_env is False")

            if self._is_auto_string(env_bounds):
                if np.any(np.sign([flag_homes[Team.BLUE_TEAM], flag_homes[Team.RED_TEAM]]) == -1):
                    raise Exception("Flag coordinates must be in the positive quadrant when gps_env is False")

                if np.any(np.sign([flag_homes[Team.BLUE_TEAM], flag_homes[Team.RED_TEAM]]) == 0):
                    raise Exception("Flag coordinates must not lie on the axes of the positive quadrant when gps_env is False")

                #environment size
                flag_xmin = min(flag_homes[Team.BLUE_TEAM][0], flag_homes[Team.RED_TEAM][0])
                flag_ymin = min(flag_homes[Team.BLUE_TEAM][1], flag_homes[Team.RED_TEAM][1])

                flag_xmax = max(flag_homes[Team.BLUE_TEAM][0], flag_homes[Team.RED_TEAM][0])
                flag_ymax = max(flag_homes[Team.BLUE_TEAM][1], flag_homes[Team.RED_TEAM][1])

                self.env_size = np.array([
                    flag_xmax + flag_xmin,
                    flag_ymax + flag_ymin
                ])
                env_bounds = np.array([
                    (0., 0.),
                    self.env_size
                ])
            else:
                env_bounds = np.asarray(env_bounds)

                if len(env_bounds.shape) == 1:
                    if np.any(env_bounds == 0.):
                        raise Exception("Environment max bounds must be > 0 when specified in meters")

                    #environment size
                    self.env_size = env_bounds
                    env_bounds = np.array([
                        (0., 0.),
                        env_bounds
                    ])
                else:
                    if not np.all(env_bounds[0] == 0.):
                        raise Exception("Environment min bounds must be 0 when specified in meters")

                    if np.any(env_bounds[1] == 0.):
                        raise Exception("Environment max bounds must be > 0 when specified in meters")

            self.env_diag = np.linalg.norm(self.env_size)
            self.env_bounds_vertices = np.array([
                env_bounds[0],
                (env_bounds[1][0], env_bounds[0][1]),
                env_bounds[1],
                (env_bounds[0][0], env_bounds[1][1])
            ])

            ### flags home ###
            #auto home
            if self._is_auto_string(flag_homes[Team.BLUE_TEAM]) and self._is_auto_string(flag_homes[Team.RED_TEAM]):
                if flag_homes_unit == "ll" or flag_homes_unit == "wm_xy":
                    raise Exception("'ll' (Lat/Long) and 'wm_xy' (web mercator xy) units should only be used when gps_env is True")
                flag_homes[Team.BLUE_TEAM] = np.array([7/8*self.env_size[0], 0.5*self.env_size[0]])
                flag_homes[Team.RED_TEAM] = np.array([1/8*self.env_size[0], 0.5*self.env_size[0]])
            elif self._is_auto_string(flag_homes[Team.BLUE_TEAM]) or self._is_auto_string(flag_homes[Team.RED_TEAM]):
                raise Exception("Flag homes are either all 'auto', or all specified")
            else:
                flag_homes[Team.BLUE_TEAM] = np.asarray(flag_homes[Team.BLUE_TEAM])
                flag_homes[Team.RED_TEAM] = np.asarray(flag_homes[Team.RED_TEAM])

            #blue flag
            if (
                np.any(flag_homes[Team.BLUE_TEAM] <= env_bounds[0]) or
                np.any(flag_homes[Team.BLUE_TEAM] >= env_bounds[1])
            ):
                raise Exception(f"Blue flag home {flag_homes[Team.BLUE_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}")

            #red flag
            if (
                np.any(flag_homes[Team.RED_TEAM] <= env_bounds[0]) or
                np.any(flag_homes[Team.RED_TEAM] >= env_bounds[1])
            ):
                raise Exception(f"Red flag home {flag_homes[Team.RED_TEAM]} must fall within (non-inclusive) environment bounds {env_bounds}")

            ### scrimmage line ###
            #TODO, check that flags are not on scrimamge line
            if self._is_auto_string(scrimmage_coords):
                flags_vec = flag_homes[Team.BLUE_TEAM] - flag_homes[Team.RED_TEAM]

                scrim_vec1 = np.array([-flags_vec[1], flags_vec[0]])
                scrim_vec2 = np.array([flags_vec[1], -flags_vec[0]])
                flags_midpoint = 0.5 * (flag_homes[Team.BLUE_TEAM] + flag_homes[Team.RED_TEAM])

                scrimmage_coord1 = self._get_polygon_intersection(flags_midpoint, scrim_vec1, self.env_bounds_vertices)[1]
                scrimmage_coord2 = self._get_polygon_intersection(flags_midpoint, scrim_vec2, self.env_bounds_vertices)[1]
                scrimmage_coords = np.asarray([scrimmage_coord1, scrimmage_coord2])
            else:
                if scrimmage_coords_unit == "ll" or scrimmage_coords_unit == "wm_xy":
                    raise Exception("'ll' (Lat/Long) and 'wm_xy' (web mercator xy) units should only be used when gps_env is True")

                scrimmage_coords = np.asarray(scrimmage_coords)

                if np.all(scrimmage_coords[0] == scrimmage_coords[1]):
                    raise Exception("Scrimmage line must be specified with two DIFFERENT coordinates")

                #coord1 on border check
                coord1_on_border = True
                if not (
                    (
                        (env_bounds[0][0] <= scrimmage_coords[0][0] <= env_bounds[1][0]) and
                        (scrimmage_coords[0][1] == env_bounds[0][1]) or (scrimmage_coords[0][1] == env_bounds[1][1])
                        ) or
                    (
                        (env_bounds[0][1] <= scrimmage_coords[0][1] <= env_bounds[1][1]) and
                        (scrimmage_coords[0][0] == env_bounds[0][0]) or (scrimmage_coords[0][0] == env_bounds[1][0])
                        )
                ):
                    coord1_on_border = False

                #coord2 on border check
                coord2_on_border = True
                if not (
                    (
                        (env_bounds[0][0] <= scrimmage_coords[1][0] <= env_bounds[1][0]) and
                        (scrimmage_coords[1][1] == env_bounds[0][1]) or (scrimmage_coords[1][1] == env_bounds[1][1])
                        ) or
                    (
                        (env_bounds[0][1] <= scrimmage_coords[1][1] <= env_bounds[1][1]) and
                        (scrimmage_coords[1][0] == env_bounds[0][0]) or (scrimmage_coords[1][0] == env_bounds[1][0])
                        )
                ):
                    coord2_on_border = False

                #env biseciton check
                if not coord1_on_border and not coord2_on_border:
                    full_scrim_line = LineString(scrimmage_coords)
                    scrim_line_env_intersection = intersection(full_scrim_line, Polygon(self.env_bounds_vertices))

                    if (
                        scrim_line_env_intersection.is_empty or
                        len(scrim_line_env_intersection.coords) == 1 #only intersects a vertex
                    ):
                        raise Exception(
                            f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                        )
                    else:
                        no_bisection = False
                        scrim_line_env_intersection = np.array(scrim_line_env_intersection.coords)

                        #does not make it all the way across environment
                        if (
                            np.any(np.all(scrim_line_env_intersection == scrimmage_coords[0], axis=1)) or
                            np.any(np.all(scrim_line_env_intersection == scrimmage_coords[1], axis=1))
                        ):
                            no_bisection = True
                        #along left and bottom env bounds
                        elif np.any(np.all(scrim_line_env_intersection == self.env_bounds_vertices[0], axis=1)):
                            if (
                                np.any(np.all(scrim_line_env_intersection == self.env_bounds_vertices[1], axis=1)) or
                                np.any(np.all(scrim_line_env_intersection == self.env_bounds_vertices[3], axis=1))
                            ):
                                no_bisection = True
                        #along right and top env bounds
                        elif np.any(np.all(scrim_line_env_intersection == self.env_bounds_vertices[1], axis=1)):
                            if (
                                np.any(np.all(scrim_line_env_intersection == self.env_bounds_vertices[1], axis=1)) or
                                np.any(np.all(scrim_line_env_intersection == self.env_bounds_vertices[3], axis=1))
                            ):
                                no_bisection = True
                        #no bisection
                        if no_bisection:
                            raise Exception(
                                f"Specified scrimmage line coordinates {scrimmage_coords} create a line that does not bisect the environment of bounds {env_bounds}"
                            )
                #flag biseciton check
                #TODO

        ### Set Attributes ###
        #environment geometries
        self.env_bounds = env_bounds
        self.env_bounds_unit = env_bounds_unit

        self.flag_homes = flag_homes
        self.flag_homes_unit = flag_homes_unit

        self.scrimmage_coords = scrimmage_coords
        self.scrimmage_coords_unit = scrimmage_coords_unit
        self.scrimmage_vec = scrimmage_coords[1] - scrimmage_coords[0]

        #on sides
        scrim2blue = self.flag_homes[Team.BLUE_TEAM] - scrimmage_coords[0]
        scrim2red = self.flag_homes[Team.RED_TEAM] - scrimmage_coords[0]

        self.on_sides_sign = {}
        self.on_sides_sign[Team.BLUE_TEAM] = np.sign(self._cross_product(self.scrimmage_vec, scrim2blue))
        self.on_sides_sign[Team.RED_TEAM] = np.sign(self._cross_product(self.scrimmage_vec, scrim2red))

        #agent and flag geometries
        if self.lidar_obs:
            self.lidar_range = lidar_range

        self.agent_radius = agent_radius
        self.flag_radius = flag_radius
        self.catch_radius = catch_radius
        self.flag_keepout = flag_keepout

    def _get_line_intersection(self, origin: np.ndarray, vec: np.ndarray, line: np.ndarray):
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

    def _get_polygon_intersection(self, origin: np.ndarray, vec: np.ndarray, polygon: np.ndarray):
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
        map_caching_dir = str(pathlib.Path(__file__).resolve().parents[1] / '__mapcache__')
        if not os.path.isdir(map_caching_dir):
            os.mkdir(map_caching_dir)

        lon1, lat1 = np.round(_sm2ll(*self.env_bounds[0]), 7)
        lon2, lat2 = np.round(_sm2ll(*self.env_bounds[1]), 7)

        map_cache_path = os.path.join(map_caching_dir, f'tile@(({lat1},{lon1}), ({lat2},{lon2})).pkl')

        if os.path.exists(map_cache_path):
            #load cached environment map(s)
            with open(map_cache_path, 'rb') as f:
                map_cache = pickle.load(f)

            topo_img = map_cache["topographical_image"]
            self.background_img = map_cache["render_image"]
        else:
            #retrieve maps from tile provider
            topo_tile_source = cx.providers.CartoDB.DarkMatterNoLabels #DO NOT CHANGE!
            render_tile_source = cx.providers.CartoDB.Voyager #DO NOT CHANGE!

            render_tile_bounds = self.env_bounds + self.arena_buffer_frac * np.asarray([[-self.env_diag], [self.env_diag]])

            topo_tile, topo_ext = cx.bounds2img(
                *self.env_bounds.flatten(), zoom='auto', source=topo_tile_source, ll=False,
                wait=0, max_retries=2, n_connections=1, use_cache=False, zoom_adjust=None
            )
            render_tile, render_ext = cx.bounds2img(
                *render_tile_bounds.flatten(), zoom='auto', source=render_tile_source, ll=False,
                wait=0, max_retries=2, n_connections=1, use_cache=False, zoom_adjust=None
            )

            topo_img = self._crop_tiles(topo_tile[:,:,:-1], topo_ext, *self.env_bounds.flatten(), ll=False)
            self.background_img = self._crop_tiles(render_tile[:,:,:-1], render_ext, *render_tile_bounds.flatten(), ll=False)

            #cache maps
            map_cache = {"topographical_image": topo_img, "render_image": self.background_img}
            with open(map_cache_path, 'wb') as f:
                pickle.dump(map_cache, f)

        ### Topology Construction ###
        #mask by water color on topo image
        water_x, water_y  = self.flag_homes[Team.BLUE_TEAM] #assume flag is in water
        water_pixel_x = ceil(topo_img.shape[1] * (water_x / self.env_size[0])) - 1
        water_pixel_y = ceil(topo_img.shape[0] * (1 - water_y / self.env_size[1])) - 1

        water_pixel_color = topo_img[water_pixel_y, water_pixel_x]
        mask = np.all(topo_img == water_pixel_color, axis=-1)
        water_connectivity = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]]
        )
        labeled_mask, _ = label(mask, structure=water_connectivity)
        target_label = labeled_mask[water_pixel_y, water_pixel_x]

        grayscale_topo_img = cv2.cvtColor(topo_img, cv2.COLOR_RGB2GRAY)
        water_pixel_color_gray = grayscale_topo_img[water_pixel_y, water_pixel_x]
        
        land_mask = (
            (labeled_mask == target_label) +
            (water_pixel_color_gray <= grayscale_topo_img) * (grayscale_topo_img <= water_pixel_color_gray + 2)
        )

        #water contours
        land_mask_binary = 255*land_mask.astype(np.uint8)
        water_contours, _ = cv2.findContours(land_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #https://docs.opencv.org/4.10.0/d4/d73/tutorial_py_contours_begin.html
        #https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html

        border_contour = max(water_contours, key=cv2.contourArea)
        #TODO: check if this is just the environment bounds, then non-convex approximation will go to the largest island
        border_land_mask = cv2.drawContours(np.zeros_like(land_mask_binary), [border_contour], -1, 255, -1)

        #island contours
        water_mask = np.logical_not(land_mask)
        island_binary = 255*(border_land_mask * water_mask).astype(np.uint8)
        island_contours, _ = cv2.findContours(island_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #approximate outer contour (border land)
        eps = self.topo_contour_eps * cv2.arcLength(border_contour, True)
        border_cnt_approx = cv2.approxPolyDP(border_contour, eps, True)

        border_land_mask_approx = cv2.drawContours(np.zeros_like(land_mask_binary), [border_cnt_approx], -1, 255, -1)
        border_land_mask_approx = cv2.drawContours(border_land_mask_approx, [border_cnt_approx], -1, 0, 0)

        labeled_border_land_mask_approx, _ = label(border_land_mask_approx, structure=water_connectivity)
        target_water_label = labeled_border_land_mask_approx[water_pixel_y, water_pixel_x]
        border_land_mask_approx = labeled_border_land_mask_approx == target_water_label

        #approximate island contours
        island_cnts_approx = []
        for i, cnt in enumerate(island_contours):
            eps = self.topo_contour_eps * cv2.arcLength(cnt, True)
            cnt_approx = cv2.approxPolyDP(cnt, 10*eps, True)
            cvx_hull = cv2.convexHull(cnt_approx)
            if len(cvx_hull) > 1:
                island_cnts_approx.append(cvx_hull)

        island_mask_approx = cv2.drawContours(255*np.ones_like(island_binary), island_cnts_approx, -1, 0, -1) #convex island masks

        #final approximate land mask
        land_mask_approx = border_land_mask_approx * island_mask_approx/255

        #squeeze contours
        border_cnt = self._img2env_coords(border_cnt_approx.squeeze(), topo_img.shape)
        island_cnts = [self._img2env_coords(cnt.squeeze(), topo_img.shape) for cnt in island_cnts_approx]

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
        #convert lat/lon bounds to Web Mercator XY (EPSG:3857)
        if ll:
            left, bottom = mt.xy(w, s)
            right, top = mt.xy(e, n)
        else:
            left, bottom = w, s
            right, top = e, n

        #determine crop
        X_size = ext[1] - ext[0]
        Y_size = ext[3] - ext[2]

        img_size_x = img.shape[1]
        img_size_y = img.shape[0]

        crop_start_x = ceil(img_size_x * (left - ext[0]) / X_size) - 1
        crop_end_x = ceil(img_size_x * (right - ext[0]) / X_size) - 1

        crop_start_y = ceil(img_size_y * (ext[2] - top) / Y_size)
        crop_end_y = ceil(img_size_y * (ext[2] - bottom) / Y_size) - 1

        #crop image
        cropped_img = img[
            crop_start_y : crop_end_y,
            crop_start_x : crop_end_x,
            :
        ]

        return cropped_img

    def _img2env_coords(self, cnt, image_shape):
        cnt = cnt.astype(float) #convert contour array to float64 so as not to lose precision
        cnt[:, 0] =  self.env_size[0] * cnt[:, 0] / (image_shape[1] - 1)
        cnt[:, 1] =  self.env_size[1] * (1 - cnt[:, 1] / (image_shape[0] - 1))

        return cnt

    def _generate_segments_from_obstacles(self, obstacle, n_quad_segs):
        if isinstance(obstacle, PolygonObstacle):
            vertices = obstacle.anchor_points
        else: #CircleObstacle
            radius = obstacle.radius 
            center = obstacle.center_point
            vertices = list(Point(*center).buffer(radius, quad_segs=n_quad_segs).exterior.coords)[:-1] #approximate circle with an octagon

        segments = [[*vertex, *vertices[(i+1) % len(vertices)]] for i, vertex in enumerate(vertices)]
        
        return segments

    def render(self):
        """Overridden method inherited from `Gym`."""
        return self._render()

    def _render(self):
        """
        Overridden method inherited from `Gym`.

        Draws all players/flags/etc on the pygame screen.
        """
        if self.screen is None:
            #create screen
            pygame.init()
            pygame.display.set_caption("Capture The Flag")
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                self.isopen = True
                self.agent_id_font = pygame.font.SysFont(None, int(2*self.pixel_size*self.agent_radius))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            else:
                raise Exception(
                    f"Sorry, render modes other than f{self.metadata['render_modes']} are not supported"
                )
            #render background
            if self.gps_env:
                pygame_background_img = pygame.surfarray.make_surface(
                    np.transpose(self.background_img, (1,0,2)) #pygame assumes images are (h, w, 3)
                )
                self.pygame_background_img = pygame.transform.scale(pygame_background_img, (self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state == {}:
            return None

        # Background
        if self.gps_env:
            self.screen.blit(self.pygame_background_img, (0, 0))
        else:
            self.screen.fill((255, 255, 255))

        # Scrimmage line
        draw.line(
            self.screen,
            (128, 128, 128),
            self.env_to_screen(self.scrimmage_coords[0]),
            self.env_to_screen(self.scrimmage_coords[1]),
            width=self.boundary_width
        )

        # Obstacles
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                draw.circle(
                    self.screen,
                    (0, 0, 0),
                    self.env_to_screen(obstacle.center_point),
                    radius=obstacle.radius * self.pixel_size,
                    width=self.boundary_width
                )
            elif isinstance(obstacle, PolygonObstacle):
                draw.polygon(
                        self.screen,
                        (0, 0, 0),
                        [self.env_to_screen(p) for p in obstacle.anchor_points],
                        width=self.boundary_width,
                    )
        
        # Aquaticus field points
        if self.render_field_points:
            for v in self.config_dict["aquaticus_field_points"]:
                draw.circle(
                    self.screen,
                    (128, 0, 128),
                    self.env_to_screen(self.config_dict["aquaticus_field_points"][v]),
                    radius=5
                )

        # Flags and players
        for team in Team:
            flag = self.flags[int(team)]
            teams_players = self.agents_of_team[team]
            color = "blue" if team == Team.BLUE_TEAM else "red"
            opp_color = "red" if team == Team.BLUE_TEAM else "blue"

            # team home region
            home_center_screen = self.env_to_screen(self.flags[int(team)].home)
            draw.circle(
                    self.screen,
                    (128, 128, 128),
                    home_center_screen,
                    radius=self.catch_radius * self.pixel_size,
                    width=self.boundary_width,
                )

            # team flag (not picked up)
            if not self.state["flag_taken"][int(team)]:
                flag_pos_screen = self.env_to_screen(flag.pos)
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    radius=self.flag_radius * self.pixel_size,
                )
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    radius=(self.flag_keepout - self.agent_radius) * self.pixel_size,
                    width=self.boundary_width,
                )

            # players
            for player in teams_players:
                blit_pos = self.env_to_screen(player.pos)

                #trajectory
                if self.render_traj_mode:
                    #traj
                    if self.render_traj_mode.startswith("traj"):
                        for prev_blit_pos in reversed(self.traj_render_buffer[player.id]['traj']):
                            draw.circle(
                                self.screen,
                                color,
                                prev_blit_pos,
                                radius=2,
                                width=0
                            )
                    #agent 
                    if self.render_traj_mode.endswith("agent"):
                        for prev_rot_blit_pos, prev_agent_surf in reversed(self.traj_render_buffer[player.id]['agent']):
                            self.screen.blit(prev_agent_surf, prev_rot_blit_pos)
                    #history
                    elif self.render_traj_mode.endswith("history"):
                        raise NotImplementedError()

                #lidar
                if self.lidar_obs and self.render_lidar:
                    ray_headings_global = np.deg2rad((heading_angle_conversion(player.heading) + self.lidar_ray_headings) % 360)
                    ray_vecs = np.array([np.cos(ray_headings_global), np.sin(ray_headings_global)]).T
                    lidar_starts = player.pos + self.agent_radius * ray_vecs
                    for i in range(self.num_lidar_rays):
                        draw.line(
                            self.screen,
                            color,
                            self.env_to_screen(lidar_starts[i]),
                            self.env_to_screen(self.state["lidar_ends"][player.id][i]),
                            width=2
                        )
                #tagging
                player.render_tagging(self.tagging_cooldown)

                #heading
                orientation = Vector2(list(mag_heading_to_vec(1.0, player.heading)))
                ref_angle = -orientation.angle_to(self.PYGAME_UP)

                #transform position to pygame coordinates
                rotated_surface = rotozoom(player.pygame_agent, ref_angle, 1.0)
                rotated_surface_size = np.array(rotated_surface.get_size())
                rotated_blit_pos = blit_pos - 0.5*rotated_surface_size

                #flag pickup
                if player.has_flag:
                    draw.circle(
                        rotated_surface,
                        opp_color,
                        0.5*rotated_surface_size,
                        radius=0.55*(self.pixel_size * self.agent_radius)
                    )

                #agent id
                if self.render_ids:
                    agent_id_blit_pos = (
                        0.5*rotated_surface_size[0] - 0.35 * self.pixel_size * self.agent_radius,
                        0.5*rotated_surface_size[1] - 0.6 * self.pixel_size * self.agent_radius
                    ) #TODO: adjust this based on a rect of the number and maybe agent too?
                    if self.gps_env:
                        font_color = "white"
                    else:
                        font_color = "white" if team == Team.BLUE_TEAM else "black"

                    player_number_label = self.agent_id_font.render(str(player.id), True, font_color)
                    rotated_surface.blit(player_number_label, agent_id_blit_pos)

                #blit agent onto screen
                self.screen.blit(rotated_surface, rotated_blit_pos)

                #save agent surface for trajectory rendering
                if (
                    self.render_traj_mode and
                    self.render_ctr % self.num_renders_per_step == 0
                ):
                    #add traj/ agent render data
                    if self.render_traj_mode.startswith("traj"):
                        self.traj_render_buffer[player.id]['traj'].insert(0, blit_pos)
 
                    if (
                        self.render_traj_mode.endswith("agent") and
                        (self.render_ctr / self.num_renders_per_step) % self.render_traj_freq == 0
                    ):
                        self.traj_render_buffer[player.id]['agent'].insert(0, (rotated_blit_pos, rotated_surface))

                    elif self.render_traj_mode.endswith("history"):
                        raise NotImplementedError()

                    #truncate traj
                    if self.render_traj_cutoff is not None:
                        agent_render_cutoff = (
                            floor(self.render_traj_cutoff / self.render_traj_freq) +
                            (
                                (
                                    (self.render_ctr / self.num_renders_per_step) % self.render_traj_freq +
                                    self.render_traj_freq * floor(self.render_traj_cutoff / self.render_traj_freq)
                                ) <= self.render_traj_cutoff
                            )
                        )
                        self.traj_render_buffer[player.id]['traj'] = self.traj_render_buffer[player.id]['traj'][
                            : self.render_traj_cutoff
                        ]
                        self.traj_render_buffer[player.id]['agent'] = self.traj_render_buffer[player.id]['agent'][
                            : agent_render_cutoff
                        ]

        # Agent-to-agent distances 
        for blue_player in self.agents_of_team[Team.BLUE_TEAM]:
            if not blue_player.is_tagged or (blue_player.is_tagged and blue_player.on_own_side):
                for red_player in self.agents_of_team[Team.RED_TEAM]:
                    if not red_player.is_tagged or (red_player.is_tagged and red_player.on_own_side):
                        blue_player_pos = np.asarray(blue_player.pos)
                        red_player_pos = np.asarray(red_player.pos)
                        a2a_dis = np.linalg.norm(blue_player_pos - red_player_pos)
                        if a2a_dis <= 2*self.catch_radius:
                            hsv_hue = (a2a_dis - self.catch_radius) / (2*self.catch_radius - self.catch_radius)
                            hsv_hue = 0.33 * np.clip(hsv_hue, 0, 1)
                            line_color = tuple(255 * np.asarray(colorsys.hsv_to_rgb(hsv_hue, 0.9, 0.9)))

                            draw.line(
                                self.screen, 
                                line_color,
                                self.env_to_screen(blue_player_pos),
                                self.env_to_screen(red_player_pos),
                                width=self.a2a_line_width
                            )

        # Render
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()
        
        # Record
        if self.record_render:
            self.render_buffer.append(
                np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
            )

        # Update counter
        self.render_ctr += 1

    def env_to_screen(self, pos):
        screen_pos = self.pixel_size * np.asarray(pos)
        screen_pos[0] += self.arena_buffer
        screen_pos[1] = self.arena_height - screen_pos[1] + self.arena_buffer

        return screen_pos

    def buffer_to_video(self):
        """Convert and save current render buffer as a video"""
        if len(self.render_buffer) > 0:
            video_file_dir = str(pathlib.Path(__file__).resolve().parents[1] / 'videos')
            if not os.path.isdir(video_file_dir):
                os.mkdir(video_file_dir)

            now = datetime.now() #get date and time
            video_id = now.strftime("%m-%d-%Y_%H-%M-%S")

            if self.recording_format == "mp4":
                video_file_name = f"pyquaticus_{video_id}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif self.recording_format == "avi":
                video_file_name = f"pyquaticus_{video_id}.avi"
                fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
            else:
                raise NotImplementedError(f"Saving video as .{self.recording_format} file is not supported.")

            video_file_path = os.path.join(video_file_dir, video_file_name)

            out = cv2.VideoWriter(video_file_path, fourcc, self.render_fps, (self.screen_width, self.screen_height))
            for img in self.render_buffer:
                out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            out.release()
        else:
            print("Attempted to save video but render_buffer is empty!")
            print()

    def close(self):
        """Overridden method inherited from `Gym`."""
        #TODO: save video if have not already on early exiting
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _min(self, a, b) -> bool:
        """Convenience method for determining a minimum value. The standard `min()` takes much longer to run."""
        if a < b:
            return a
        else:
            return b

    def _cross_product(self, u, v):
        return u[0] * v[1] - u[1] * v[0]

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
            for i, obstacle in enumerate(self.state["dist_to_obstacles"][agent_id]):
                orig_obs[f"obstacle_{i}_distance"] = obstacle[0]
                orig_obs[f"obstacle_{i}_bearing"] = obstacle[1]

        if normalize:
            orig_obs = self.agent_obs_normalizer.normalized(orig_obs)

        return orig_obs
