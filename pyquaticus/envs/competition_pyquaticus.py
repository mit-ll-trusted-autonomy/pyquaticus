

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

from pyquaticus.envs.pyquaticus import PyQuaticusEnvBase, PyQuaticusEnv





class CompPyquaticusEnv(PyQuaticusEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pyquaticus_vComp",
    }
    def __init__(
                self, 
                action_space: Union[str, list[str], dict[str, str]] = "discrete",
                reward_config: dict = None,
                config_dict = config_dict_std,
                render_mode: Optional[str] = None
                ):

        # Set Competition Specific Variables

        # Set Base Pyquaticus Variables
        super().__init__(team_size=4,
                        action_space=action_space,
                        reward_config=reward_config,
                        config_dict=config_dict,
                        render_mode=render_mode
                        )


        # Untag Coordinates offset from bounder corners by a little bit
        self.blue_untag_coords = [[self.env_ul[0]+10, self.env_ul[1]-10]
                                    , [self.env_ll[0]+10, self.env_ll[1]+10]]

        self.red_untag_coords = [[self.env_ur[0]-10, self.env_ur[1]-10]
                                    , [self.env_lr[0]-10, self.env_lr[1]+10]]

        #[self.env_ul, self.env_ll]

        #TODO: Decide if untag location needs to be in observation?
        #TODO: Add home locations to state
        #TODO: Fix Rendering to show untag locations
        #Probability tag config

        #Add Additional State Elements

        #Base Pyquaticus Class Will Initialize Most Game Events


    def step(self, raw_action_dict):
        #TODO Likely will need to rewrite this entire function

        return super().step(raw_action_dict)


    def set_config_values(self, config_dict):
        
        # Handle Competition Specific Variables

        # assert 'tag_probability' in config_dict and 'power_play_percentage' in config_dict, 
        #   "Error: Please use MCTF competition configuration dictionary"
        
        self.tag_probability = config_dict['tag_probability']

        # assert config_dict['power_play_percentage'] <= 0.5,
        #   "Power Play Percentage cannot exceed 50%% of game time"
        self.power_play_percentage = config_dict['power_play_percentage']
        self.untag_radius = config_dict['untag_radius']


        del config_dict['tag_probability']
        del config_dict['power_play_percentage']
        del config_dict['untag_radius']
        
        # Handle Normal MCTF Configurations
        super().set_config_values(config_dict)
    def _check_untag(self):
        """Untags the player if they return to their own home base."""

        for i, player in enumerate(self.players.values()):
            if self.state['agent_oob'][self.agents.index(player.id)]:
                #If agent is OOB skip untag check
                continue
            team = int(player.team)
            coords = []
            if int(team) == int(Team.BLUE_TEAM):
                coords = [self.env_ul, self.env_ll]
            else: 
                coords = [self.env_ur, self.env_lr]
            # flag_home = self.flags[team].home
            flag_distance = None
            for home in coords:
                dist = self.get_distance_between_2_points(
                    player.pos, home
                )
                if flag_distance == None or dist < flag_distance:
                    flag_distance = dist

            if flag_distance < self.untag_radius and player.is_tagged:
                player.is_tagged = False
                self.state['agent_is_tagged'][i] = 0
    # def set_geom_config(self,):
    def _check_untag_vectorized(self):
        """Untags the player if they return to their own home base."""
        for team, team_agent_inds in self.agent_inds_of_team.items():
            agent_poses = self.state['agent_position'][team_agent_inds]
            coords = []
            if int(team) == Team.BLUE_TEAM:
                coords = self.blue_untag_coords
            else: 
                coords = self.red_untag_coords
            flag_distances = []
            for home in coords:
                flag_distances.append(np.linalg.norm(home - agent_poses))

            flag_distances = np.minimum(flag_distances[0], flag_distances[1])
            agent_is_tagged = self.state['agent_is_tagged'][team_agent_inds]

            agent_untagged = (flag_distances < self.untag_radius) & agent_is_tagged
            agent_untagged_inds = team_agent_inds[np.where(agent_untagged)[0]]

            self.state['agent_is_tagged'][agent_untagged_inds] = 0
            for agent_idx in agent_untagged_inds:
                self.players[self.agents[agent_idx]].is_tagged = False
                
    def create_background_image(self,):

        # Create Default Pyquaticus CTF game field
        super().create_background_image()
        # Add Untag Region 

        # Blue Team
        draw.circle(self.pygame_background_img,
                        (135,206,235),
                        self.env_to_screen(self.env_ul),
                        radius=20*self.pixel_size,
                        width=0,#self.boundary_width,
                        draw_bottom_right=True,
                    )
        draw.circle(self.pygame_background_img,
                        (135,206,235),
                        self.env_to_screen(self.env_ll),
                        radius=20*self.pixel_size,
                        width=0,#self.boundary_width,
                        draw_top_right=True,
                    )

        # Red Team
        draw.circle(self.pygame_background_img,
                        (254,132,132),
                        self.env_to_screen(self.env_ur),
                        radius=20*self.pixel_size,
                        width=0,#self.boundary_width,
                        draw_bottom_left=True,
                    )
        draw.circle(self.pygame_background_img,
                        (254,132,132),
                        self.env_to_screen(self.env_lr),
                        radius=20*self.pixel_size,
                        width=0,#self.boundary_width,
                        draw_top_left=True,
                    )

        #Redraw Background TODO Clean this or seperate it out from pyquaticus class
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



    def reset(self, seed=None, options: Optional[dict] = None):
        obs, infos = super().reset(seed=seed, options=options)

        # Add in MCTF Competition State Elements
        self.state['disabled'] = np.zeros(self.num_agents, dtype=bool)
        # self.state['']
        return obs, infos


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
                    # untag_coords = []
                    if team_idx == int(Team.BLUE_TEAM):
                        untag_coords = self.blue_untag_coords
                    else:

                        untag_coords = self.red_untag_coords
                    dist = None 
                    heading_error = None
                    for coord in untag_coords:
                        temp_dist, temp_heading_error = mag_bearing_to(player.pos, coord, player.heading)
                        if dist == None or temp_dist < dist:
                            dist = temp_dist 
                            heading_error = temp_heading_error


                    # _, heading_error = mag_bearing_to(player.pos, self.env_ur, player.heading)
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
    def _render(self,):
        super()._render()

    # _set_state_from_init_dict

    # def _move_agents(self, action_dict): Use Parents

    # def _get_oob_recover_rel_heading(self, pos, heading):

    # def _check_oob(self):

    # def _check_oob_vectorized(self):

    # def _check_flag_pickups_vectorized(self):
    # def _check_agent_made_tag_vectorized(self):

    #def _check_untag_vectorized(self)

    #def _check_flag_captures(self)
    #def get_distance_between_2_points(self,)
    #def _set_dones(self)
    #def compute_rewards(self)
    # def _reset_dones(self):
    # def _set_state_from_init_dict(self,)
    # def _set_player_attributes_from_state(sef,)
    # def _set_flag_attributes_from_state(self):

    # def _generate_agent_starts()

    # def _check_valid_pos()

    # def _update_dist_bearing_to_obstacles(self)

    # def _build_base_env_geom(self)

    #def render(self)

    #
















