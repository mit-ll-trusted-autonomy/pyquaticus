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

import copy
import functools
import math
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pygame
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pygame import SRCALPHA, Surface, draw
from pygame.math import Vector2
from pygame.transform import rotozoom

from pyquaticus.utils.obs_utils import ObsNormalizer
from pyquaticus.utils.pid import PID
from pyquaticus.utils.utils import (
    angle180,
    clip,
    closest_point_on_line,
    get_rot_angle,
    get_screen_res,
    mag_bearing_to,
    mag_heading_to_vec,
    rc_intersection,
    reflect_vector,
    rot2d,
    vec_to_mag_heading,
)

MAX_SPEED = 5.0

config_dict_std = {
    "world_size": [110.0, 55.0],  # meters
    "pixel_size": 10,  # pixels/meter
    "scrimmage_line": 55.0,  # horizontal location (meters)
    "agent_radius": 2.0,  # meters
    "catch_radius": 10.0,  # meters
    "flag_keepout": 5.0,  # minimum distance (meters) between agent and flag centers
    "keepout_bounce": (
        0.0 # In the WP competition there is no penalty for going withing the ctach radius of the flag
    ),  # [0, 1] percentage of velocity in direction of collision at which an agent is repelled from its keepout zone
    "max_speed": MAX_SPEED,  # meters / s
    "own_side_accel": (
        1.0
    ),  # [0, 1] percentage of max acceleration that can be used on your side of scrimmage
    "opp_side_accel": (
        1.0
    ),  # [0, 1] percentage of max acceleration that can be used on opponent's side of scrimmage
    "wall_bounce": (
        0.5
    ),  # [0, 1] percentage of current speed (x or y) at which an agent is repelled from a wall (vertical or horizontal)
    "tau": (
        1 / 10
    ),  # length of timestep (seconds) between state updates and for updating action input from demonstrator or rl
    "max_time": 120.0,  # maximum time (seconds) per episode
    "max_screen_size": get_screen_res(),
    "random_init": (
        False
    ),  # randomly initialize agents' positions for ctf mode (within fairness constraints)
    "save_traj": False,  # save traj as pickle
    "render_fps": 30,
    "normalize": True,  # Flag for normalizing the observation space.
    "tagging_cooldown": (
        10.0
    ),  # Cooldown on an agent after they tag another agent, to prevent consecutive tags
    # MOOS dynamics parameters
    "speed_factor": 20.0,  # Multiplicative factor for desired_speed -> desired_thrust
    "thrust_map": np.array(  # Piecewise linear mapping from desired_thrust to speed
        [[-100, 0, 20, 40, 60, 80, 100], [-2, 0, 1, 2, 3, 5, 5]]
    ),
    "max_thrust": 70,  # Limit on vehicle thrust
    "max_rudder": 100,  # Limit on vehicle rudder actuation
    "turn_loss": 0.85,
    "turn_rate": 70,
    "max_acc": 1,  # m / s**2
    "max_dec": 1,  # m / s**2
    "suppress_numpy_warnings": (
        True  # Option to stop numpy from printing warnings to the console
    ),
    "teleport_on_tag" : False, # Option for the agent when tagged to teleport home or not
}
""" Standard configuration setup """


def get_std_config() -> dict:
    """Gets a copy of the standard configuration, ideal for minor modifications to the standard configuration."""
    return copy.deepcopy(config_dict_std)


# action space key combos
# maps discrete action id to (speed, heading)
ACTION_MAP = []
for spd in [MAX_SPEED, MAX_SPEED / 2.0]:
    for hdg in range(180, -180, -45):
        ACTION_MAP.append([spd, hdg])
# add a none action
ACTION_MAP.append([0.0, 0.0])


class Team(Enum):
    """Enum for teams."""

    BLUE_TEAM = 0
    RED_TEAM = 1

    def __int__(self):
        """Returns integer equivalent of enum for indexing."""
        return self.value

    def __str__(self):
        """Returns string equivalent for enum."""
        return self.name

    def __repr__(self):
        return f"{self.name}({self.value})"


@dataclass
class Player:
    """
    Class to hold data on each player/agent in the game.

    Attributes
    ----------
        id: The ID of the agent (also used as an index)
        team: The team of the agent (red or blue)
        r: Agent radius
        thrust: The engine thrust
        pos: The position of the agent [x, y]
        speed: The speed of the agent (m / s)
        heading: The heading of the agent (deg), maritime convention: north is 0, east is 90
        pygame_agent: The pygame object that is drawn on screen.
        prev_pos: The previous position of the agent
        has_flag: Indicator for whether or not the agent has the flag
        on_own_side: Indicator for whether or not the agent is on its own side of the field.
    """

    id: int
    team: Team
    r: float
    thrust: float = field(init=False, default_factory=float)
    pos: list[float] = field(init=False, default_factory=list)
    speed: float = field(init=False, default_factory=float)
    heading: float = field(init=False, default_factory=float)
    pygame_agent: Surface = field(init=False, default=None)
    prev_pos: list[float] = field(init=False, default_factory=list)
    has_flag: bool = field(init=False, default=False)
    on_own_side: bool = field(init=False, default=True)
    tagging_cooldown: float = field(init=False)
    home: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Called automatically after __init__ to set up pygame object interface."""
        # Create the shape of the arrow indicating agent orientation
        top_vertex = (self.r, 0)
        left_vertex = (
            self.r - self.r * np.sqrt(2) / 2 + 1,
            self.r + self.r * np.sqrt(2) / 2 - 1,
        )
        right_vertex = (
            self.r + self.r * np.sqrt(2) / 2 - 1,
            self.r + self.r * np.sqrt(2) / 2 - 1,
        )
        center_vertex = (self.r, 1.25 * self.r)

        # Create the actual object
        self.pygame_agent = Surface((2 * self.r, 2 * self.r), SRCALPHA)

        # Adjust color based on which team
        if self.team == Team.BLUE_TEAM:
            draw.circle(self.pygame_agent, (0, 0, 255, 50), (self.r, self.r), self.r)
            draw.circle(
                self.pygame_agent,
                (0, 0, 255),
                (self.r, self.r),
                self.r,
                width=round(self.r / 20),
            )
            draw.polygon(
                self.pygame_agent,
                (0, 0, 255),
                (
                    top_vertex,
                    left_vertex,
                    center_vertex,
                    right_vertex,
                ),
            )
        else:
            draw.circle(self.pygame_agent, (255, 0, 0, 50), (self.r, self.r), self.r)
            draw.circle(
                self.pygame_agent,
                (255, 0, 0),
                (self.r, self.r),
                self.r,
                width=round(self.r / 20),
            )
            draw.polygon(
                self.pygame_agent,
                (255, 0, 0),
                (
                    top_vertex,
                    left_vertex,
                    center_vertex,
                    right_vertex,
                ),
            )

    def reset(self):
        """Method to return a player to their original starting position."""
        self.prev_pos = self.pos
        self.pos = self.home
        self.speed = 0
        if self.team == Team.RED_TEAM:
            self.heading = 90
        else:
            self.heading = -90
        self.thrust = 0
        self.has_flag = False
        self.on_sides = True

    def rotate(self):
        """Method to rotate the player 180"""
        self.prev_pos = self.pos
        self.speed = 0
        self.thrust = 0
        self.has_flag = False
       
        # Need to get which wall the agent bumped into
        x_pos = self.pos[0]
        y_pos = self.pos[1]

        if (x_pos < self.r):
            self.pos[0] += 1
        elif(config_dict_std["world_size"][0] - self.r < x_pos):
            self.pos[0] -= 1
        
        if (y_pos < self.r):
            self.pos[1] += 1
        elif(config_dict_std["world_size"][1] - self.r < y_pos):
            self.pos[1] -= 1
        
        # Rotate 180 degrees
        self.heading = angle180(self.heading + 180)

            
            



@dataclass
class Flag:
    """
    Class for data on the flag.

    Attributes
    ----------
        team: The team the flag belongs to
        home: The flags original position at the start of the round/game
        pos: The flags current position
    """

    team: Team
    home: list[float] = field(default_factory=list, init=False)
    pos: list[float] = field(default_factory=list, init=False)

    def reset(self):
        """Resets the flags `pos` to be `home`."""
        self.pos = copy.deepcopy(self.home)


class PyQuaticusEnv(ParallelEnv):
    """
    ### Description.

    This environment simulates a game of capture the flag with agent dynamics based on MOOS-IvP
    (https://oceanai.mit.edu/ivpman/pmwiki/pmwiki.php?n=IvPTools.USimMarine#section5).

    ### Action Space
    A discrete action space with all combinations of
    max speed, half speed; and
    45 degree heading intervals

    ### Observation Space

        Per Agent (supplied in a dictionary from agent-id to a Box):
            Retrieve flag relative bearing (clockwise degrees)
            Retrieve flag distance (meters)
            Home flag relative bearing (clockwise degrees)
            Home flag distance (meters)
            Wall 1 relative bearing (clockwise degrees)
            Wall 1 distance (meters)
            Wall 2 relative bearing (clockwise degrees)
            Wall 2 distance (meters)
            Wall 3 relative bearing (clockwise degrees)
            Wall 3 distance (meters)
            Wall 4 relative bearing (clockwise degrees)
            Wall 4 distance (meters)
            Own speed (meters per second)
            Own flag status (boolean)
            On side (boolean)
            Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
                Bearing from you (clockwise degrees)
                Distance (meters)
                Heading of other agent relative to the vector to you (clockwise degrees)
                Speed (meters per second)
                Has flag status (boolean)
                On their side status (boolean)
                Tagging cooldown (seconds)
        Note 1 : the angles are 0 when the agent is pointed directly at the object
                 and increase in the clockwise direction
        Note 2 : the wall distances can be negative when the agent is out of bounds
        Note 3 : the boolean args Tag/Flag status are -1 false and +1 true
        Note 4: the values are normalized by default

    ### Rewards

    Reward functions will be learned using an inverse reinforcement learning algorithm (D-REX)

    ### Starting State (need to update)

    Each flag is placed at 0° latitude and 1/8 horizontal world size distance away from the back
    wall of its respective team's territory.

    If random_init is False, agents are spawned facing the scrimmage line at 0° latitude and
    1/4 horizontal world size distance back from the scrimmage line.

    If random_init is True, agents are spawned facing the scrimmage line at 0° latitude and
    equidistant from the scrimmage line (this distance is variable). Then, they are shifted a variable
    distance to either the left or right (same direction for both agents) with the constraints that
    they cannot start inside the flag_keepout zone or behind their flag.

    ### Arguments

    ```
    gym.make('pyquaticus')
    ```
    team_size: number of agents per team
    reward_config: a dictionary configuring the reward structure (see rewards.py)
    config_dict: a dictionary configuring the environment (see config_dict_std above)

    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        team_size: int = 1,
        reward_config: dict = None,
        config_dict=config_dict_std,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config_dict = config_dict

        # set variables from config
        self.set_config_values(self.config_dict)

        self.state = {}
        self.dones = {}
        self.reset_count = 0

        self.learning_iteration = 0

        self.seed()

        self.current_time = 0

        self.num_blue = team_size
        self.num_red = team_size

        self.players = []
        b_players = []
        r_players = []
        self.team_size = team_size

        # Create players, use IDs from [0, (2 * team size) - 1] so their IDs can also be used as indices.
        for i in range(0, self.team_size):
            b_players.append(
                Player(i, Team.BLUE_TEAM, (self.agent_radius * self.pixel_size))
            )
        for i in range(self.team_size, 2 * self.team_size):
            r_players.append(
                Player(i, Team.RED_TEAM, (self.agent_radius * self.pixel_size))
            )
        self.players = b_players + r_players
        self.player_ids = {p.id for p in self.players}

        self.agents = [p.id for p in self.players]
        self.possible_agents = self.agents[:]

        self.agents_of_team = {Team.BLUE_TEAM: b_players, Team.RED_TEAM: r_players}

        # Setup Rewards
        self.reward_config = {} if reward_config is None else reward_config
        for a in self.player_ids:
            if a not in self.reward_config:
                self.reward_config[a] = None
        # Create a PID controller for each agent
        self._pid_controllers = {}
        for player in self.players:
            self._pid_controllers[player.id] = {
                "speed": PID(self.tau, kp=1.0, ki=0.0, kd=0.0, integral_max=0.07),
                "heading": PID(self.tau, kp=0.35, ki=0.0, kd=0.07, integral_max=0.07),
            }

        self.params = {agent.id: {} for agent in self.players}
        self.prev_params = {agent.id: {} for agent in self.players}
        # Create the list of flags that are indexed by self.flags[int(player.team)]

        self.flags = []
        for team in Team:
            self.flags.append(Flag(team))

        # Set tagging cooldown
        for player in self.players:
            player.tagging_cooldown = self.tagging_cooldown

        self.agent_obs_normalizer = self._register_state_elements()

        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.player_ids
        }
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.player_ids
        }

        self.render_mode = render_mode

        # pygame screen
        self.screen = None
        self.clock = None
        self.isopen = False
        if self.render_mode:
            # Pygame Orientation Vector
            self.UP = Vector2((0.0, 1.0))

            # arena
            self.arena_offset = 20  # pixels
            self.border_width = 4  # pixels

            self.arena_width = self.world_size[0] * self.pixel_size
            self.arena_height = self.world_size[1] * self.pixel_size

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

        # set the time
        self.current_time += self.tau
        self.state["current_time"] = self.current_time
        if not set(raw_action_dict.keys()) <= self.player_ids:
            raise ValueError(
                "Keys of action dict should be player ids but got"
                f" {raw_action_dict.keys()}"
            )

        for player in self.players:
            if player.tagging_cooldown != self.tagging_cooldown:
                # player is still under a cooldown from tagging, advance their cooldown timer, clip at the configured tagging cooldown
                player.tagging_cooldown = self._min(
                    (player.tagging_cooldown + self.tau), self.tagging_cooldown
                )

        self.flag_collision_bool = np.zeros(self.num_agents)

        action_dict = self._to_speed_heading(raw_action_dict)
        if self.render_mode:
            for _i in range(self.num_renders_per_step):
                self._move_agents(action_dict, self.tau / self.num_renders_per_step)
                self._render()
        else:
            self._move_agents(action_dict, self.tau)

        # agent and flag capture checks and more
        self._check_pickup_flags()
        self._check_agent_captures()
        self._check_flag_captures()
        if config_dict_std["teleport_on_tag"] == False:
            self._check_untag()
        self._set_dones()

        if self.message and self.render_mode:
            print(self.message)

        rewards = {agent.id: self.compute_rewards(agent.id) for agent in self.players}
        obs = {agent: self.state_to_obs(agent) for agent in raw_action_dict}
        info = {agent: {"blue": {}, "red": {}} for agent in range(self.num_agents)}

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
        for player in self.players:
            pos_x = player.pos[0]
            pos_y = player.pos[1]
            flag_loc = self.flags[int(player.team)].home

            if player.team == Team.BLUE_TEAM:
                player.on_own_side = pos_x >= self.scrimmage
            else:
                player.on_own_side = pos_x <= self.scrimmage

            # convert desired_speed   and  desired_heading to
            #         desired_thrust  and  desired_rudder
            # requested heading is relative so it directly maps to the heading error
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
            new_speed = min(
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
            if not (
                (self.agent_radius <= pos_x <= self.world_size[0] - self.agent_radius)
                and (
                    self.agent_radius <= pos_y <= self.world_size[1] - self.agent_radius
                )
            ):
                if player.has_flag:
                    # If they have a flag, return the flag to it's home area
                    self.flags[int(not int(player.team))].reset()
                    self.state["flag_taken"][int(not int(player.team))] = 0
                self.state["agent_oob"][player.id] = 1
                if (config_dict_std["teleport_on_tag"]):
                    # print("here")
                    player.reset()
                else:
                    self.state["agent_tagged"][player.id] = 1
                    player.rotate()
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
                and config_dict_std["keepout_bounce"] > 0
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
                vel_ref[
                    1
                ] *= (
                    self.keepout_bounce
                )  # convention is that vector pointing from keepout intersection to flag center is y' axis in new reference frame

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

    def _to_speed_heading(self, action_dict):
        """Returns an array of velocities for each agent."""
        processed_action_dict = OrderedDict()
        for player in self.players:
            if player.id in action_dict:
                speed, heading = ACTION_MAP[action_dict[player.id]]
            else:
                # if no action provided, stop moving
                speed, heading = 0.0, player.heading
            processed_action_dict[player.id] = np.array(
                [speed, heading], dtype=np.float32
            )
        return processed_action_dict

    def _check_pickup_flags(self):
        """Updates player states if they picked up the flag."""
        for player in self.players:
            team = int(player.team)
            other_team = int(not team)
            if not (player.has_flag or self.state["flag_taken"][other_team]) and (not self.state["agent_tagged"][player.id]):
                flag_pos = self.flags[other_team].pos
                distance_to_flag = self.get_distance_between_2_points(
                    player.pos, flag_pos
                )

                if distance_to_flag < self.catch_radius:
                    player.has_flag = True
                    self.state["flag_taken"][other_team] = 1
                    if player.team == Team.BLUE_TEAM:
                        self.blue_team_flag_pickup = True
                    else:
                        self.red_team_flag_pickup = True
                        break
    def _check_untag(self):
        """Untags the player if they return to their own flag."""
        for player in self.players:
            team = int(player.team)
            flag_home = self.flags[team].home
            distance_to_flag = self.get_distance_between_2_points(
                player.pos, flag_home
            )
            if distance_to_flag < self.catch_radius and self.state["agent_tagged"][player.id]:
                self.state["agent_tagged"][player.id] = 0

    def _check_agent_captures(self):
        """Updates player states if they tagged another player."""
        # Reset capture state if teleport_on_tag is true
        if config_dict_std["teleport_on_tag"] == True:
            self.state["agent_tagged"] = [0] * self.num_agents

        self.state["agent_captures"] = [None] * self.num_agents
        for player in self.players:
            # Only continue logic check if player tagged someone if it's on its own side and is untagged.
            if player.on_own_side and (
                player.tagging_cooldown == self.tagging_cooldown
            ) and not self.state["agent_tagged"][player.id]:
                for other_player in self.players:
                    o_team = int(other_player.team)
                    # Only do the rest if the other player is NOT on sides and they are not on the same team.
                    if (
                        not other_player.on_own_side
                        and other_player.team != player.team
                    ):
                        dist_between_agents = self.get_distance_between_2_points(
                            player.pos, other_player.pos
                        )
                        if (
                            dist_between_agents > 0.0
                            and dist_between_agents < self.catch_radius
                        ):
                            self.state["agent_tagged"][other_player.id] = 1
                            self.state["agent_captures"][player.id] = other_player.id
                            # If we get here, then `player` tagged `other_player` and we need to reset `other_player`
                            # Only if config["teleport_on_capture"] == True
                            if config_dict_std["teleport_on_tag"] == True:
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

    def _check_flag_captures(self):
        """Updates states if a player captured a flag."""
        for player in self.players:
            if player.on_own_side and player.has_flag:
                if player.team == Team.BLUE_TEAM:
                    self.blue_team_flag_capture = True
                else:
                    self.red_team_flag_capture = True
                break
        if self.blue_team_flag_capture or self.red_team_flag_capture:
            for flag in self.flags:
                flag.reset()

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
        # set variables from config

        self.world_size = config_dict.get("world_size", config_dict_std["world_size"])
        self.pixel_size = config_dict.get("pixel_size", config_dict_std["pixel_size"])
        self.scrimmage = config_dict.get(
            "scrimmage_line", config_dict_std["scrimmage_line"]
        )
        self.agent_radius = config_dict.get(
            "agent_radius", config_dict_std["agent_radius"]
        )
        self.flag_radius = self.agent_radius  # agent and flag radius will be the same
        self.catch_radius = config_dict.get(
            "catch_radius", config_dict_std["catch_radius"]
        )
        self.flag_keepout = config_dict.get(
            "flag_keepout", config_dict_std["flag_keepout"]
        )
        self.keepout_bounce = config_dict.get(
            "keepout_bounce", config_dict_std["keepout_bounce"]
        )
        self.max_speed = config_dict.get("max_speed", config_dict_std["max_speed"])
        self.own_side_accel = config_dict.get(
            "own_side_accel", config_dict_std["own_side_accel"]
        )
        self.opp_side_accel = config_dict.get(
            "opp_side_accel", config_dict_std["opp_side_accel"]
        )
        self.wall_bounce = config_dict.get(
            "wall_bounce", config_dict_std["wall_bounce"]
        )
        self.tau = config_dict.get("tau", config_dict_std["tau"])
        self.max_time = config_dict.get("max_time", config_dict_std["max_time"])
        self.max_screen_size = config_dict.get(
            "max_screen_size", config_dict_std["max_screen_size"]
        )
        self.random_init = config_dict.get(
            "random_init", config_dict_std["random_init"]
        )
        self.save_traj = config_dict.get("save_traj", config_dict_std["save_traj"])
        self.render_fps = config_dict.get("render_fps", config_dict_std["render_fps"])
        self.normalize = config_dict.get("normalize", config_dict_std["normalize"])
        self.tagging_cooldown = config_dict.get(
            "tagging_cooldown", config_dict_std["tagging_cooldown"]
        )
        # MOOS Dynamics Parameters
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

        if config_dict.get(
            "suppress_numpy_warnings", config_dict_std["suppress_numpy_warnings"]
        ):
            # Suppress numpy warnings to avoid printing out extra stuff to the console
            np.seterr(all="ignore")

        # check that world size (pixels) does not exceed the screen dimensions
        world_screen_err_msg = (
            "Specified world_size {} exceeds the maximum size {} in at least one"
            " dimension".format(self.world_size, self.max_screen_size)
        )
        assert (
            self.world_size[0] * self.pixel_size < self.max_screen_size[0]
            and self.world_size[1] * self.pixel_size < self.max_screen_size[1]
        ), world_screen_err_msg

        # check that world dimensions (pixels) are even
        world_even_err_msg = (
            "Specified world_size {} has at least one dimension that is not even"
            .format(self.world_size)
        )
        assert (self.world_size[0] * self.pixel_size) % 2 == 0 and (
            self.world_size[1] * self.pixel_size
        ) % 2 == 0, world_even_err_msg

        self.screen_size = [self.pixel_size * d for d in self.world_size]

        # check that time between frames (1/render_fps) is not larger than timestep (tau)
        frame_rate_err_msg = (
            "Specified frame rate ({}) creates time intervals between frames larger"
            " than specified timestep ({})".format(self.render_fps, self.tau)
        )
        assert 1 / self.render_fps <= self.tau, frame_rate_err_msg

        self.num_renders_per_step = int(self.render_fps * self.tau)

        # check that flag_keepout is >= flag_radius
        flag_keepout_err_msg = (
            "Specified flag_keepout ({}) is too small, allows for overlap between agent"
            " and flag".format(self.flag_keepout)
        )
        assert (
            self.flag_keepout >= self.flag_radius + self.agent_radius
        ), flag_keepout_err_msg

        # check that agents and flags properly fit within world
        agent_flag_err_msg = (
            "Specified agent_radius ({}), flag_radius ({}), and flag_keepout ({})"
            " create impossible initialization based on world_size({})".format(
                self.agent_radius, self.flag_radius, self.flag_keepout, self.world_size
            )
        )
        horizontal_fit = (
            2 * self.agent_radius + 2 * self.flag_keepout < self.world_size[0] / 2
        )
        vertical_fit = self.flag_keepout + self.agent_radius < self.world_size[1] / 2
        assert horizontal_fit and vertical_fit, agent_flag_err_msg

        # check that keepout_bounce and wall_bounce are between 0 and 1
        bounce_err_msg = (
            "Either keepout_bounce or wall_bounce falls outside of allowed range [0, 1]"
        )
        assert (0 <= self.keepout_bounce <= 1) and (
            0 <= self.wall_bounce <= 1
        ), bounce_err_msg

        # set reference variables for world boundaries
        # ll = lower left, lr = lower right
        # ul = upper left, ur = upper right
        self.boundary_ll = np.array([0.0, 0.0], dtype=np.float32)
        self.boundary_lr = np.array([self.world_size[0], 0.0], dtype=np.float32)
        self.boundary_ul = np.array([0.0, self.world_size[1]])
        self.boundary_ur = np.array(self.world_size, dtype=np.float32)

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
        if self.red_team_flag_capture:
            self.dones["red"] = True
            self.dones["__all__"] = True
            self.message = "Red Wins! Blue Loses"

        elif self.blue_team_flag_capture:
            self.dones["blue"] = True
            self.dones["__all__"] = True
            self.message = "Blue Wins! Red Loses"

        elif self.state["current_time"] >= self.max_time:
            self.dones["__all__"] = True
            self.message = "Game Over. No Winner"

    def update_params(self, agent_id):
        agent = self.players[agent_id]
        self.normalize = False
        obs = self.state_to_obs(agent.id)
        self.normalize = True
        self.params[agent.id]["team"] = agent.team
        self.params[agent.id]["capture_radius"] = self.catch_radius
        self.params[agent.id]["agent_id"] = agent.id
        self.params[agent.id]["agent_oob"] = self.state["agent_oob"]
        if agent.team == Team.RED_TEAM:
            # Game Events
            self.params[agent.id]["num_teammates"] = self.num_red
            self.params[agent.id]["num_opponents"] = self.num_blue
            self.params[agent.id]["team_flag_pickup"] = self.red_team_flag_pickup
            self.params[agent.id]["team_flag_capture"] = self.red_team_flag_capture
            self.params[agent.id]["opponent_flag_pickup"] = self.blue_team_flag_pickup
            self.params[agent.id]["opponent_flag_capture"] = self.blue_team_flag_capture
            # Elements
            self.params[agent.id]["team_flag_bearing"] = obs["protect_flag_bearing"]
            self.params[agent.id]["team_flag_distance"] = obs["protect_flag_distance"]
            self.params[agent.id]["opponent_flag_bearing"] = obs[
                "retrieve_flag_bearing"
            ]
            self.params[agent.id]["opponent_flag_distance"] = obs[
                "retrieve_flag_distance"
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
            self.params[agent.id]["team_flag_bearing"] = obs["protect_flag_bearing"]
            self.params[agent.id]["team_flag_distance"] = obs["protect_flag_distance"]
            self.params[agent.id]["opponent_flag_bearing"] = obs[
                "retrieve_flag_bearing"
            ]
            self.params[agent.id]["opponent_flag_distance"] = obs[
                "retrieve_flag_distance"
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
        self.params[agent.id]["agent_captures"] = self.state["agent_captures"]
        self.params[agent.id]["agent_tagged"] = self.state["agent_tagged"]
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
        self.prev_params[agent_id] = self.params[agent_id].copy()
        # Update Params
        self.update_params(agent_id)
        if self.prev_params[agent_id] == {}:
            self.prev_params[agent_id] = self.params[agent_id].copy()
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

        flag_locations = [
            [
                self.world_size[0] - self.world_size[0] / 8,
                self.world_size[1] / 2,
            ],  # Blue Team
            [self.world_size[0] / 8, self.world_size[1] / 2],  # Red Team
        ]

        for flag in self.flags:
            flag.home = flag_locations[int(flag.team)]
            flag.reset()

        self.dones = self._reset_dones()

        agent_positions, agent_spd_hdg, agent_on_sides = self._generate_agent_starts(
            np.array(flag_locations)
        )

        self.state = {
            "agent_position": agent_positions,
            "prev_agent_position": copy.deepcopy(agent_positions),
            "agent_spd_hdg": agent_spd_hdg,
            "agent_has_flag": np.zeros(self.num_agents),
            "agent_on_sides": agent_on_sides,
            "flag_home": copy.deepcopy(flag_locations),
            "flag_locations": flag_locations,
            "flag_taken": np.zeros(2),
            "current_time": 0.0,
            "agent_captures": [
                None
            ] * self.num_agents,  # whether this agent tagged something
            "agent_tagged": [0] * self.num_agents,  # if this agent was tagged
            "agent_oob": [0] * self.num_agents,  # if this agent went out of bounds
        }

        self.blue_team_flag_pickup = False
        self.red_team_flag_pickup = False
        self.blue_team_flag_capture = False
        self.red_team_flag_capture = False
        self.message = ""
        self.current_time = 0
        self.reset_count += 1
        reset_obs = {agent.id: self.state_to_obs(agent.id) for agent in self.players}

        if self.render_mode:
            self._render()
        return reset_obs

    def _generate_agent_starts(self, flag_locations):
        """
        Generates starting positions for all players based on flag locations.

        If `random_init` is `True`, then agent positons are generated randomly using their home
        flag and random offsets.

        If `random_init` is `False`, then agent positions are programmatically generated using the formula
        init_x = world_size[x] / 4
        init_y = (world_size[y] / (team_size + 1)) * ((player_id % team_size) + 1)

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

        if self.random_init:
            flags_separation = self.get_distance_between_2_points(
                flag_locations[0], flag_locations[1]
            )

        for player in self.players:
            player.thrust = 0.0
            player.speed = 0.0
            player.has_flag = False
            player.on_sides = True
            player.tagging_cooldown = self.tagging_cooldown
            if self.random_init:
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
                    init_x_pos = self.world_size[0] / 4
                    player.heading = 90
                else:
                    init_x_pos = self.world_size[0] - self.world_size[0] / 4
                    player.heading = -90

                init_y_pos = (self.world_size[1] / (self.team_size + 1)) * (
                    (player.id % self.team_size) + 1
                )
                player.pos = [init_x_pos, init_y_pos]
                player.prev_pos = copy.deepcopy(player.pos)
            player.home = copy.deepcopy(player.pos)
            agent_locations.append(player.pos)
            agent_spd_hdg.append([player.speed, player.heading])
            agent_on_sides.append(True)

        return agent_locations, agent_spd_hdg, agent_on_sides

    def _get_dists_to_boundary(self):
        """
        Returns a list of numbers of length self.num_agents
        where each number is the corresponding agents min distance to a boundary wall.
        """
        distances_to_walls = defaultdict(dict)
        for player in self.players:
            i = player.id
            x_pos = player.pos[0]
            y_pos = player.pos[1]

            distances_to_walls[i]["left_dist"] = x_pos
            distances_to_walls[i]["right_dist"] = self.world_size[0] - x_pos
            distances_to_walls[i]["bottom_dist"] = y_pos
            distances_to_walls[i]["top_dist"] = self.world_size[1] - y_pos

        return distances_to_walls

    def _get_dists_between_agents(self):
        """Returns dictionary of distances between agents indexed by agent numbers."""
        agt_to_agt_vecs = {}

        for player in self.players:
            i = player.id
            agt_to_agt_vecs[i] = {}
            i_pos = player.pos
            for other_player in self.players:
                j = other_player.id
                j_pos = other_player.pos
                agt_to_agt_vecs[i][j] = [j_pos[0] - i_pos[0], j_pos[1] - i_pos[1]]

        return agt_to_agt_vecs

    def _get_dist_to_flags(self):
        """Returns a dictionary mapping observation keys to 2d arrays."""
        flag_vecs = {}

        for player in self.players:
            flag_vecs[player.id] = {}
            team_idx = int(player.team)
            i_pos = player.pos
            proFlag_pos = self.flags[team_idx].pos
            retFlag_pos = self.flags[int(not team_idx)].pos
            flag_vecs[player.id]["protect_flag_dist"] = [
                proFlag_pos[0] - i_pos[0],
                proFlag_pos[1] - i_pos[1],
            ]
            flag_vecs[player.id]["retrieve_flag_dist"] = [
                retFlag_pos[0] - i_pos[0],
                retFlag_pos[1] - i_pos[1],
            ]

        return flag_vecs

    def _register_state_elements(self):
        """Initializes the normalizer."""
        agent_obs_normalizer = ObsNormalizer(True)
        max_bearing = [180]
        max_dist = [np.linalg.norm(self.world_size) + 10]  # add a ten meter buffer
        min_dist = [0.0]
        max_bool, min_bool = [1.0], [0.0]
        max_speed, min_speed = [self.max_speed], [0.0]
        agent_obs_normalizer.register("retrieve_flag_bearing", max_bearing)
        agent_obs_normalizer.register("retrieve_flag_distance", max_dist, min_dist)
        agent_obs_normalizer.register("protect_flag_bearing", max_bearing)
        agent_obs_normalizer.register("protect_flag_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_0_bearing", max_bearing)
        agent_obs_normalizer.register("wall_0_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_1_bearing", max_bearing)
        agent_obs_normalizer.register("wall_1_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_2_bearing", max_bearing)
        agent_obs_normalizer.register("wall_2_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_3_bearing", max_bearing)
        agent_obs_normalizer.register("wall_3_distance", max_dist, min_dist)
        agent_obs_normalizer.register("speed", max_speed, min_speed)
        agent_obs_normalizer.register("has_flag", max_bool, min_bool)
        agent_obs_normalizer.register("on_side", max_bool, min_bool)
        agent_obs_normalizer.register(
            "tagging_cooldown", [self.tagging_cooldown], [0.0]
        )
        agent_obs_normalizer.register("is_tagged", max_bool, min_bool)
        assert len(self.agents_of_team[Team.BLUE_TEAM]) == len(
            self.agents_of_team[Team.RED_TEAM]
        )
        num_on_team = len(self.agents_of_team[Team.BLUE_TEAM])

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

        return agent_obs_normalizer

    def state_to_obs(self, agent_id):
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
            Retrieve flag relative bearing (clockwise degrees)
            Retrieve flag distance (meters)
            Home flag relative bearing (clockwise degrees)
            Home flag distance (meters)
            Wall 1 relative bearing (clockwise degrees)
            Wall 1 distance (meters)
            Wall 2 relative bearing (clockwise degrees)
            Wall 2 distance (meters)
            Wall 3 relative bearing (clockwise degrees)
            Wall 3 distance (meters)
            Wall 4 relative bearing (clockwise degrees)
            Wall 4 distance (meters)
            Own speed (meters per second)
            Own flag status (boolean)
            On side (boolean)
            Is tagged (boolean)
            Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
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
        Developer Note 2: changes here should be reflected in register_state_elements in base_policies.py
        """
        agent = self.players[agent_id]
        obs_dict = OrderedDict()
        own_team = agent.team
        protect_flag_loc = self.flags[int(own_team)].pos
        retrieve_flag_loc = self.flags[not int(own_team)].pos
        other_team = Team.BLUE_TEAM if own_team == Team.RED_TEAM else Team.RED_TEAM
        obs = OrderedDict()
        np_pos = np.array(agent.pos, dtype=np.float32)
        # Goal flag
        retrieve_flag_dist, retrieve_flag_bearing = mag_bearing_to(
            np_pos, retrieve_flag_loc, agent.heading
        )
        # Defend flag
        protect_flag_dist, protect_flag_bearing = mag_bearing_to(
            np_pos, protect_flag_loc, agent.heading
        )

        # TODO: consider swapping goal location once flag is retrieved
        #       especially if we're bringing the flag all the way back

        obs["retrieve_flag_bearing"] = retrieve_flag_bearing
        obs["retrieve_flag_distance"] = retrieve_flag_dist
        obs["protect_flag_bearing"] = protect_flag_bearing
        obs["protect_flag_distance"] = protect_flag_dist

        # Walls
        wall_0_closest_point = closest_point_on_line(
            self.boundary_ul, self.boundary_ur, np_pos
        )
        wall_0_dist, wall_0_bearing = mag_bearing_to(
            np_pos, wall_0_closest_point, agent.heading
        )
        obs["wall_0_bearing"] = wall_0_bearing
        obs["wall_0_distance"] = wall_0_dist

        wall_1_closest_point = closest_point_on_line(
            self.boundary_ur, self.boundary_lr, np_pos
        )
        wall_1_dist, wall_1_bearing = mag_bearing_to(
            np_pos, wall_1_closest_point, agent.heading
        )
        obs["wall_1_bearing"] = wall_1_bearing
        obs["wall_1_distance"] = wall_1_dist

        wall_2_closest_point = closest_point_on_line(
            self.boundary_lr, self.boundary_ll, np_pos
        )
        wall_2_dist, wall_2_bearing = mag_bearing_to(
            np_pos, wall_2_closest_point, agent.heading
        )
        obs["wall_2_bearing"] = wall_2_bearing
        obs["wall_2_distance"] = wall_2_dist

        wall_3_closest_point = closest_point_on_line(
            self.boundary_ll, self.boundary_ul, np_pos
        )
        wall_3_dist, wall_3_bearing = mag_bearing_to(
            np_pos, wall_3_closest_point, agent.heading
        )
        obs["wall_3_bearing"] = wall_3_bearing
        obs["wall_3_distance"] = wall_3_dist

        # Own speed
        obs["speed"] = agent.speed
        # Own flag status
        obs["has_flag"] = agent.has_flag
        # On side
        obs["on_side"] = agent.on_own_side
        obs["tagging_cooldown"] = agent.tagging_cooldown

        #Is tagged
        obs["is_tagged"] = self.state["agent_tagged"][agent.id]

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
                obs[(entry_name, "is_tagged")] = self.state["agent_tagged"][dif_agent.id]

        obs_dict[agent.id] = obs
        if self.normalize:
            obs_dict[agent.id] = self.agent_obs_normalizer.normalized(
                obs_dict[agent.id]
            )
        return obs_dict[agent.id]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        """Overridden method inherited from `Gym`."""
        if self.normalize:
            agent_obs_space = self.agent_obs_normalizer.normalized_space
        else:
            agent_obs_space = self.agent_obs_normalizer.unnormalized_space
            raise Warning(
                "Unnormalized observation space has not been thoroughly tested"
            )
        return agent_obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id):
        """Overridden method inherited from `Gym`."""
        return Discrete(len(ACTION_MAP))

    def render(self):
        """Overridden method inherited from `Gym`."""
        return self._render()

    def _render(self):
        """
        Overridden method inherited from `Gym`.

        Draws all players/flags/etc on the pygame screen.
        """
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Capture the Flag")
            if self.render_mode:
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (
                        self.arena_width + 2 * self.arena_offset,
                        self.arena_height + 2 * self.arena_offset,
                    )
                )
                self.isopen = True
                self.font = pygame.font.Font(None, 64)
            else:
                raise Exception(
                    "Sorry, render modes other than 'human' are not supported"
                )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state == {}:
            return None

        # arena coordinates
        if self.border_width % 2 == 0:
            top_left = (
                self.arena_offset - self.border_width / 2 - 1,
                self.arena_offset - self.border_width / 2 - 1,
            )
            top_middle = (
                self.arena_width / 2 + self.arena_offset - 1,
                self.arena_offset - 1,
            )
            top_right = (
                self.arena_width + self.arena_offset + self.border_width / 2 - 1,
                self.arena_offset - self.border_width / 2 - 1,
            )

            bottom_left = (
                self.arena_offset - self.border_width / 2 - 1,
                self.arena_height + self.arena_offset + self.border_width / 2 - 1,
            )
            bottom_middle = (
                self.arena_width / 2 + self.arena_offset - 1,
                self.arena_height + self.arena_offset - 1,
            )
            bottom_right = (
                self.arena_width + self.arena_offset + self.border_width / 2 - 1,
                self.arena_height + self.arena_offset + self.border_width / 2 - 1,
            )

        elif self.border_width % 2 != 0:
            top_left = (
                self.arena_offset - self.border_width / 2,
                self.arena_offset - self.border_width / 2,
            )
            top_middle = (self.arena_width / 2 + self.arena_offset, self.arena_offset)
            top_right = (
                self.arena_width + self.arena_offset + self.border_width / 2,
                self.arena_offset - self.border_width / 2,
            )

            bottom_left = (
                self.arena_offset - self.border_width / 2,
                self.arena_height + self.arena_offset + self.border_width / 2,
            )
            bottom_middle = (
                self.arena_width / 2 + self.arena_offset,
                self.arena_height + self.arena_offset,
            )
            bottom_right = (
                self.arena_width + self.arena_offset + self.border_width / 2,
                self.arena_height + self.arena_offset + self.border_width / 2,
            )

        # screen
        self.screen.fill((255, 255, 255))

        # arena border and scrimmage line
        draw.line(self.screen, (0, 0, 0), top_left, top_right, width=self.border_width)
        draw.line(
            self.screen, (0, 0, 0), bottom_left, bottom_right, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_left, bottom_left, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_right, bottom_right, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_middle, bottom_middle, width=self.border_width
        )

        for team in Team:
            flag = self.flags[int(team)]
            teams_players = self.agents_of_team[team]
            color = "blue" if team == Team.BLUE_TEAM else "red"
            if not self.state["flag_taken"][int(team)]:
                # Flag is not captured, draw normally.
                flag_pos_x_m, flag_pos_y_m = flag.pos
                flag_pos_x, flag_pos_y = (
                    self.pixel_size * flag_pos_x_m,
                    self.pixel_size * flag_pos_y_m,
                )
                flag_pos_x += self.arena_offset
                flag_pos_y = (
                    0.5 * self.arena_height
                    - (flag_pos_y - 0.5 * self.arena_height)
                    + self.arena_offset
                )
                flag_pos_screen = (flag_pos_x, flag_pos_y)
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    self.flag_radius * self.pixel_size,
                )
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    (self.flag_keepout - self.agent_radius) * self.pixel_size,
                    width=round(self.agent_radius * self.pixel_size / 20),
                )
            else:
                # Flag is captured so draw a different shape.
                flag_pos_x_m, flag_pos_y_m = flag.pos
                flag_pos_x, flag_pos_y = (
                    self.pixel_size * flag_pos_x_m,
                    self.pixel_size * flag_pos_y_m,
                )
                flag_pos_x += self.arena_offset
                flag_pos_y = (
                    0.5 * self.arena_height
                    - (flag_pos_y - 0.5 * self.arena_height)
                    + self.arena_offset
                )
                flag_pos_screen = (flag_pos_x, flag_pos_y)
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    self.agent_radius * self.pixel_size / 2,
                )

            for player in teams_players:
                orientation = Vector2(list(mag_heading_to_vec(1.0, player.heading)))
                ref_angle = -orientation.angle_to(self.UP)

                # transform position to pygame coordinates
                pos_x_m, pos_y_m = player.pos
                pos_x, pos_y = pos_x_m * self.pixel_size, pos_y_m * self.pixel_size
                pos_x += self.arena_offset
                pos_y = (
                    0.5 * self.arena_height
                    - (pos_y - 0.5 * self.arena_height)
                    + self.arena_offset
                )
                blit_position = np.array([pos_x, pos_y])
                rotated_surface = rotozoom(player.pygame_agent, ref_angle, 1.0)
                rotated_surface_size = np.array(rotated_surface.get_size())
                blit_position -= 0.5 * rotated_surface_size
                self.screen.blit(rotated_surface, blit_position)

        if self.render_mode:
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

    def close(self):
        """Overridden method inherited from `Gym`."""
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
