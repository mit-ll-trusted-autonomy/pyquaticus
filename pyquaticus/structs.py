import copy
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pygame import SRCALPHA, Surface, draw
from typing import Hashable

from pyquaticus.utils.utils import angle180

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
        config_dict: the configuration dictionary
        thrust: The engine thrust
        pos: The position of the agent [x, y]
        speed: The speed of the agent (m / s)
        heading: The heading of the agent (deg), maritime convention: north is 0, east is 90
        pygame_agent: The pygame object that is drawn on screen.
        prev_pos: The previous position of the agent
        has_flag: Indicator for whether or not the agent has the flag
        on_own_side: Indicator for whether or not the agent is on its own side of the field.
    """

    id: Hashable
    team: Team
    r: float
    config_dict: dict
    thrust: float = field(init=False, default_factory=float)
    pos: list[float] = field(init=False, default_factory=list)
    speed: float = field(init=False, default_factory=float)
    heading: float = field(init=False, default_factory=float)
    pygame_agent: Surface = field(init=False, default=None)
    prev_pos: list[float] = field(init=False, default_factory=list)
    has_flag: bool = field(init=False, default=False)
    on_own_side: bool = field(init=False, default=True)
    tagging_cooldown: float = field(init=False)
    is_tagged: bool = field(init = False, default=False)
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
        self.is_tagged = False
        self.has_flag = False
        self.on_sides = True

    def rotate(self, angle=180):
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
        elif(self.config_dict["world_size"][0] - self.r < x_pos):
            self.pos[0] -= 1
        
        if (y_pos < self.r):
            self.pos[1] += 1
        elif(self.config_dict["world_size"][1] - self.r < y_pos):
            self.pos[1] -= 1
        
        # Rotate 180 degrees
        self.heading = angle180(self.heading + angle)


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
