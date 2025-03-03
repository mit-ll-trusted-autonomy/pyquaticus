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
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.base_policies.rrt.utils import (
    Point,
    intersect,
    intersect_circles,
    get_ungrouped_seglist,
)
from pyquaticus.base_policies.rrt.rrt_star import rrt_star
from pyquaticus.structs import PolygonObstacle, CircleObstacle


from typing import Optional

from multiprocessing.dummy import Pool
from multiprocessing.pool import ThreadPool
from functools import partial


class WaypointPolicy(BaseAgentPolicy):
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(
        self,
        agent_id: str,
        team: Team,
        env: PyQuaticusEnv,
        continuous: bool = False,
        capture_radius: float = 1,
        slip_radius: Optional[float] = None,
        avoid_radius: float = 2,
        wps: list[np.ndarray] = [],
    ):
        super().__init__(agent_id, team, env)

        self.capture_radius = capture_radius

        self.slip_radius = slip_radius

        self.cur_dist = None

        self.avoid_radius = avoid_radius

        self.wps = wps

        self.continuous = continuous

        self.plan_process = Pool(processes=1)

        self.get_env_geom(env)

        self.tree = None

        if team not in Team:
            raise AttributeError(f"Invalid team {team}")

    def get_env_geom(self, env: PyQuaticusEnv):
        poly_obstacles = []
        circle_obstacles = []
        for obstacle in env.obstacles:
            assert isinstance(obstacle, (PolygonObstacle, CircleObstacle))
            if isinstance(obstacle, PolygonObstacle):
                poly = np.array(obstacle.anchor_points).reshape(-1, 2)
                poly_obstacles.append(poly)
            else:
                circle = (*obstacle.center_point, obstacle.radius)
                circle_obstacles.append(circle)

        self.env_bounds = np.array(((0, 0), env.env_size))

        if len(poly_obstacles) == 0:
            poly_obstacles = None
        if len(circle_obstacles) == 0:
            circle_obstacles = None

        self.poly_obstacles = poly_obstacles
        self.circle_obstacles = circle_obstacles
        if self.circle_obstacles is not None:
            self.circles_for_intersect = np.array(circle_obstacles)
        else:
            self.circles_for_intersect = None
        if poly_obstacles is not None:
            self.ungrouped_seglist = get_ungrouped_seglist(poly_obstacles)
        else:
            self.ungrouped_seglist = None

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

        desired_speed = self.max_speed / 2
        heading_error = 0

        self.update_wps(self.pos)

        if self.is_tagged:
            self.wps = []

        if len(self.wps) == 0:
            if self.continuous:
                return 0, 0
            else:
                return -1

        pos_err = self.wps[0] - self.pos

        desired_heading = self.angle180(-1 * self.vec_to_heading(pos_err) + 90)

        heading_error = self.angle180(desired_heading - self.heading)

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

    def update_wps(self, pos: np.ndarray):
        """
        Remove the current waypoint from the list of waypoints, if necessary.
        """

        if len(self.wps) == 0:
            return

        # Check if any future waypoints have an obstacle-free path
        # If so, skip directly to the closest one to the goal
        wps = np.array(self.wps).reshape((-1, 1, 2))
        pos_array = np.repeat(pos.reshape((-1, 2)), len(self.wps), 0).reshape(-1, 1, 2)
        segs = np.concatenate((pos_array, wps), axis=1)
        clear = np.logical_not(
            intersect(segs, self.ungrouped_seglist, radius=self.avoid_radius)
        )
        clear = np.logical_and(
            clear,
            np.logical_not(
                intersect_circles(
                    segs, self.circles_for_intersect, agent_radius=self.avoid_radius
                )
            ),
        )
        try:
            last_free_wp_index = len(list(clear)) - 1 - list(clear)[::-1].index(True)
        except ValueError:
            # No points are obstacle-free, even the current waypoint
            # TODO: Continue to current waypoint? Replan?
            pass
        else:
            if last_free_wp_index > 0:
                self.wps = self.wps[last_free_wp_index:]
                self.cur_dist = None
                return

        new_dist = np.linalg.norm(self.wps[0] - pos)

        if (new_dist <= self.capture_radius) and clear[0]:
            self.wps.pop(0)
            self.cur_dist = None

        elif (
            (self.slip_radius is not None)
            and (self.cur_dist is not None)
            and (new_dist > self.cur_dist)
            and (new_dist <= self.slip_radius)
            and clear[0]
        ):
            self.wps.pop(0)
            self.cur_dist = None

        else:
            self.cur_dist = new_dist

    def set_wps(self, wps: list[np.ndarray]):
        self.wps = wps

    def plan(
        self,
        wp: np.ndarray,
        poly_obstacles: Optional[list[np.ndarray]] = None,
        circle_obstacles: Optional[list[tuple[float, float, float]]] = None,
        area: Optional[np.ndarray] = None,
        max_step_size: Optional[float] = None,
        num_iters: int = 1000,
        timeout: float = 10,
    ):
        """
        Asynchronously run RRT* from the agent's current position, and update the waypoints if a valid path to the goal was found
        """

        # Use obstacles, area, and step size from environment if not provided

        if poly_obstacles is None:
            poly_obstacles = self.poly_obstacles

        if circle_obstacles is None:
            circle_obstacles = self.circle_obstacles

        if area is None:
            area = self.env_bounds

        if max_step_size is None:
            max_step_size = np.max(self.env_bounds[1] - self.env_bounds[0]) / 10

        # Run RRT in a separate process

        kwargs = dict(
            start=self.pos,
            goal=wp,
            poly_obstacles=poly_obstacles,
            circle_obstacles=circle_obstacles,
            area=area,
            max_step_size=max_step_size,
            num_iters=num_iters,
            agent_radius=self.avoid_radius,
        )

        assert isinstance(self.plan_process, ThreadPool)

        self.plan_process.apply_async(
            rrt_star, kwds=kwargs, callback=partial(self.get_path, wp=wp)
        )

    def get_path(self, tree: list[Point], wp: np.ndarray):
        """
        Given a tree, search the tree for points near the goal and create a list of waypoints
        that determine the shortest path to the goal from the starting point
        """

        possible_points = []

        # Find all points that satisfy the goal
        for point in tree:
            if np.linalg.norm(point.pos - wp) <= self.capture_radius:
                possible_points.append(point)

        # Check for no possible paths
        if len(possible_points) == 0:
            print("No path found.")
            self.tree = tree
            return

        # Find the satisfying point with the minimum cost
        min_point = possible_points[0]
        assert isinstance(min_point, Point)
        min_cost = possible_points[0].cost

        for point in possible_points:
            assert isinstance(point, Point)
            if point.cost < min_cost:
                min_cost = point.cost
                min_point = point

        wps = [min_point.pos]

        # Trace the path back to the root of the tree (the agent's current position)
        while min_point.parent is not None:
            wps.insert(0, min_point.parent.pos)
            min_point = min_point.parent

        self.tree = tree
        self.wps = wps
