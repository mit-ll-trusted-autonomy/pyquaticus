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

from pyquaticus.base_policies.rrt.utils import Point
from pyquaticus.base_policies.rrt.rrt_star import rrt_star
from pyquaticus.structs import PolygonObstacle, CircleObstacle


from typing import Optional

from multiprocessing.dummy import Pool
from multiprocessing.pool import ThreadPool

from functools import partial


class EnvWaypointPolicy:
    """Lightweight version of WaypointPolicy for use inside Pyquaticus environment only."""

    def __init__(
        self,
        obstacles: list,
        env_size,
        max_speed: float,
        capture_radius: float = 1,
        slip_radius: Optional[float] = None,
        avoid_radius: float = 2,
    ):

        self.capture_radius = capture_radius

        self.slip_radius = slip_radius

        self.cur_dist = None

        self.avoid_radius = avoid_radius

        self.wps = []

        self.plan_process = Pool(processes=1)

        self.poly_obstacles, self.circle_obstacles = self.get_env_geom(obstacles)

        self.env_bounds = np.array(((0, 0), env_size))

        self.tree = None

        self.max_speed = max_speed

        self.planning = False

    def get_env_geom(
        self, obstacles: list
    ) -> tuple[Optional[list[np.ndarray]], Optional[list[tuple[float, float, float]]]]:
        final_obstacles = []
        circle_obstacles = []
        for obstacle in obstacles:
            assert isinstance(obstacle, (PolygonObstacle, CircleObstacle))
            if isinstance(obstacle, PolygonObstacle):
                poly = np.array(obstacle.anchor_points).reshape(-1, 2)
                final_obstacles.append(poly)
            else:
                circle = (*obstacle.center_point, obstacle.radius)
                circle_obstacles.append(circle)

        if len(final_obstacles) == 0:
            final_obstacles = None
        if len(circle_obstacles) == 0:
            circle_obstacles = None

        return final_obstacles, circle_obstacles

    def compute_action(self, pos: np.ndarray, heading: float):
        """
        Compute an action for the given position. This function uses observations
        of both teams.

        Args:
            obs: Unnormalized observation from the gym

        Returns
        -------
            desired_speed: m/s
            heading_error: deg
        """
        desired_speed = self.max_speed
        heading_error = 0

        self.update_wps(pos)

        if len(self.wps) == 0:
            return 0, 0

        pos_err = self.wps[0] - pos

        desired_heading = self.angle180(-1 * self.vec_to_heading(pos_err) + 90)

        heading_error = self.angle180(desired_heading - heading)

        if np.isnan(heading_error):
            heading_error = 0

        if np.abs(heading_error) < 5:
            heading_error = 0

        return (desired_speed, heading_error)

    def update_wps(self, pos: np.ndarray):

        if len(self.wps) == 0:
            return

        new_dist = np.linalg.norm(self.wps[0] - pos)
        if new_dist <= self.capture_radius:
            self.wps.pop(0)
            self.cur_dist = None
        elif (
            (self.slip_radius is not None)
            and (self.cur_dist is not None)
            and (new_dist > self.cur_dist)
            and (new_dist <= self.slip_radius)
        ):
            self.wps.pop(0)
            self.cur_dist = None
        else:
            self.cur_dist = new_dist

    def set_wps(self, wps: list[np.ndarray]):
        self.wps = wps

    def plan(
        self,
        pos: np.ndarray,
        wp: np.ndarray,
        poly_obstacles: Optional[list[np.ndarray]] = None,
        circle_obstacles: Optional[list[tuple[float, float, float]]] = None,
        area: Optional[np.ndarray] = None,
        max_step_size: Optional[float] = None,
        num_iters: int = 400,
    ):
        """
        Asynchronously run RRT* from the agent's current position, and update the waypoints if a valid path to the goal was found
        """

        self.planning = True

        if poly_obstacles is None:
            poly_obstacles = self.poly_obstacles

        if circle_obstacles is None:
            circle_obstacles = self.circle_obstacles

        if area is None:
            area = self.env_bounds

        if max_step_size is None:
            max_step_size = np.max(self.env_bounds[1] - self.env_bounds[0]) / 10

        kwargs = dict(
            start=pos,
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

        self.planning = False

        possible_points = []

        # Find all points that satisfy the goal
        for point in tree:
            if np.linalg.norm(point.pos - wp) <= self.capture_radius:
                possible_points.append(point)

        if len(possible_points) == 0:
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

        # Trace the path back to the root of the tree
        while min_point.parent is not None:
            wps.insert(0, min_point.parent.pos)
            min_point = min_point.parent

        self.tree = tree

        self.wps = wps

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
