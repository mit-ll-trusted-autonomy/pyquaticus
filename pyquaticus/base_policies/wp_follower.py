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
from pyquaticus.utils.obs_utils import ObsNormalizer
from pyquaticus.base_policies.rrt.utils import Point
from pyquaticus.base_policies.rrt.rrt_star import rrt_star


from typing import Union

from multiprocessing.dummy import Pool
from functools import partial

class WaypointFollower(BaseAgentPolicy):
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(
        self,
        agent_id: int,
        team: Team,
        teammate_ids: Union[list[int], int, None],
        opponent_ids: Union[list[int], int, None],
        obs_normalizer: ObsNormalizer,
        state_normalizer: ObsNormalizer,
        continuous: bool = False,
        capture_radius: float = 1,
        agent_radius: float = 2,
        wps: list[np.ndarray] = [],
    ):
        super().__init__(agent_id, team, teammate_ids, opponent_ids, obs_normalizer, state_normalizer)

        self.capture_radius = capture_radius

        self.agent_radius = agent_radius

        self.wps = wps

        self.continuous = continuous

        self.plan_process = Pool(processes=1)

        if team not in Team:
            raise AttributeError(f"Invalid team {team}")

    def compute_action(self, obs, info):
        """
        **THIS FUNCTION REQUIRES UNNORMALIZED GLOBAL STATE**.

        Compute an action for the given position. This function uses observations
        of both teams.

        Args:
            obs: Unnormalized observation from the gym

        Returns
        -------
            desired_speed: m/s
            heading_error: deg
        """
        self.update_state(obs, info)

        # Some big speed hard-coded so that every agent drives at max speed
        desired_speed = 50
        heading_error = 0

        self.update_wps(self.pos)

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

        if len(self.wps) == 0:
            return
        elif np.linalg.norm(self.wps[0] - pos) <= self.capture_radius:
            self.wps.pop(0)

    def set_wps(self, wps: list[np.ndarray]):
        self.wps = wps

    def plan(self, wp: np.ndarray, obstacles: Union[list[np.ndarray], None], area: np.ndarray, max_step_size: float = 2, num_iters: int = 1000):
        kwargs=dict(start=self.pos, obstacles=obstacles, area=area, max_step_size=max_step_size, num_iters=num_iters, agent_radius=self.agent_radius)
        tree = self.plan_process.apply_async(rrt_star, kwds=kwargs, callback=partial(self.get_path, wp=wp))
        
        
    def get_path(self, tree: list[Point], wp: np.ndarray):
        possible_points = []
        for point in tree:
            if np.linalg.norm(point.pos - wp) <= self.capture_radius:
                possible_points.append(point)
        if len(possible_points) == 0:
            return
        min_cost = possible_points[0].cost
        min_point = possible_points[0]
        for point in possible_points:
            if point.cost < min_cost:
                min_cost = point.cost
                min_point = point
        wps = [min_point.pos]
        while min_point.parent is not None:
            wps.insert(0, min_point.parent.pos)
            min_point = min_point.parent

        self.wps = wps
