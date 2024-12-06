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
from pyquaticus.envs.pyquaticus import config_dict_std, Team

from typing import Union


class WaypointFollowerContinuous(BaseAgentPolicy):
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(
        self,
        agent_id: int,
        teammate_ids: Union[list[int], int, None],
        opponent_ids: Union[list[int], int, None],
        team=Team.RED_TEAM,
        capture_radius: float = 20,
        wps: list[np.ndarray] = [],
    ):
        super().__init__(agent_id, team, teammate_ids, opponent_ids)

        self.capture_radius = capture_radius

        self.wps = wps

        if team not in Team:
            raise AttributeError(f"Invalid team {team}")

    def compute_action(self, obs):
        """
        **THIS FUNCTION REQUIRES UNNORMALIZED OBSERVATIONS**.

        Compute an action for the given position. This function uses observations
        of both teams.

        Args:
            obs: Unnormalized observation from the gym

        Returns
        -------
            desired_speed: m/s
            heading_error: deg
        """
        my_obs = self.update_state(obs)

        # Some big speed hard-coded so that every agent drives at max speed
        desired_speed = 50
        heading_error = 0

        pos, heading = self.get_pos_heading(my_obs)

        self.update_wps(pos)

        if len(self.wps) == 0:
            return 0, 0

        pos_err = self.wps[0] - pos

        desired_heading = self.angle180(-1 * self.vec_to_heading(pos_err) + 90)

        heading_error = self.angle180(desired_heading - heading)

        return 4, heading_error

    def update_wps(self, pos: np.ndarray):

        if len(self.wps) == 0:
            return
        elif np.linalg.norm(self.wps[0] - pos) <= self.capture_radius:
            self.wps.pop(0)

    def get_pos_heading(self, my_obs):
        if self.team == Team.RED_TEAM:
            x = my_obs["wall_1_distance"]
            y = my_obs["wall_0_distance"]
            heading = -my_obs["wall_2_bearing"]
        else:
            x = my_obs["wall_3_distance"]
            y = my_obs["wall_2_distance"]
            heading = -my_obs["wall_0_bearing"]

        return np.array([x, y]), heading
