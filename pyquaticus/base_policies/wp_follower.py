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

from typing import Union


class WaypointFollower(BaseAgentPolicy):
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(
        self,
        agent_id: int,
        team: Team,
        teammate_ids: Union[list[int], int, None],
        opponent_ids: Union[list[int], int, None],
        continuous: bool = False,
        capture_radius: float = 1,
        wps: list[np.ndarray] = [],
    ):
        super().__init__(agent_id, team, teammate_ids, opponent_ids)

        self.capture_radius = capture_radius

        self.wps = wps

        self.continuous = continuous

        if team not in Team:
            raise AttributeError(f"Invalid team {team}")

    def compute_action(self, global_state):
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
        global_state = self.update_state(global_state)

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
