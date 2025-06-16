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

from typing import Any, Union

import numpy as np

from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge


class BaseAgentPolicy:
    """
    Parent class for all base policies.
    """

    def __init__(
        self,
        agent_id: str,
        team: Team,
        env: Union[PyQuaticusEnv, PyQuaticusMoosBridge],
        suppress_numpy_warnings=True,
    ):
        self.id = agent_id

        if isinstance(team, str):
            if team == "red":
                team = Team.RED_TEAM
            elif team == "blue":
                team = Team.BLUE_TEAM
            else:
                raise ValueError(f"Got unknown team: {team}")
        self.team = team

        if team == Team.BLUE_TEAM:
            self.teammate_ids = env.agent_ids_of_team[Team.BLUE_TEAM]
            self.opponent_ids = env.agent_ids_of_team[Team.RED_TEAM]
        else:
            self.teammate_ids = env.agent_ids_of_team[Team.RED_TEAM]
            self.opponent_ids = env.agent_ids_of_team[Team.BLUE_TEAM]

        if suppress_numpy_warnings:
            np.seterr(all="ignore")

    def compute_action(self, obs, info: dict[str, dict]) -> Any:
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and relative bearing.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """
        raise NotImplementedError

    def update_state(self, obs, info: dict[str, dict]) -> None:
        """
        Method to convert the gym obs and info into data useful for compute_action

        Args:
            obs: observation from gym
            info: info from gym
        """
        raise NotImplementedError
