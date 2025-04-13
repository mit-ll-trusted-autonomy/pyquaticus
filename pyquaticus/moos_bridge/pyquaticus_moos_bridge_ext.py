import numpy as np

from pyquaticus.moos_bridge.config import FieldReaderConfig, pyquaticus_config_std
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from typing import Optional


class PyQuaticusMoosBridgeFullObs(PyQuaticusMoosBridge):
    """
    This class is identical to the PyQuaticusMoosBridge except it
    provides the observations and info for every agent on the field at each step.

    It also includes the unnormalized observations in the info.
    """
    def __init__(
        self,
        server: str,
        agent_name: str,
        agent_port: int,
        team_names: list,
        opponent_names: list,
        all_agent_names: list,
        moos_config: FieldReaderConfig,
        pyquaticus_config: dict = pyquaticus_config_std,
        action_space: str = "continuous",
        team = None,
        quiet = True,
        timewarp = 1
    ):
        """
        Args:
            server: server for the MOOS process (either 'localhost' (if running locally), or vehicle/backseat computer IP address)
            name: the name of the agent
            port: the MOOS port for this agent
            team_names: list of names of team members
            opponent_names: list of names of opponents
            all_agent_names: list of all agents names in the order they should be indexed (keep consistent across all instances of PyQuaticusMoosBridge)
            moos_config: one of the objects from pyquaticus.moos.config
            pyquaticus_config: observation configuration dictionary (see pyquaticus/pyquaticus/moos_bridge/config.py for pyquaticus_config example)
            action_space: 'discrete', 'continuous', or 'afp' action space for this agent (see pyquaticus/pyquaticus/envs/pyquaticus.py for doc)
            quiet: tell the pymoos comms object to be quiet
            team: which team this agent is (will infer based on name if not passed)
            timewarp: specify the moos timewarp (IMPORTANT for messages to send correctly)
        """
        super().__init__(
            server=server,
            agent_name=agent_name,
            agent_port=agent_port,
            team_names=team_names,
            opponent_names=opponent_names,
            all_agent_names=all_agent_names,
            moos_config=moos_config,
            pyquaticus_config=pyquaticus_config,
            action_space=action_space,
            team=team,
            quiet=quiet,
            timewarp=timewarp
        )

    def reset(self, return_info=True, options: Optional[dict] = None):
        """
        Resets variables and (re)connects to MOOS node for the provided agent name.

        Args:
            return_info (boolean): whether or not to return the info dict for the episode (when calling reset and step)
            options (optional): Additonal options for resetting the environment:
                -"normalize_obs": whether or not to normalize observations
                -"normalize_state": whether or not to normalize the global state
                -"unnormalized_obs_info": whether or not to include the unnormalized obs in the info dictionary
        """
        agent_obs, agent_info = super().reset(return_info, options) #agent's observation and info

        # Observation History
        for agent_id in self.agents:
            if agent_id != self._agent_name:
                reset_obs, reset_unnorm_obs = self.state_to_obs(agent_id, self.normalize_obs)
                self.state["obs_hist_buffer"][agent_id] = np.array(self.obs_hist_buffer_len * [reset_obs])

                if self.unnorm_obs_info:
                    self.state["unnorm_obs_hist_buffer"][agent_id] = np.array(self.obs_hist_buffer_len * [reset_unnorm_obs])

        # Current Observation
        obs = {
            agent_id: self._history_to_obs(agent_id, "obs_hist_buffer")
            if agent_id != self._agent_name
            else agent_obs
            for agent_id in self.agents
        }

        # Info
        info = {}
        if self.return_info:
            #global state
            info["global_state"] = agent_info["global_state"]

            #unnormalized obs
            if self.unnorm_obs_info:
                info["unnorm_obs"] = {
                        agent_id: self._history_to_obs(agent_id, "unnorm_obs_hist_buffer")
                        if agent_id != self._agent_name
                        else agent_info["unnorm_obs"]
                        for agent_id in self.agents
                    }

        return obs, info

    def step(self, action):
        agent_obs, reward, terminated, truncated, agent_info = super().step(action) #agent's observation and info

        # Observation History
        for agent_id in self.agents:
            if agent_id != self._agent_name:
                next_obs, next_unnorm_obs = self.state_to_obs(agent_id, self.normalize_obs)

                self.state["obs_hist_buffer"][agent_id][1:] = self.state["obs_hist_buffer"][agent_id][:-1]
                self.state["obs_hist_buffer"][agent_id][0] = next_obs

                if self.unnorm_obs_info:
                    self.state["unnorm_obs_hist_buffer"][agent_id][1:] = self.state["unnorm_obs_hist_buffer"][agent_id][:-1]
                    self.state["unnorm_obs_hist_buffer"][agent_id][0] = next_unnorm_obs

        # Current Observation
        obs = {
            agent_id: self._history_to_obs(agent_id, "obs_hist_buffer")
            if agent_id != self._agent_name
            else agent_obs
            for agent_id in self.agents
        }

        # Info
        info = {}
        if self.return_info:
            #global state
            info["global_state"] = agent_info["global_state"]

            #unnormalized obs
            if self.unnorm_obs_info:
                info["unnorm_obs"] = {
                    agent_id: self._history_to_obs(agent_id, "unnorm_obs_hist_buffer")
                    if agent_id != self._agent_name
                    else agent_info["unnorm_obs"]
                    for agent_id in self.agents
                }

        return obs, reward, terminated, truncated, info