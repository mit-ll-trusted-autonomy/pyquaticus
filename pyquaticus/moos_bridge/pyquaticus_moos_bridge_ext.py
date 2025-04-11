from pyquaticus.moos_bridge.config import FieldReaderConfig, pyquaticus_config_std
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge


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

        unnormalized_obs = OrderedDict((n, self.state_to_obs(n, False)) for n in self.players)
        normalized_obs = OrderedDict()
        for n, obs in unnormalized_obs.items():
            normalized_obs[n] = self.agent_obs_normalizer.normalized(obs)

        return obs, info #normalized_obs, unnormalized_obs

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)

        unnormalized_obs = OrderedDict((n, self.state_to_obs(n, False)) for n in self.players)
        normalized_obs = OrderedDict()
        for n, obs in unnormalized_obs.items():
            normalized_obs[n] = self.agent_obs_normalizer.normalized(obs)

        info['unnormalized_obs'] = unnormalized_obs
        return normalized_obs, reward, terminated, truncated, info