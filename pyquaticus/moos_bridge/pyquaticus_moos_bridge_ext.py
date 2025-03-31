from collections import OrderedDict

from pyquaticus.moos.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.config import config_dict_std


class PyQuaticusMoosBridgeFullObs(PyQuaticusMoosBridge):
    """
    This class is identical to the PyQuaticusMoosBridge except it
    provides observations for every agent on the field at each step.

    It also includes the unnormalized observations in the info.
    """
    def __init__(self, server, agent_name, agent_port, team_names, opponent_names, moos_config, \
                 quiet=True, team=None, timewarp=None, tagging_cooldown=config_dict_std["tagging_cooldown"],
                 normalize=True):
        """
        Args:
            server: server for the MOOS process
            name: the name of the agent
            port: the MOOS port for this agent
            team_names: list of names of team members
            opponent_names: list of names of opponents
            moos_config: one of the objects from pyquaticus.moos.config
            quiet: tell the pymoos comms object to be quiet
            team: which team this agent is (will infer based on name if not passed)
            timewarp: specify the moos timewarp (IMPORTANT for messages to send correctly)
                      uses moos_config default if not passed
        """
        super().__init__(server, agent_name, agent_port, team_names, opponent_names, moos_config,
                         quiet, team, timewarp, tagging_cooldown, normalize)

    def reset(self):
        super().reset()
        unnormalized_obs = OrderedDict((n, self.state_to_obs(n, False)) for n in self.players)
        normalized_obs = OrderedDict()
        for n, obs in unnormalized_obs.items():
            normalized_obs[n] = self.agent_obs_normalizer.normalized(obs)

        return normalized_obs, unnormalized_obs

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)

        unnormalized_obs = OrderedDict((n, self.state_to_obs(n, False)) for n in self.players)
        normalized_obs = OrderedDict()
        for n, obs in unnormalized_obs.items():
            normalized_obs[n] = self.agent_obs_normalizer.normalized(obs)

        info['unnormalized_obs'] = unnormalized_obs
        return normalized_obs, reward, terminated, truncated, info