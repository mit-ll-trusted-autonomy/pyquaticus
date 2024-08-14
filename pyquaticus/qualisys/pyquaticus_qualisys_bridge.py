import itertools
import numpy as np
import time

from pyquaticus.envs.pyquaticus import PyQuaticusEnv, PyQuaticusEnvBase
from pyquaticus.structs import Player, Team, Flag
from pyquaticus.config import config_dict_std
from pyquaticus.utils.utils import mag_bearing_to
from pyquaticus.qualisys.qualisys_comms

class PyQuaticusQualisysBridge(PyQuaticusEnv):
    """
    This class is used to control an agent in MOOS Aquaticus.
    It does *not* start anything on the MOOS side. Instead, you start everything you want
    and then run this and pass actions (for example from a policy learned in Pyquaticus)
    once you're ready to run the agent.

    Important differences from pyquaticus_team_env_bridge:
    * This class only connects to a single MOOS node, not all of them
        -- this choice makes it much more efficient
    * This class only controls a single agent
        -- run multiple instances (e.g., one per docker) to run multiple agents
    """
    def __init__(self, 
            team_size: int=1,
            reward_config: dict=None,
            config_dict=config_dict_std,
            render_mode: Optional[str] = 'human',
            render_agent_ids: Optional[bool]=True):
        """
        Args:
            server: server for the qualisys connection
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
        super().__init__(team_size=team_size, reward_config=reward_config, config_dict=config_dict_std, render_mode=render_mode, render_agent_ids=render_agent_ids)
        
        self.qtm_server = QualisysComms(,self.config_dict['qualisys_mapper']) 
        # Note: not using _ convention to match pyquaticus

        self.game_score = {'blue_captures':0, 'red_captures':0}

        self.qtm_server.connect()
        

    def reset(self, seed=None, return_info=False, options: Optional[dict]=None):
        """
        Sets up the players and resets variables.
        (Re)connects to MOOS node for the provided agent name.
        """
        return super.reset(seed=seed, return_info=return_info, options=options)#self.state_to_obs(self._agent_name), {}

    #def render(self, mode="human"):
        """
        This is a pass through, all rendering is handled by pMarineViewer on the MOOS side.
        """
        #pass
        #super().render()

    def close(self):
        """
        Close the connection to MOOS
        """
        self.qtm_server.close()
        super().close()

    def step(self, action):
        """
        Take a single action and sleep until it has taken effect.

        Args:
            action: a discrete action that aligns with Pyquaticus
                    see pyquaticus.config.ACTION_MAP for the specific mapping

        Returns (aligns with Gymnasium interface):
            obs: a normalized observation space (this can be "unnormalized" via self.agent_obs_normalizer.unnormalized(obs))
            reward: this always returns -1 for now -- only using this env for deploying, not training
            terminated: always False (runs until you stop)
            truncated: always False (runs until you stop)
            info: additional information
        """
        #Update Player Positions
        print("Positions: ", self.qtm_server.get_positions)
        #player = self.players[self._agent_name]
        
        return obs, reward, terminated, truncated, {}

'''    def set_config(self, moos_config):
        """
        Reads a configuration object to set the field and other configuration variables
        See the default moos_config value in the constructor (__init__)
        """
        self._moos_config = moos_config
        self._blue_flag = np.asarray(self._moos_config.blue_flag, dtype=np.float32)
        self._red_flag = np.asarray(self._moos_config.red_flag, dtype=np.float32)

        self.flags = []
        for team in Team:
            flag = Flag(team)
            if team == Team.BLUE_TEAM:
                flag.home = self._blue_flag
                flag.pos = self._blue_flag
            else:
                assert team == Team.RED_TEAM
                flag.home = self._red_flag
                flag.pos = self._red_flag
            self.flags.append(flag)

        self.scrimmage_pnts = np.asarray(self._moos_config.scrimmage_pnts, dtype=np.float32)
        # save the lower and upper point (used to determine distance to scrimmage line)
        self.scrimmage_l = self.scrimmage_pnts[0]
        self.scrimmage_u = self.scrimmage_pnts[1]
        # define function for checking which side an agent is on
        if abs(self.scrimmage_pnts[0][0] - self.scrimmage_pnts[1][0]) < 1e-2:
            # Vertical scrimmage line
            if self._red_flag[0] > self.scrimmage_pnts[0][0]:
                def check_side(agent):
                    x, y = agent.pos
                    if agent.team == Team.RED_TEAM:
                        return x > self.scrimmage_pnts[0][0]
                    else:
                        return x < self.scrimmage_pnts[0][0]
            else:
                def check_side(agent):
                    x, y = agent.pos
                    if agent.team == Team.RED_TEAM:
                        return x < self.scrimmage_pnts[0][0]
                    else:
                        return x > self.scrimmage_pnts[0][0]

        elif abs(self.scrimmage_pnts[0][1] - self.scrimmage_pnts[1][1]) < 1e-2:
            raise RuntimeError('Horizontal scrimmage lines not yet supported')
        else:
            m = (self.scrimmage_pnts[1][1] - self.scrimmage_pnts[0][1]) / (self.scrimmage_pnts[1][0] - self.scrimmage_pnts[0][0])
            b = self.scrimmage_pnts[0][1] - m*self.scrimmage_pnts[0][0]

            if self._red_flag[1] > m*self._red_flag[0] + b:
                def check_side(agent):
                    x, y = agent.pos
                    if agent.team == Team.RED_TEAM:
                        return y > m*x + b
                    else:
                        return y < m*x + b
            else:
                def check_side(agent):
                    x, y = agent.pos
                    if agent.team == Team.RED_TEAM:
                        return y < m*x + b
                    else:
                        return y > m*x + b

        self._check_on_side = check_side

        # The operating boundary is defined in shoreside/meta_shoreside.moos
        self.boundary_ul = np.asarray(self._moos_config.boundary_ul, dtype=np.float32)
        self.boundary_ur = np.asarray(self._moos_config.boundary_ur, dtype=np.float32)
        self.boundary_ll = np.asarray(self._moos_config.boundary_ll, dtype=np.float32)
        self.boundary_lr = np.asarray(self._moos_config.boundary_lr, dtype=np.float32)
        self.world_size  = np.array([np.linalg.norm(self.boundary_lr - self.boundary_ll),
                                     np.linalg.norm(self.boundary_ul - self.boundary_ll)])
        
        # save the horizontal location of scrimmage line (relative to world/ playing field)
        self.scrimmage = 0.5*self.world_size[0]
        
        if self.timewarp is not None:
            self._moos_config.moos_timewarp = self.timewarp
            self._moos_config.sim_timestep = self._moos_config.moos_timewarp / 10.0
        self.steptime = self._moos_config.sim_timestep
        self.time_limit = self._moos_config.sim_time_limit
        self.timewarp = self._moos_config.moos_timewarp

        # add some padding becaus it can end up going faster than requested speed
        self.max_speed = self._moos_config.speed_bounds[1] + 0.5

        self.capture_radius = self._moos_config.capture_radius

        # game score
        self.max_score = self._moos_config.max_score

        # mark this function called already
        # if called again, nothing will happen
        self._config_set = True'''
