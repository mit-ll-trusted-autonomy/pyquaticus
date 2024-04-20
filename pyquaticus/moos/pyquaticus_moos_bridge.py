import itertools
import numpy as np
import pymoos
import time

from pyquaticus.envs.pyquaticus import PyQuaticusEnvBase
from pyquaticus.structs import Player, Team, Flag
from pyquaticus.config import config_dict_std
from pyquaticus.utils.utils import mag_bearing_to


class PyQuaticusMoosBridge(PyQuaticusEnvBase):
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
        self._server = server
        self._agent_name = agent_name
        self._agent_port = agent_port
        self._team_names = team_names
        self._opponent_names = opponent_names
        self._quiet = quiet
        # Note: not using _ convention to match pyquaticus
        self.timewarp = timewarp
        self.tagging_cooldown = tagging_cooldown
        self.normalize = normalize

        self.set_config(moos_config)

        self.game_score = {'blue_captures':0, 'red_captures':0}

        if isinstance(team, str) and team.lower() in {"red", "blue"}:
            self.team = Team.RED_TEAM if team == "red" else Team.BLUE_TEAM
        elif "red" in agent_name and "blue" not in agent_name:
            self.team = Team.RED_TEAM
        elif "blue" in agent_name and "red" not in agent_name:
            self.team = Team.BLUE_TEAM
        else:
            raise ValueError(f"Unknown team: please pass team=[red|blue]")

        self._opponent_team = Team.BLUE_TEAM if self.team == Team.RED_TEAM else Team.RED_TEAM
        self._moos_comm = None

        own_team_len = len(self._team_names) + 1
        opp_team_len = len(self._opponent_names)
        if own_team_len != opp_team_len:
            raise ValueError(f"Expecting equal team sizes but got: {own_team_len} vs {opp_team_len}")

        self.agent_obs_normalizer = self._register_state_elements(own_team_len)

        self.observation_space = self.get_agent_observation_space()
        self.action_space = self.get_agent_action_space()

    def reset(self):
        """
        Sets up the players and resets variables.
        (Re)connects to MOOS node for the provided agent name.
        """
        self._action_count = 0
        assert isinstance(self.timewarp, int)
        pymoos.set_moos_timewarp(self.timewarp)
        self.agents_of_team = {t: [] for t in Team}

        self.agents_of_team[self.team].append(Player(self._agent_name, self.team))
        for name in self._team_names:
            self.agents_of_team[self.team].append(Player(name, self.team))
        for name in self._opponent_names:
            self.agents_of_team[self._opponent_team].append(Player(name, self._opponent_team))

        self.players = {}
        for agent_list in self.agents_of_team.values():
            for agent in agent_list:
                self.players[agent.id] = agent

        # reset auto return status
        self._auto_returning_flag = False

        # Set tagging cooldown
        for player in self.players.values():
            player.tagging_cooldown = self.tagging_cooldown

        for player in self.players.values():
            player.pos = [None, None]

        if self._moos_comm is not None:
            self._moos_comm.close()

        self._init_moos_comm()
        self._wait_for_all_players()

        for k in self.game_score:
            self.game_score[k] = 0

        self._determine_team_wall_orient()

        return self.state_to_obs(self._agent_name)

    def _wait_for_all_players(self):
        wait_time = 5
        missing_agents = [p.id for p in filter(lambda p: None in p.pos, self.players.values())]
        num_iters = 0
        while missing_agents:
            print("Waiting for other players to connect...")
            print(f"\tMissing Agents: {','.join(missing_agents)}")
            time.sleep(wait_time)
            missing_agents = [p.id for p in filter(lambda p: None in p.pos, self.players.values())]
            num_iters += 1
            if num_iters > 20:
                raise RuntimeError(f"No other agents connected after {num_iters*wait_time} seconds. Failing and exiting.")
        print("All agents connected!")
        return

    def render(self, mode="human"):
        """
        This is a pass through, all rendering is handled by pMarineViewer on the MOOS side.
        """
        pass

    def close(self):
        """
        Close the connection to MOOS
        """
        max_nice_attempts = 2
        for i in range(max_nice_attempts):
            if self._moos_comm.close(nice=True):
                time.sleep(0.1)
                return
            time.sleep(0.2)
        self._moos_comm.close(nice=False)

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
        player = self.players[self._agent_name]
        moostime = pymoos.time()
        if isinstance(action,str):
            self._moos_comm.notify("ACTION", action, moostime)
            self._auto_returning_flag = False
        elif player.on_own_side and player.has_flag:
            # automatically return agent to home region
            desired_spd = self.max_speed
            player_flag_home = self.flags[int(self.team)].home
            _, desired_hdg = mag_bearing_to(player.pos,
                                            player_flag_home)
            if not self._auto_returning_flag:
                print("Taking over control to return flag")
            self._auto_returning_flag = True
            self._moos_comm.notify("ACTION", "CONTROL", moostime)
            self._moos_comm.notify("RLA_SPEED", desired_spd, moostime)
            self._moos_comm.notify("RLA_HEADING", desired_hdg%360, moostime)
        else:
            # translate actions and publish them
            desired_spd, delta_hdg = self._discrete_action_to_speed_relheading(action)
            desired_hdg = self._relheading_to_global_heading(
                self.players[self._agent_name].heading,
                delta_hdg)

            # notify the moos agent that we're controlling it directly
            # NOTE: the name of this variable depends on the mission files
            self._moos_comm.notify("ACTION", "CONTROL", moostime)
            self._moos_comm.notify("RLA_SPEED", desired_spd, moostime)
            self._moos_comm.notify("RLA_HEADING", desired_hdg%360, moostime)
            self._action_count += 1
            self._moos_comm.notify("RLA_ACTION_COUNT", self._action_count, moostime)
            # if close enough to flag, will attempt to grab
            self._flag_grab_publisher()
            self._auto_returning_flag = False
        # always returning zero reward for now
        # this is only for running policy, not traning
        # TODO: implement a sparse reward for evaluation
        reward = -1.
        # just for evaluation, never need to reset (might be running real robots)
        terminated, truncated = False, False

        # let the action occur
        time.sleep(self.steptime / self.timewarp)

        obs = self.state_to_obs(self._agent_name)
        return obs, reward, terminated, truncated, {}

    def state_to_obs(self, agent_id, normalize=True):
        """
        Light wrapper around parent class state_to_obs function
        """
        # set on_own_side for each agent using _check_side(player)
        for agent in self.players.values():
            agent.on_own_side = self._check_on_side(agent)
        # update tagging_cooldown value
        for agent in self.players.values():
            # should count up from 0.0 to tagging_cooldown (at which point it can tag again)
            agent.tagging_cooldown = self.tagging_cooldown - max(0.0, agent.cantag_time - time.time())
            agent.tagging_cooldown = max(0, min(self.tagging_cooldown, agent.tagging_cooldown))
        return super().state_to_obs(agent_id, normalize)

    def _init_moos_comm(self):
        """
        Create a connection to the MOOS node for self._agent_name
        """
        self._moos_comm = pymoos.comms()
        self._moos_vars = [
            "NAV_X", "NAV_Y",
            "NAV_SPEED", "NAV_HEADING",
            "FLAG_SUMMARY",
            "TAGGED_VEHICLES",
            "CANTAG_SUMMARY",
            "BLUE_SCORES",
            "RED_SCORES"
        ]
        self._moos_vars.extend([
            f"NODE_REPORT_{n.upper()}"
            for n in itertools.chain(self._team_names, self._opponent_names)
        ])

        def _on_connect():
            for vname in self._moos_vars:
                self._moos_comm.register(vname, 0)
            self._moos_comm.register('DEPLOY_ALL', 0)
            return True

        self._moos_comm.set_on_connect_callback(_on_connect)
        self._moos_comm.set_on_mail_callback(self._on_mail)
        self._moos_comm.run(
            str(self._server),
            int(self._agent_port),
            str(self._agent_name))
        self._moos_comm.set_quiet(self._quiet)

        start_time = time.time()
        wait_time = 1.
        while not self._moos_comm.is_connected():
            print(f"Waiting for agent to connect...")
            time.sleep(wait_time)
            wait_time = wait_time + 1 if wait_time < 10 else wait_time
            if time.time() - start_time > 300:
                print(f"Timed out!")

    def _on_mail(self):
        """
        Performed everytime there are updates to the MOOSDB
        for variables that we registered for in _init_moos_comm
        """
        try:
            for msg in self._moos_comm.fetch():
                self._dispatch_message(msg)
            return True
        except Exception as e:
            print(f"Got exception: {e}")
            return False

    def pause(self):
        self._moos_comm.notify("DEPLOY_ALL", "FALSE", pymoos.time())

    def _dispatch_message(self, msg):
        """
        Dispatch MOOSDB messages to appropriate handlers
        """
        if "NAV_" in msg.key():
            self._nav_handler(msg)
        elif "FLAG_SUMMARY" == msg.key():
            self._flag_handler(msg)
        elif "TAGGED_VEHICLES" == msg.key():
            self._tag_handler(msg)
        elif "NODE_REPORT_" in msg.key():
            self._node_report_handler(msg)
        elif "CANTAG_SUMMARY" in msg.key():
            self._cantag_handler(msg)
        elif "_SCORES" in msg.key():
            self._score_handler(msg)
        else:
            raise ValueError(f"Unexpected message: {msg.key()}")

    def _nav_handler(self, msg):
        """
        Handles navigation messages that provide information
        about this agent's nodes
        """
        if msg.key() == "NAV_X":
            self.players[self._agent_name].pos[0] = msg.double()
        elif msg.key() == "NAV_Y":
            self.players[self._agent_name].pos[1] = msg.double()
        elif msg.key() == "NAV_SPEED":
            self.players[self._agent_name].speed = msg.double()
        elif msg.key() == "NAV_HEADING":
            self.players[self._agent_name].heading = msg.double()
        else:
            raise ValueError(f"Unexpected message: {msg.key()}")

    def _flag_handler(self, msg):
        """
        Handles messages about the flags.
        """
        # Note: assuming the underlying MOOS logic only allows
        #       agents to grab the opponent's flag, so not even
        #       checking for that
        flag_holders = set()
        for msg_entry in msg.string().split("#"):
            for col in msg_entry.split(","):
                field, val = col.split("=")
                if field.lower() == "owner":
                    # the value for owner is the agent name
                    flag_holders.add(val)
        # update all player objects
        for name, agent in self.players.items():
            agent.has_flag = name in flag_holders

    def _tag_handler(self, msg):
        """
        Handles messages about tags.
        """
        tagged_agents = set(msg.string().split(","))
        # update all player objects
        for name, agent in self.players.items():
            tag_status = name in tagged_agents
            agent.is_tagged = tag_status

    def _cantag_handler(self, msg):
        """
        Handles messages about whether an agent can tag
        """
        # reset all cantag times to 0
        for p in self.players.values():
            p.cantag_time = 0.0
        strmsg = msg.string().strip()
        if not strmsg:
            return
        for entry in strmsg.split(","):
            # the agent cannot tag again until the specified utc
            agent_name, utc = entry.split('=')
            self.players[agent_name].cantag_time = float(utc)

    def _node_report_handler(self, msg):
        """
        Handles node reports about the state of other agents.
        """
        agent_name = msg.key().removeprefix("NODE_REPORT_").lower()
        data = {field: val
                for field, val in (entry.split("=") for entry in msg.string().split(","))}
        assert agent_name == data["NAME"]

        agent = self.players[agent_name]
        agent.pos = [float(data["X"]), float(data["Y"])]
        agent.speed = float(data["SPD"])
        agent.heading = float(data["HDG"])

    def _score_handler(self, msg):
        """
        Handles messages about scores.
        """
        if msg.key() == "BLUE_SCORES":
            self.game_score['blue_captures'] = msg.double()
        elif msg.key() == "RED_SCORES":
            self.game_score['red_captures'] = msg.double()
        else:
            raise ValueError(f"Unexpected message: {msg.key()}")

    def _flag_grab_publisher(self):
        player = self.players[self._agent_name]
        if any([a.has_flag for a in self.agents_of_team[self.team]]) or player.is_tagged:
            return

        goal_flag = self.flags[not int(self.team)]
        flag_dist = np.hypot(goal_flag.home[0]-player.pos[0],
                             goal_flag.home[1]-player.pos[1])
        if (flag_dist < self.capture_radius):
            print("SENDING A FLAG GRAB REQUEST!")
            self._moos_comm.notify('FLAG_GRAB_REQUEST', f'vname={self._agent_name}', -1)

    def set_config(self, moos_config):
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
        self._config_set = True
