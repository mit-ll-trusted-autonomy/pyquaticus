"""
Observation Space (per agent):
    Retrieve flag relative bearing (clockwise degrees)
    Retrieve flag distance (meters)
    Home flag relative bearing (clockwise degrees)
    Home flag distance (meters)
    Wall 1 relative bearing (clockwise degrees)
    Wall 1 distance (meters)
    Wall 2 relative bearing (clockwise degrees)
    Wall 2 distance (meters)
    Wall 3 relative bearing (clockwise degrees)
    Wall 3 distance (meters)
    Wall 4 relative bearing (clockwise degrees)
    Wall 4 distance (meters)
    Own speed (meters per second)
    Own flag status (boolean)
    On side (boolean)
    Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
    Is tagged (boolean)
    For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
        Bearing from you (clockwise degrees)
        Distance (meters)
        Heading of other agent relative to the vector to you (clockwise degrees)
        Speed (meters per second)
        Has flag status (boolean)
        On their side status (boolean)
        Tagging cooldown (seconds)
        Is tagged (boolean)
"""

"""
MOOS Variables of Interest -- based on red_one
Own State and Control:
    position: NAV_X, NAV_Y
    speed and heading: NAV_SPEED, NAV_HEADING
Own Flag Status:
    HAS_FLAG
Own Tag Status and cooldown:
    TAGGED
    TAGGED_VEHICLES (Mike adding)

Other:
    position and control: NODE_REPORT_{NAME}
        includes: X, Y, SPD, HDG -- needs parsing
        it's a csv with name=value pairs
    flag status:
        FLAG_SUMMARY -- see if your flag (identified with
                        label or color) has an owner
                        that's not you (not sure if it can
                        be you anyway -- can't pick up your
                        own flag)
    tag status and cooldown:
        TAGGED_VEHICLES (Mike adding)
        
                     


To Publish:
FLAG_GRAB_REQUEST -- won't pick up otherwise?
TAG_REQUEST? -- doesn't seem like we're using it currently and it still works -- can't find with ag
DEPLOY_ALL

To Update in State:
    position, control actions, flag status, tag status
    update flag status on successful grab or when tagged

Notes:
flag homes and boundaries are configuration variables


Create a Pyquaticus State
self.state = {
    "agent_position": agent_positions,
    "prev_agent_position": copy.deepcopy(agent_positions),
    "agent_spd_hdg": agent_spd_hdg,
    "agent_has_flag": np.zeros(self.num_agents),
    "agent_on_sides": agent_on_sides,
    "flag_home": copy.deepcopy(flag_locations),
    "flag_locations": flag_locations,
    "flag_taken": np.zeros(2),
    "current_time": 0.0,
    "agent_captures": [
        None
    ] * self.num_agents,  # whether this agent tagged something
    "agent_tagged": [0] * self.num_agents,  # if this agent was tagged
    "agent_oob": [0] * self.num_agents,  # if this agent went out of bounds
}
"""

import itertools
import numpy as np
import pymoos
import time

from pyquaticus.envs.pyquaticus import PyQuaticusEnvBase
from pyquaticus.structs import Player, Team, Flag
from pyquaticus.config import config_dict_std


class PyQuaticusMoosBridge(PyQuaticusEnvBase):
    def __init__(self, server, agent_name, agent_port, team_names, opponent_names, moos_config, \
                 quiet=True, team=None, timewarp=None, tagging_cooldown=config_dict_std["tagging_cooldown"],
                 normalize=True):
        """
        Subscribe to the relevant MOOSDB variables to form a Pyquaticus state.

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

        # Set tagging cooldown
        for player in self.players.values():
            player.tagging_cooldown = self.tagging_cooldown

        for player in self.players.values():
            player.pos = [None, None]

        if self._moos_comm is not None:
            self._moos_comm.close()

        self._init_moos_comm()

        return self.state_to_obs(self._agent_name)

    def render(self, mode="human"):
        pass

    def close(self):
        max_nice_attempts = 2
        for i in range(max_nice_attempts):
            if self._moos_comm.close(nice=True):
                time.sleep(0.1)
                return
            time.sleep(0.2)
        self._moos_comm.close(nice=False)

    def step(self, action):
        # set on_own_side for each agent using _check_side(player)
        for agent in self.players.values():
            agent.on_own_side = self._check_on_side(agent)
        # translate actions and publish them
        desired_spd, delta_hdg = self._discrete_action_to_speed_relheading(action)
        desired_hdg = self._relheading_to_global_heading(
            self.players[self._agent_name].heading,
            delta_hdg)
        # notify the moos agent that we're controlling it directly
        # NOTE: the name of this variable depends on the mission files
        moostime = pymoos.time()
        self._moos_comm.notify("ACTION", "CONTROL", moostime)
        self._moos_comm.notify("RLA_SPEED", desired_spd, moostime)
        self._moos_comm.notify("RLA_HEADING", desired_hdg, moostime)
        self._action_count += 1
        self._moos_comm.notify("RLA_ACTION_COUNT", self._action_count, moostime)
        # always returning zero reward for now
        # this is only for running policy, not traning
        # TODO: implement a sparse reward for evaluation
        reward = 0.
        # just for evaluation, never need to reset (might be running real robots)
        terminated, truncated = False, False

        # let the action occur
        time.sleep(self.steptime / self.timewarp)

        obs = self.state_to_obs(self._agent_name)
        return obs, reward, terminated, truncated, {}

    def _init_moos_comm(self):
        self._moos_comm = pymoos.comms()
        self._moos_vars = [
            "NAV_X", "NAV_Y",
            "NAV_SPEED", "NAV_HEADING",
            "FLAG_SUMMARY",
            "TAGGED_VEHICLES"
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
        try:
            for msg in self._moos_comm.fetch():
                self._dispatch_message(msg)
            return True
        except Exception as e:
            print(f"Got exception: {e}")
            return False

    def _dispatch_message(self, msg):
        if "NAV_" in msg.key():
            self._nav_handler(msg)
        elif "FLAG_SUMMARY" == msg.key():
            self._flag_handler(msg)
        elif "TAGGED_VEHICLES" == msg.key():
            self._tag_handler(msg)
        elif "NODE_REPORT_" in msg.key():
            self._node_report_handler(msg)
        else:
            raise ValueError(f"Unexpected message: {msg.key()}")

    def _nav_handler(self, msg):
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
        # Note: assuming the underlying MOOS logic only allows
        #       agents to grab the opponent's flag, so not even
        #       checking for that
        for msg_entry in msg.string().split("#"):
            flag_holders = set()
            for col in msg_entry.split(","):
                field, val = col.split("=")
                if field.lower() == "owner":
                    # the value for owner is the agent name
                    flag_holders.add(val)
            # update all player objects
            for name, agent in self.players.items():
                agent.has_flag = name in flag_holders

    def _tag_handler(self, msg):
        tagged_agents = set(msg.string().split(","))
        # update all player objects
        for name, agent in self.players.items():
            tag_status = name in tagged_agents
            if tag_status != agent.is_tagged:
                print(f"Warning: getting no tag cooldown information!")
            agent.is_tagged = tag_status

    def _node_report_handler(self, msg):
        agent_name = msg.key().removeprefix("NODE_REPORT_").lower()
        data = {field: val 
                for field, val in (entry.split("=") for entry in msg.string().split(","))}
        assert agent_name == data["NAME"]

        agent = self.players[agent_name]
        agent.pos = [float(data["X"]), float(data["Y"])]
        agent.speed = float(data["SPD"])
        agent.heading = float(data["HDG"])

    def set_config(self, moos_config):
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
        # define function for checking which side an agent is on
        if abs(self.scrimmage_pnts[0][0] - self.scrimmage_pnts[1][0]) < 1e-2:
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
            m = (self.scrimmage_pnts[1][1] - self.scrimmage_pnts[0][1]) / (self.scrimmage_pnts[1][0] - self.scrimmage_pnts[0][1])
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
        if self.timewarp is not None:
            self._moos_config.moos_timewarp = self.timewarp
            self._moos_config.sim_timestep = self._moos_config.moos_timewarp / 10.0
        self.steptime = self._moos_config.sim_timestep
        self.time_limit = self._moos_config.sim_time_limit
        self.timewarp = self._moos_config.moos_timewarp

        # add some padding becaus it can end up going faster than requested speed
        self.max_speed = self._moos_config.speed_bounds[1] + 0.5

        # mark this function called already
        # if called again, nothing will happen
        self._config_set = True
