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
import pymoos
import time

from pyquaticus.config import get_std_config
from pyquaticus.structs import Player, Team


class Watcher:
    def __init__(self, server, agent_name, agent_port, team_names, opponent_names, quiet=True, team=None):
        """
        Subscribe to the relevant MOOSDB variables to form a Pyquaticus state.

        Args:
            server: server for the MOOS process
            name: the name of the agent
            port: the MOOS port for this agent
            team_names: list of names of team members
            opponent_names: list of names of opponents
            quiet: tell the pymoos comms object to be quiet
        """
        self._server = server
        self._agent_name = agent_name
        self._agent_port = agent_port
        self._team_names = team_names
        self._opponent_names = opponent_names
        self._quiet = quiet

        if isinstance(team, str) and team.lower() in {"red", "blue"}:
            self._team = Team.RED_TEAM if team == "red" else Team.BLUE_TEAM
        elif "red" in agent_name and "blue" not in agent_name:
            self._team = Team.RED_TEAM
        elif "blue" in agent_name and "red" not in agent_name:
            self._team = Team.BLUE_TEAM
        else:
            raise ValueError(f"Unknown team: please pass team=[red|blue]")

        self._opponent_team = Team.BLUE_TEAM if self._team == Team.RED_TEAM else Team.RED_TEAM

        # TODO: consider refactoring so base player class doesn't have rendering info
        radius = 10.
        player_id = 0
        config_dict = get_std_config()
        self.players = {agent_name: Player(player_id, self._team, radius, config_dict)}
        player_id += 1
        for name in self._team_names:
            self.players[name] = (Player(player_id, self._team, radius, config_dict))
            player_id += 1
        for name in self._opponent_names:
            self.players[name] = (Player(player_id, self._opponent_team, radius, config_dict))
            player_id += 1

        for player in self.players.values():
            player.pos = [None, None]

        self._init_moos_comm()

    def __del__(self):
        if self._moos_comm.close(nice=True):
            return
        else:
            time.sleep(2)

        if self._moos_comm.close(nice=True):
            return
        else:
            self._moos_comm.close(nice=False)


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

    def _on_mail(self):
        try:
            for msg in self._moos_comm.fetch():
                self._dispatch_message(msg)
                return True
        except Exception as e:
            print(f"Got exception: {e}")
            raise e

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
            agent.tagged = name in tagged_agents

    def _node_report_handler(self, msg):
        agent_name = msg.key().removeprefix("NODE_REPORT_").lower()
        data = {field: val 
                for field, val in (entry.split("=") for entry in msg.string().split(","))}
        assert agent_name == data["NAME"]

        agent = self.players[agent_name]
        agent.pos = [float(data["X"]), float(data["Y"])]
        agent.speed = float(data["SPD"])
        agent.heading = float(data["HDG"])
