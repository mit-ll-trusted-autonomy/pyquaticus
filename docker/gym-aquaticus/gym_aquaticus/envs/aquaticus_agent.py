import itertools
import numpy as np
import pymoos
import time

from gym_aquaticus.utils import PNT, Vertex

class AquaticusAgent(pymoos.comms):
    '''
    This class connects to a MOOS vehicle process and
    asynchronously updates its state
    '''
    def __init__(self, config, server, port, name, team):
        '''
        Args:
            config:       configuration object (see config.py)
            name:         the MOOS name
            port:         the MOOS shoreside port
            team:         'red' or 'blue'
        '''
        super().__init__()
        self._server = server
        self._port = port
        self._name = name
        self._team = team
        self._opponent_team = 'red' if self._team == 'blue' else 'blue'
        self._grab_report_str = f'{self._opponent_team.upper()}_FLAG_GRABBED'

        self.pnt = PNT(self._name, config.tagging_cooldown)

        # The flag positions are defined in shoreside/launch_shoreside.sh
        # which are propagated to targ_shoreside.moos and have to match config.py
        self.capture_radius = config.capture_radius
        self.flag_range = config.flag_range
        self.flag_grab_request = False
        self.flag_grabbed = False # this is set by the environment via notify_flag_status
        # Messages are asynchronous; we need to track bonus/penalty due
        self.flag_grab_bonus_due = False
        self.capture_bonus_due = False
        self.tag_penalty_due = False
        self.flag_lost_penalty_due = False
        self.tagging_cooldown = config.tagging_cooldown

        blue_flag = np.asarray(config.blue_flag, dtype=np.float32)
        red_flag = np.asarray(config.red_flag, dtype=np.float32)

        # The own-zone boundary is defined in shoreside/meta_shoreside.moos
        if team == 'blue':
            self.zone_ul = Vertex(config.blue_zone_ul)
            self.zone_ur = Vertex(config.blue_zone_ur)
            self.zone_ll = Vertex(config.blue_zone_ll)
            self.zone_lr = Vertex(config.blue_zone_lr)

            self._goal_flag = red_flag
            self._defend_flag = blue_flag
        else:
            assert team == 'red'
            self.zone_ul = Vertex(config.red_zone_ul)
            self.zone_ur = Vertex(config.red_zone_ur)
            self.zone_ll = Vertex(config.red_zone_ll)
            self.zone_lr = Vertex(config.red_zone_lr)

            self._goal_flag = blue_flag
            self._defend_flag = red_flag

        # The operating boundary is defined in shoreside/meta_shoreside.moos
        self.boundary_ul = Vertex(config.boundary_ul)
        self.boundary_ur = Vertex(config.boundary_ur)
        self.boundary_ll = Vertex(config.boundary_ll)
        self.boundary_lr = Vertex(config.boundary_lr)

        # Discrete Action space
        self.speeds = config.speeds
        self.n_speeds = len(self.speeds)
        self.headings = config.headings

        # Continuous Action space
        self.speed_bounds = config.speed_bounds
        self.heading_bounds = config.heading_bounds

        self.action_type = config.action_type
        if self.action_type == 'discrete':
            self.action_space_size = len(self.speeds)*len(self.headings)
        else:
            assert self.action_type == 'continuous', "Expecting action_type to be 'discrete' or 'continuous'"
            self.action_space_size = 2 # speed and heading
        self.action_count = 0

        # Set up callbacks
        self.set_on_connect_callback(self._on_connect)
        self.set_on_mail_callback(self._on_mail)

        # Timewarp must match value in MOOSDB (in sim_script)
        pymoos.set_moos_timewarp(config.moos_timewarp)

    #-----------------------------------------------------------------
    # Called when the application has made contact with the MOOSDB
    # and a channel has been opened.
    #-----------------------------------------------------------------
    def _on_connect(self):
        self._register_vars()
        return True


    #-----------------------------------------------------------------
    # Register for variables of interest in the MOOSDB.
    #-----------------------------------------------------------------
    def _register_vars(self):
        info_vars = ["NAV_X", "NAV_Y", "NAV_SPEED", "NAV_HEADING",
                     "FLAG_SUMMARY", "TAGGED_VEHICLES",
                     "BLUE_FLAG_GRABBED", "RED_FLAG_GRABBED"]
        for var in info_vars:
            self.register(var, 0)


    #----------------------------------------------------------------------
    # Called whenever new mail has arrived (notifications that something has
    # changed in the MOOSDB).
    #----------------------------------------------------------------------
    def _on_mail(self):

        for msg in self.fetch():
            # process ownship report (updates 10 times/sec from uSimMarine)
            if msg.key() == 'NAV_X':
                self.pnt.x = msg.double()
            elif msg.key() == 'NAV_Y':
                self.pnt.y = msg.double()
            elif msg.key() == 'NAV_SPEED':
                self.pnt.speed = msg.double()
            elif msg.key() == 'NAV_HEADING':
                self.pnt.heading = msg.double()

            # process tagged message (updates 4 times/sec from uFldTagManager)
            # updates whether any vehicle are tagged or not (empty string if
            # nobody is tagged)
            elif msg.key() == 'TAGGED_VEHICLES':
                if self.pnt.name in msg.string():
                    # note: tagged msg gets re-issued as long as vehicle is tagged
                    if not self.pnt.tagged:
                        # transition untagged-to-tagged gets a penalty
                        if self.flag_grabbed:
                            self.flag_lost_penalty_due = True
                        else:
                            self.tag_penalty_due = True
                    self.pnt.tagged = True
                    self.flag_grab_request = False
                    # training episode ends on tag
                    self._done = True
                else:
                    self.pnt.tagged = False

            # Note: these messages aren't currently broadcast to the herons
            #       currently managing this through the environment by
            #       subscribing to messages from shoreside
            #       alternative solution is to bridge messages to the agents
            # process flag grabbed
            # elif msg.key() == self._grab_report_str:
            #     if msg.string() == 'true':
            #         # grab/tag messages may come out of order
            #         if not self.pnt.tagged:
            #             self.flag_grabbed = True
            #             self.flag_grab_bonus_due = True
            #         else:
            #             self.flag_lost_penalty_due = True
            #             self.tag_penalty_due = False
            #     else:
            #         self.flag_grab_request = False
            #         self.flag_grabbed = False

        # ----- End of message processing ------

        # If we are within tag range of the enemy flag, issue a grab request
        flag_dist = np.hypot(self._goal_flag[0]-self.pnt.x,
                             self._goal_flag[1]-self.pnt.y)
        if (flag_dist < self.capture_radius) and not self.flag_grab_request and not self.pnt.tagged:
            self.notify('FLAG_GRAB_REQUEST', f'vname={self.pnt.name}', -1)
            self.flag_grab_request = True

        # If we have the flag and reach our own flag, we captured the flag!
        own_flag_dist = np.hypot(self._defend_flag[0]-self.pnt.x,
                                 self._defend_flag[1]-self.pnt.y)
        if self.flag_grabbed and (own_flag_dist < self.flag_range):
            # TODO: consider making these counts so that we eventually update them
            self.flag_grab_request = False
            self.capture_bonus_due = True
            self._done = True

        return True


    #----------------------------------------------------------------------
    # Post new desired speed, heading commands for BHV_RLAgent.
    # Optionally can directly send MOOS action (e.g., ATTACK_E/ATTACK_MED/DEFEND_E)
    #----------------------------------------------------------------------
    def take_action(self, action):
        if isinstance(action, str):
            # pass MOOS action directly
            self.notify('ACTION', action, -1)
        elif self.action_type == 'discrete':
            # look up corresponding speed/heading and send
            self.notify('RLA_SPEED', self.speeds[action%self.n_speeds], -1)
            self.notify('RLA_HEADING', self.headings[action//self.n_speeds], -1)
            self.notify('RLA_ACTION_COUNT', self.action_count, -1)
            self.action_count += 1
        else:
            assert self.action_type == 'continuous', "Expecting action_type to be 'discrete' or 'continuous'"
            self.notify('ACTION', 'CONTROL', -1)
            spd, hdg = action
            self.notify('RLA_SPEED', spd, -1)
            self.notify('RLA_HEADING', hdg, -1)
            self.notify('RLA_ACTION_COUNT', self.action_count, -1)
            self.action_count += 1

    @property
    def name(self):
        return self._name


    def get_action_high_limit(self):
        '''
        Return the upper bounds for the continuous action space.
        Note: select discrete vs continuous action space in config.py
        '''
        return np.array([self.speed_bounds[1], self.heading_bounds[1]],
                        dtype=np.float32)


    def get_action_low_limit(self):
        '''
        Return the lower bounds for the continuous action space.
        Note: select discrete vs continuous action space in config.py
        '''
        return np.array([self.speed_bounds[0], self.heading_bounds[0]],
                        dtype=np.float32)

    def get_obs_high_limit(self):
        boundaries = [self.boundary_ul,
                      self.boundary_ur,
                      self.boundary_ll,
                      self.boundary_lr]

        x_bounds = [b.x for b in boundaries]
        y_bounds = [b.y for b in boundaries]
        cushion = 20
        # x_pos, y_pos, has_flag boolean
        return np.array([max(x_bounds)+cushion, max(y_bounds)+cushion, 1.])

    def get_obs_low_limit(self):
        boundaries = [self.boundary_ul,
                      self.boundary_ur,
                      self.boundary_ll,
                      self.boundary_lr]

        x_bounds = [b.x for b in boundaries]
        y_bounds = [b.y for b in boundaries]
        cushion = 20
        # x_pos, y_pos, has_flag boolean
        return np.array([min(x_bounds)-cushion, min(y_bounds)-cushion, 0.])


    def connect_to_moos(self):
        # Start the client (waits for connection in a new thread)
        if self._port:
            self.run(self._server, self._port, self._name)

        # Don't print connection information
        self.set_quiet(True)


    def notify_flag_status(self, status):
        '''
        Used by the environment to notify this agent of its flag grab request status
        If the agent has the flag, the status is True
        otherwise it is False

        Args:
            status: boolean that is True iff the agent has the flag
        '''
        self.flag_grabbed = status