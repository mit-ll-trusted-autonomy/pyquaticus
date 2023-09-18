#----------------------------------------------------------------------
# config.py -- configuration parameters for Python wrappers of MOOS-IvP
#
# originally developed at NRL and modified at MIT LL


class MITConfig:
    '''
    This is the original configuration class used for the MIT Sailing Pavilion MOOS shoreside.
    '''
    def __init__(self):
        # Simulation environment
        self.moos_server = 'localhost' # these must match values in simulation script
        self.moos_port_attacker = 9006 # crap. this is from tar_felix.moos
        self.moos_port_defender = 9005
        self.sim_timestep = 0.4      # moostime (sec) between steps
        self.sim_time_limit = 180.0  # moostime (sec) before terminating episode
        # Aquaticus field setup
        self.attacker_name = 'felix'
        self.defender_name = 'evan'
        # The flag positions are defined in shoreside/launch_shoreside.sh
        # (propagated to targ_shoreside.moos) and must be matched here
        self.blue_flag = [-58.0, -71.0]
        self.red_flag = [50.0, -24.0]
        self.capture_radius = 8.0
        self.flag_range = 10.0
        # The operating boundary is defined in shoreside/meta_shoreside.moos
        # (note that it is not rectangular)
        self.boundary_ul = [-83.0,  -49.0]
        self.boundary_ur = [ 56.0,   16.0]
        self.boundary_ll = [-53.0, -114.0]
        self.boundary_lr = [ 82.0,  -56.0]

        self.red_zone_ul = [-15.0, -17.0]
        self.red_zone_ur = [ 56.0,  16.0]
        self.red_zone_ll = [ 17.0, -83.0]
        self.red_zone_lr = [ 82.0, -56.0]

        self.blue_zone_ul = [-83.0,  -49.0]
        self.blue_zone_ur = [-15.0,  -17.0]
        self.blue_zone_ll = [-53.0, -114.0]
        self.blue_zone_lr = [ 17.0,  -83.0]

        self.scrimmage_pnts = [self.blue_zone_lr, self.blue_zone_ur]

        # Discrete Action Space
        self.speeds = [0.5, 2.0]   # two speeds
        self.headings = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]

        # Continuous Action Space
        self.speed_bounds = (0.0, 2.5)
        self.heading_bounds = (0.0, 360.0)

        # Set action space type to 'discrete' vs 'continuous'
        # Note: most scripts from MIT LL are written for continuous only
        self.action_type = 'continuous'

        # Constants used to define the reward function
        self.max_reward = 1.0
        self.neg_reward = -0.5
        self.reward_dropoff = 0.96
        self.max_reward_radius = 8.0

        # Scoring bonus and tag penalty rules
        # per Michael Novitzky 11 Feb 2021
        self.grab_bonus = 1
        self.capture_bonus = 2
        self.tag_penalty = -1
        self.lost_penalty = -2
        # Increase bonus/penalty weight compared to reward function
        self.bonus_weight = 50

        # Step penalty - encourage faster play by decrementing reward each step
        self.step_penalty = 0.05

        self.tagging_cooldown = 10.

        # MOOS Timewarp is the simulation speed-up factor
        # This must match the value in your simulation script
        self.moos_timewarp = 4.0

class PopolopenConfig(MITConfig):
    '''
    This configuration class modifies the MIT configuration to match the
    Lake Popolopen shoreside environment used in the 2022 competition.
    '''
    def __init__(self):
        super().__init__()
        # defined in targ_red_one.moos and targ_blue_one.moos
        self.moos_port_attacker = 9011
        self.moos_port_defender = 9015
        self.attacker_name = 'red_one'
        self.defender_name = 'blue_one'
        self.blue_flag = [20, 40]
        self.red_flag = [140.0, 40.0]

        # defined in targ_shoreside.moos and must match
        # zone_one = pts={ 80,0 : 80,80 : 160,80 : 160,0 }
        # zone_two = pts={ 0,0 : 0,80 : 80,80 : 80,0 }
        self.red_zone_ll = [80, 0]
        self.red_zone_lr = [160, 0]
        self.red_zone_ur = [160, 80]
        self.red_zone_ul = [80, 80]

        self.blue_zone_ll = [0, 0]
        self.blue_zone_lr = [80, 0]
        self.blue_zone_ur = [80, 80]
        self.blue_zone_ul = [0, 80]

        self.boundary_ll = [0, 0]
        self.boundary_lr = [160, 0]
        self.boundary_ur = [160, 80]
        self.boundary_ul = [0, 80]

        self.scrimmage_pnts = [self.blue_zone_lr, self.blue_zone_ur]


class WestPointConfig(MITConfig):
    def __init__(self):
        super().__init__()
        self.moos_port_attacker = 9011
        self.moos_port_defender = 9015
        self.attacker_name = 'red_one'
        self.defender_name = 'blue_one'
        # defined in wp_competition-2022/shoreside/launch_shoreside
        # and propagated to targ_shoreside.moos -- must be matched here
        self.blue_flag = [225.5, 79.8]
        self.red_flag = [254.7, 154.5]

        # defined in uFldTagManager configuration block of meta_shoreside.moos
        self.tag_radius = 10.0

        # Trying sampling less so that RL can see impact of choice
        self.sim_timestep = 0.4      # moostime (sec) between steps
        self.moos_timewarp = 4

        # will sometimes exceed this boundary so it's a good idea to pad a bit
        # when setting the observation limits for normalizing
        self.speed_bounds = (0.0, 5.)

        # THIS
        # defined in meta_shoreside.moos and must match
        # zone_one = pts={ 266,107: 286,158.3 : 234.7,178.3: 214.7, 127 }
        # zone_two = pts={ 246,55.7 : 266, 107 : 214.7, 127: 194.7, 75.7 }

        self.red_zone_ll = [214.7, 127]
        self.red_zone_lr = [266, 107]
        self.red_zone_ur = [286, 158.3]
        self.red_zone_ul = [234.7, 178.3]

        self.blue_zone_ll = [194.7, 75.7]
        self.blue_zone_lr = [246, 55.7]
        self.blue_zone_ur = [266, 107]
        self.blue_zone_ul = [214.7, 127]

        self.boundary_ll = [194.7, 75.7]
        self.boundary_lr = [246, 55.7]
        self.boundary_ur = [286, 158.3]
        self.boundary_ul = [234.7, 178.3]

        self.scrimmage_pnts = [self.blue_zone_ul, self.blue_zone_ur]
