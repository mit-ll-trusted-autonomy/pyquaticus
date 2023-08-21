#----------------------------------------------------------------------
# config.py -- configuration parameters for AquaticusEnv and
#       AquaticusAttacker


class Config:
    def __init__(self):
        # Simulation environment
        self.moos_server = 'localhost' # these must match values in simulation script
        self.ownship_name = 'red_one' #'blue_one' # 'red_one'#'felix'
        self.enemy_name = 'blue_one' # 'red_one' # 'blue_one' #'evan'
        self.moos_timewarp = 1

        self.moos_port = 9011 # 9015 # 9011 #9006
        self.moos_port_enemy = 9015 # 9011 #9015



        self.sim_timestep = 0.4      # moostime (sec) between steps
        self.sim_time_limit = 600000 #000000000 # moostime (sec) before terminating episode
        # Aquaticus field setup
        
        # The flag positions are defined in shoreside/launch_shoreside.sh
        # (propagated to targ_shoreside.moos) and must be matched here
        self.blue_flag = [225.83, 79.67]
        self.red_flag = [254.89, 154.21]
        self.capture_radius = 7.0
        self.flag_range = 10.0
        # The operating boundary is defined in shoreside/meta_shoreside.moos
        self.boundary_ul = [194.76,   75.68]
        self.boundary_ur = [234.72, 178.16]
        self.boundary_ll = [246.00,   55.70]
        self.boundary_lr = [ 285.96, 157.88]

        # Action space
        self.speeds = [0.5, 2.5]   # two speeds
        self.headings = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        # self.headings = [240.0, 300.0, 0.0, 60.0, 120.0, 180.0]

        # Constants used in defining the reward function
        self.max_reward = 0.5 # 100
        self.neg_reward = -0.5 #0.5 # -50
        self.reward_dropoff = .96
        self.max_reward_radius = 10

        # MOOS Timewarp is the simulation speed-up factor
        # This must match the value in your simulation script
