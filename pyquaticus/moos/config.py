#----------------------------------------------------------------------
# config.py -- configuration parameters for Python wrappers of MOOS-IvP
#
# originally developed at NRL and modified at MIT LL
import subprocess
import os
from pyquaticus.config import config_dict_std

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

        # defined in meta_shoreside.moos and must match
        # October Competition:
        #   zone_one = pts={ 266,107: 286,158.3 : 234.7,178.3: 214.7, 127 }
        #   zone_two = pts={ 246,55.7 : 266, 107 : 214.7, 127: 194.7, 75.7 }
        # Pyquaticus Mission Files:
        #   zone_one = pts={214.74,126.92 : 234.72,178.16 : 285.96,157.88: 265.98,106.94 }
        #   zone_two = pts={194.76,75.68 : 214.74,126.92 : 265.98,106.94: 246.00,55.70 }

        self.red_zone_ll = [214.74, 126.92]
        self.red_zone_lr = [265.98, 106.94]
        self.red_zone_ur = [285.96, 157.88]
        self.red_zone_ul = [234.72, 178.16]

        self.blue_zone_ll = [194.76, 75.68]
        self.blue_zone_lr = [246, 55.7]
        self.blue_zone_ur = [265.98, 106.94]
        self.blue_zone_ul = [214.74, 126.92]

        self.boundary_ll = [194.76, 75.68]
        self.boundary_lr = [246, 55.7]
        self.boundary_ur = [285.96, 157.88]
        self.boundary_ul = [234.72, 178.16]

        self.scrimmage_pnts = [self.blue_zone_ul, self.blue_zone_ur]


class JervisBayConfigFixed:
    '''
    This is the configuration for Jervis Bay 2023.
    '''
    def __init__(self):
        # Simulation environment
        self.sim_timestep = 0.4      # moostime (sec) between steps
        self.sim_time_limit = 180.0  # moostime (sec) before terminating episode
        self.capture_radius = 8.0
        self.flag_range = 10.0

        # The zones and positions are generated by jerop.sh in the mission folder
        # and stored in region_info.txt
        # The flag positions are defined in shoreside/launch_shoreside.sh
        # (propagated to targ_shoreside.moos) and must be matched here
        # copy-pasted from region_info.txt
        # BLUE_FLAG="x=76.4,y=130.6"
        # RED_FLAG="x=14.59,y=233.46"
        # RED_ZONE="pts={-30,230:38.57,271.2:79.77,202.63:11.2,161.43}"
        # BLUE_ZONE="pts={11.2,161.43:79.77,202.63:120.98,134.05:52.41,92.85}"

        self.blue_flag = [76.4, 130.6]
        self.red_flag = [14.59, 233.46]

        self.red_zone_ul = [-30., 230.]
        self.red_zone_ur = [38.57, 271.2]
        self.red_zone_lr = [79.77, 202.63]
        self.red_zone_ll = [11.2, 161.43]

        self.blue_zone_ul = [11.2, 161.43]
        self.blue_zone_ur = [79.77, 202.63]
        self.blue_zone_lr = [120.98, 134.05]
        self.blue_zone_ll = [52.41, 92.85]

        self.boundary_ul = self.red_zone_ul
        self.boundary_ur = self.red_zone_ur
        self.boundary_ll = self.blue_zone_ll
        self.boundary_lr = self.blue_zone_lr

        self.scrimmage_pnts = [self.red_zone_ll, self.red_zone_lr]

        self.speed_bounds = (0.0, 3.)
        self.heading_bounds = (0.0, 360.0)

        self.tagging_cooldown = 10.

        # MOOS Timewarp is the simulation speed-up factor
        # This must match the value in your simulation script
        self.moos_timewarp = 4.0

class FieldReaderConfig:
    '''
    This configuration reads field information generated by a script.
    This format started with the jervis-2023 mission and continued with the charles-2023
    mission, both included in moos-ivp-aquaticus
    '''
    def __init__(self, mission_dir:str, script:str =f'{os.path.dirname(__file__)}//get_field.sh', pyquaticus_config:dict =config_dict_std):
        print(f"Using field script: {script}")
        subprocess.call([script, mission_dir])
        # get values from saved value
        f = open(f'{mission_dir}/field.txt', 'r')
        x = f.readlines()[:3]
        red_zone = []
        blue_zone = []
        for zone in x:
            pts = []
            for coord in zone.split('{')[1].split('}')[0].split(':'):
                vals = coord.split(',')
                pts.append((float(vals[0]), float(vals[1])))

            if zone[:3].lower() == "red":
                red_zone.extend(pts)
            else:
                assert zone[:4].lower() == "blue"
                blue_zone.extend(pts)

        f.close()
        f = open(f'{mission_dir}/flags.txt', 'r')
        flags = f.readlines()

        bf = []
        rf = []
        i = 0
        for flag in flags[:2]:
            if flag[:3].lower() == "red":
                rf = [float(flag.split(',')[0].split('=')[-1]), float(flag.split(',')[1].split('=')[-1].split('\"')[0])]
            else:
                assert flag[:4].lower() == "blue"
                bf = [float(flag.split(',')[0].split('=')[-1]), float(flag.split(',')[1].split('=')[-1].split('\"')[0])]
            i += 1
        f.close()

        # Simulation environment
        self.sim_timestep = 0.4      # moostime (sec) between steps
        self.sim_time_limit = 180.0  # moostime (sec) before terminating episode
        self.capture_radius = pyquaticus_config['catch_radius']
        self.flag_range = 10.0

        # The zones and positions are generated by jerop.sh in the mission folder
        # and stored in region_info.txt
        # The flag positions are defined in shoreside/launch_shoreside.sh
        # (propagated to targ_shoreside.moos) and must be matched here
        self.blue_flag = bf#[140.0, 190.0]
        self.red_flag =  rf#[20.0,  190.0]

        # Zone order in dumped info is clockwise from north west by default (upper left in Pyquaticus terminology)
        self.blue_zone_ul = blue_zone[0]
        self.blue_zone_ur = blue_zone[1]
        self.blue_zone_lr = blue_zone[2]
        self.blue_zone_ll = blue_zone[3]

        self.red_zone_ul = red_zone[0]
        self.red_zone_ur = red_zone[1]
        self.red_zone_lr = red_zone[2]
        self.red_zone_ll = red_zone[3]

        num_rotations = 0
        while not self._set_boundary_and_scrimmage():
            # keep rotating until the scrimmage points make sense
            self._rotate_zone_points()
            num_rotations += 1
            if num_rotations > 3:
                raise RuntimeError("Cannot figure out boundary and scrimmage lines based on zone points given (likely an orientation issue).")

        print("############### Inferred Region Configuration ##################")
        print("Blue Zone:")
        print(self.blue_zone_ur)
        print(self.blue_zone_lr)
        print(self.blue_zone_ll)
        print(self.blue_zone_ul)
        print("Red Zone:")
        print(self.red_zone_ur)
        print(self.red_zone_lr)
        print(self.red_zone_ll)
        print(self.red_zone_ul)
        print("Boundaries:")
        print(self.boundary_ur)
        print(self.boundary_lr)
        print(self.boundary_ll)
        print(self.boundary_ul)
        print("Scrimmage:")
        print(self.scrimmage_pnts)

        self.speed_bounds = (0.0, 3.)
        self.heading_bounds = (0.0, 360.0)

        self.tagging_cooldown = 10.0

        # MOOS Timewarp is the simulation speed-up factor
        # This must match the value in your simulation script
        self.moos_timewarp = 4.0

    def _set_boundary_and_scrimmage(self):
        """
        Sets the boundaries and scrimmage points assuming the blue_zone_* and red_zone_*
        have already been populated.

        Returns a boolean indicating success.
        """
        if (abs(self.blue_zone_ur[1] - self.red_zone_ul[1]) < 1e-2 and
            abs(self.blue_zone_ur[0] - self.red_zone_ul[0]) < 1e-2):
            print("Blue zone to the left of red zone")
            self.scrimmage_pnts = [self.red_zone_ll, self.red_zone_ul]
            self.boundary_ul = self.blue_zone_ul
            self.boundary_ur = self.red_zone_ur
            self.boundary_ll = self.blue_zone_ll
            self.boundary_lr = self.red_zone_lr
            return True
        elif (abs(self.blue_zone_ul[1] - self.red_zone_ur[1]) < 1e-2 and
              abs(self.blue_zone_ul[0] - self.red_zone_ur[0]) < 1e-2):
            print("Red zone to the left of the blue zone")
            self.scrimmage_pnts = [self.red_zone_lr, self.red_zone_ur]
            self.boundary_ul = self.red_zone_ul
            self.boundary_ur = self.blue_zone_ur
            self.boundary_ll = self.red_zone_ll
            self.boundary_lr = self.blue_zone_lr
            return True
        else:
            return False

    def _rotate_zone_points(self):
        """
        Rotates the zone points for red and blue zone clockwise once.
        """
        self.blue_zone_ur, self.blue_zone_lr, self.blue_zone_ll, self.blue_zone_ul = \
        self.blue_zone_ul, self.blue_zone_ur, self.blue_zone_lr, self.blue_zone_ll
        self.red_zone_ur, self.red_zone_lr, self.red_zone_ll, self.red_zone_ul = \
        self.red_zone_ul, self.red_zone_ur, self.red_zone_lr, self.red_zone_ll

# for backwards compatibility with assumed docker folder structure
class JervisBayConfig(FieldReaderConfig):
    def __init__(self):
        super(f"os.path.dirname(__file__)/../../../moos-ivp-aquaticus/missions/jervis-2023")