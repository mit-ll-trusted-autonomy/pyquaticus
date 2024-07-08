import copy
import numpy as np

from pyquaticus.utils.utils import get_screen_res

EQUATORIAL_RADIUS = 6378137.0 # meters (https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)
POLAR_RADIUS = 6356752.0 # meters (https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)
MAX_SPEED = 1.5 # meters / s

config_dict_std = {
    "gps_env": False, # real world game location
    "env_bounds": [160.0, 80.0], # meters [xmax, ymax], lat/lon [(south, west), (north, east)], web mercator xy [(xmin, ymin), (xmax, ymax)], or "auto"
    "env_bounds_unit": "m", # "m", "wm_xy" (web mercator xy), "ll" (lat/lon)
    "blue_flag_home": "auto", # coordinates (lat, lon), list of coordinates, or "auto"
    "red_flag_home": "auto", # coordinates (lat, lon), list of coordinates, or "auto"
    "flag_homes_unit": "m", # "m" (meters relative to environment origin), "wm_xy" (web mercator xy), or "ll" (lat/lon)
    "scrimmage_coords": "auto", # [(coord1_x, coord1_y), (coord2_x, coord2_y)] or "auto"
    "scrimmage_coords_unit": "m", # "m" (meters relative to environment origin), "wm_xy" (web mercator xy), or "ll" (lat/lon)
    "topo_contour_eps": 0.001, # tolerance for error between approximate and true contours dividing water and land
    "agent_radius": 2.0,  # meters
    "catch_radius": 10.0,  # meters
    "flag_keepout": 5.0,  # minimum distance (meters) between agent and flag centers
    "tau": (
        1 / 10
    ),  # max dt (seconds) for updating the simulation
    "sim_speedup_factor": 1, # simulation speed multiplier (integer >= 1)
    "max_time": 120.0,  # maximum time (seconds) per episode
    "max_score": 1,     # maximum score per episode (until a winner is declared)
    "screen_frac": 0.75, # fraction of screen to use for rendering
    "max_screen_size": get_screen_res(),
    "render_fps": 30,
    "normalize": True,  # Flag for normalizing the observation space.
    "tagging_cooldown": (
        30.0
    ),  # Cooldown on an agent (seconds) after they tag another agent, to prevent consecutive tags
    # MOOS dynamics parameters
    "max_speed": 1.5,  # meters / s
    "speed_factor": 20.0,  # Multiplicative factor for desired_speed -> desired_thrust
    "thrust_map": np.array(  # Piecewise linear mapping from desired_thrust to speed
        [[-100, 0, 20, 40, 60, 80, 100], [-2, 0, 1, 2, 3, 5, 5]]
    ),
    "max_thrust": 70,  # Limit on vehicle thrust
    "max_rudder": 100,  # Limit on vehicle rudder actuation
    "turn_loss": 0.85,
    "turn_rate": 70,
    "max_acc": 1,  # m / s**2
    "max_dec": 1,  # m / s**2
    "suppress_numpy_warnings": (
        True  # Option to stop numpy from printing warnings to the console
    ),
    "teleport_on_tag": False, # Option for the agent when tagged, either out of bounds or by opponent, to teleport home or not
    "tag_on_wall_collision": False, # Option for setting the agent ot a tagged state upon wall collsion
    "render_field_points": False, #Debugging lets you see where the field points are on the field
    "obstacles": None, # Optional dictionary of obstacles in the enviornment
    # Notes: obstacles are specified via dictionary. Keys are the obstacle type ("circle" or "polygon"). 
    # Values are the parameters for the obstacle. 
    # Note: For circles, it should be a list of tuples: (radius, (center_x, center_y)) all in meters
    # Note: For polygons, it should be a list of tuples: ((x1, y1), (x2, y2), (x3, y3), ..., (xn, yn)) all in meters
    # Note for polygons, there is an implied edge between (xn, yn) and (x1, y1), to complete the polygon.

}
#Build Point Field - Taken from Moos-IvP-Aquaticus Michael Benjamin
#=================================================================
# Relative to ownship playing end
#=================================================================
#
#                   PBX        CBX       SBX
#   PPBX o---------o----------o---------o----------o SSBX  Row BX
#        |                    |                    |
#        |          PFX       |CFX       SFX       |
#   PPFX o---------o----------o---------o----------o SSFX  Row FX
#        |                    |                    |
#        |          PHX       |CHX       SHX       |
#   PPHX o---------o----------o---------o----------o SSHX  Row HX
#        |                    |                    |
#        |          PMX       |CMX       SMX       |
#   PPMX o---------o----------o---------o----------o SSMX  Row MX
#        |                    |                    |
#        |          PC        |CC        SC        |
#    PPC o=========o==========o=========o==========o SSC   Row C
#        |                    |                    |
#        |          PM        |CM        SM        |
#    PPM o---------o----------o---------o----------o SSM   Row M
#        |                    |                    |
#        |          PH        |CH        SH        |
#    PPH o---------o----------o---------o----------o SSH   Row H
#        |                    |                    |
#        |          PF        |CF        SF        |
#    PPF o---------o----------o---------o----------o SSF   Row F
#        |              ^     |     ^              |
#        |          PB  |     |CB   |    SB        |
#    PPB o---------o----------o---------o----------o SSB   Row B
inc_x = 0.125
inc_y = 0.25
config_dict_std["aquaticus_field_points"] ={"PPB":[0,inc_y*4],"PB":[0, inc_y * 3], "CB":[0, inc_y*2], "SB": [0, inc_y*1], "SSB":[0, 0],
                                            "PPF":[inc_x, inc_y*4], "PF":[inc_x, inc_y*3], "CF":[inc_x, inc_y*2], "SF":[inc_x, inc_y], "SSF":[inc_x, 0],
                                            "PPH":[inc_x*2, inc_y*4], "PH":[inc_x*2, inc_y*3], "CH":[inc_x*2, inc_y*2], "SH":[inc_x*2, inc_y], "SSH":[inc_x*2, 0],
                                            "PPM":[inc_x*3, inc_y*4], "PM":[inc_x*3, inc_y*3], "CM":[inc_x*3, inc_y*2], "SM":[inc_x*3, inc_y], "SSM":[inc_x*3, 0],
                                            "PPC":[inc_x*4, inc_y*4], "PC":[inc_x*4, inc_y*3], "CC":[inc_x*4, inc_y*2], "SC":[inc_x*4, inc_y], "SSC":[inc_x*4, 0],
                                            "PPMX":[inc_x*5, inc_y*4], "PMX":[inc_x*5, inc_y*3], "CMX":[inc_x*5, inc_y*2], "SMX":[inc_x*5, inc_y], "SSMX":[inc_x*5, 0],
                                            "PPHX":[inc_x*6, inc_y*4], "PHX":[inc_x*6, inc_y*3], "CHX":[inc_x*6, inc_y*2], "SHX":[inc_x*6, inc_y], "SSHX":[inc_x*6, 0],
                                            "PPFX":[inc_x*7, inc_y*4], "PFX":[inc_x*7, inc_y*3], "CFX":[inc_x*7, inc_y*2], "SFX":[inc_x*7, inc_y], "SSFX":[inc_x*7, 0],
                                            "PPBX":[inc_x*8, inc_y*4], "PBX":[inc_x*8, inc_y*3], "CBX":[inc_x*8, inc_y*2], "SBX":[inc_x*8, inc_y], "SSBX":[inc_x*8, 0]}


""" Standard configuration setup """


def get_std_config() -> dict:
    """Gets a copy of the standard configuration, ideal for minor modifications to the standard configuration."""
    return copy.deepcopy(config_dict_std)


# action space key combos
# maps discrete action id to (speed, heading)
ACTION_MAP = []
for spd in [MAX_SPEED, MAX_SPEED / 2.0]:
    for hdg in range(180, -180, -45):
        ACTION_MAP.append([spd, hdg])
# add a none action
ACTION_MAP.append([0.0, 0.0])
