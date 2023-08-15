# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
#Configureable Rewards
params{
    # -- NOTE --
    #   All bearings are in nautical format
    #                 0
    #                 |
    #          270 -- . -- 90
    #                 |
    #                180
    #
    # This can be converted the standard bearing format that is counterclockwise
    # using the heading_angle_conversion(deg) function found in utils.py
    #
    #
    #
    #
    #
    "num_players": int, #Number of players currently in the game
    "num_teammates": int, #Number of teammates currently in the game
    "num_opponents": int, #Number of opponents currently in the game
    "agent_id": int, #ID of agent rewards are being computed for
    "capture_radius": int, #The radius required to grab, capture a flag; and tag opponents
    "team_flag_pickup": bool,    # Indicates if team grabs flag
    "team_flag_capture": bool,   # Indicates if team captures flag
    "opponent_flag_pickup": bool, # Indicates if opponent grabs flag 
    "opponent_flag_capture": bool, #Indicates if opponent grabs flag
    "team_flag_home": float, #Agents distance to flag home (to untag)
    "team_flag_bearing": float, # Agents bearing to team flag
    "team_flag_distance": float, # Agents distance to team flag
    "opponent_flag_bearing": float, # Agents bearing to opponents flag
    "opponent_flag_distance": float, #Agents distance to opponents flag
    "speed": float, #Agents current speed
    "tagging_cooldown": bool, # True when agents tag is on cooldown False otherwise
    "thrust": float, # Agents current thrust
    "has_flag": bool, #Indicates if agent currently has opponents flag
    "on_own_side": bool, #Indicates if agent is currently on teams side
    "heading": float, #Agents yaw in degrees
    "wall_0_bearing": float, #Agents bearing towards 
    "wall_0_distance": float, #Agents distance towards
    "wall_1_bearing": float, #Agents bearing towards 
    "wall_1_distance": float, #Agents distance towards
    "wall_2_bearing": float, #Agents bearing towards
    "wall_2_distance": float, #Agents distance towards
    "wall_3_bearing": float, #Agents bearing towards
    "wall_3_distance": float, #Agents distance towards
    "wall_distances": Dict, (wallid, distance)
    "agent_captures": list, #List of agents that agent has tagged 0 not tagged 1 tagged by agent
    "agent_tagged": list, #List of agents 0 not tagged 1 tagged
    #Teamates First where n is the teammate ID
    "teammate_n_bearing": float, #Agents yaw towards teammate
    "teammate_n_distance": float, #Agents distance towards teammate
    "teammate_n_relative_heading": float, #Teammates current yaw value
    "teammate_n_speed": float, #Teammates current speed
    "teammate_n_has_flag": bool, # True if teammate currently has opponents flag
    "teammate_n_on_side": bool, # True if teammate is on teams side
    "teammate_n_tagging_cooldown": float, #Current value for tagging cooldown
    #Opponents
    "opponent_n_bearing": float, #Agents yaw towards opponent
    "opponent_n_distance": float, #Agents distance towards opponent
    "opponent_n_relative_heading": float, #Opponent current yaw value
    "opponent_n_speed": float, #Opponent current speed
    "opponent_n_has_flag": bool, # True if opponent currently has opponents flag
    "opponent_n_on_side": bool, # True if opponent is on their side
    "opponent_n_tagging_cooldown": float, #Current value for tagging cooldown.

}

#prev_params is the parameters from the previous step and have the same keys as params
"""

import math

def sparse(self, params, prev_params):
    reward = 0
    # Penalize player for opponent grabbing team flag
    if params["opponent_flag_pickup"] and not prev_params["opponent_flag_pickup"]:
        reward += -50
    # Penalize player for opponent successfully capturing team flag
    if params["opponent_flag_capture"] and not prev_params["opponent_flag_capture"]:
        reward +=  -100
    # Reward player for grabbing opponents flag
    if params["team_flag_pickup"] and not prev_params["team_flag_pickup"]:
        reward += 50
    # Reward player for capturing opponents flag
    if params["team_flag_capture"] and not prev_params["team_flag_capture"]:
        reward += 100
    # Check to see if agent was tagged
    if params["agent_tagged"][params["agent_id"]]:
        if prev_params["has_flag"]:
            reward += -100
        else:
            reward += -50
    # Check to see if agent tagged an opponent
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
        if prev_params["opponent_" + str(tagged_opponent) + "_has_flag"]:
            reward += 50
        else:
            reward += 100
    # Penalize agent if it went out of bounds (Hit border wall)
    if params["agent_oob"][params["agent_id"]] == 1:
        reward -= 100

    return reward


def custom_v1(self, params, prev_params):
    return 0
