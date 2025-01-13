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
    "team": int, #team ID (0 blue, 1 red)
    "num_players": int, #Number of players currently in the game
    "num_teammates": int, #Number of teammates currently in the game
    "num_opponents": int, #Number of opponents currently in the game
    "agent_id": str, #ID of agent rewards are being computed for
    "agent_id_index: int, # The index you should expect to find info relating to that agent in the agent_is_tagged or agent_made_tag attributes 
    "agent_oob": int, #0 indicates agent is not oob 1 indicates agent is oob
    "capture_radius": int, #The radius required to grab, capture a flag; and tag opponents
    "team_has_flag": bool,    # Indicates if team grabs flag
    "team_flag_capture": bool,   # Indicates if team captures flag
    "opponent_flag_pickup": bool, # Indicates if opponent grabs flag 
    "opponent_flag_capture": bool, #Indicates if opponent captures flag
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
    "agent_is_tagged": list, #List of agents 0 not tagged 1 tagged
    "agent_made_tag": list, #List of Nones until an agent makes a tag then the agent at index has tagged the ID of the agent that was tagged
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
    "agent_collisions": int #Current number of collisions for the current agent
    "all_agent_collisions": list #list of the number of agent collisions for all agents currently in game
}

#prev_params is the parameters from the previous step and have the same keys as params
"""

import math
#Sparse Rewards for all game events that occur
def sparse(self, params, prev_params):
    reward = 0
    #Team captured opponents flag
    if params['team_flag_capture'] and not prev_params['team_flag_capture']:
        reward += 1.0
    #Agent went out of bounds
    if params['agent_oob'] and not prev_params['agent_oob']:
        reward -= 1.0 
    #Agent grabbed opponents flag
    if params['team_has_flag'] and not prev_params['team_has_flag']:
        reward += 0.5 
    #Opposing team grabbed teams flag
    if params['opponent_flag_pickup'] and not prev_params['opponent_flag_pickup']:
        reward -= 0.5
    #Reward agent for capturing opposing teams flag
    if params['opponent_flag_capture'] and not prev_params['opponent_flag_capture']:
        reward -= 1.0
    #Reward Agent for tagging Opponent
    if not (params['agent_made_tag'][params['agent_id_index']] == None) and (prev_params['agent_made_tag'][params['agent_id_index']] == None):
        reward += 0.25
    #Penalize agent for getting tagged
    if (params['agent_is_tagged'][params['agent_id_index']]) and not (prev_params['agent_is_tagged'][params['agent_id_index']] and not prev_params['agent_oob']):
        reward -= 0.25
        
    return reward

#Reward Captures and Grabs Only
def caps_and_grabs(self, params, prev_params):
    reward = 0.0#-0.025
    #Team captured opponents flag
    if params['team_flag_capture'] and not prev_params['team_flag_capture']:
        reward += 1.0
     #Agent grabbed opponents flag
    if params['team_has_flag'] and not prev_params['team_has_flag']:
        reward += 0.5
    #Agent went out of bounds
    if params['agent_oob'] and not prev_params['agent_oob']:
        reward -= 1.0

    #Opposing team grabbed teams flag
    if params['opponent_flag_pickup'] and not prev_params['opponent_flag_pickup']:
        reward -= 0.5
    #Opposing team captures teams flag
    if params['opponent_flag_capture'] and not prev_params['opponent_flag_capture']:
        reward -= 1.0
    #Add sloped reward towards opposing teams flag when it has not been grabbed by the team.
    if not params['team_has_flag'] and not params['has_flag']:
        # Max reward when on the flag 0.45 slopes to 0 at 160m 7 = -0.003x + 0.45
        reward += (-params['opponent_flag_distance'] * 0.003 + 0.45) /8
    return reward
