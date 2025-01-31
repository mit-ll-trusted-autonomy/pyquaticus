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
    # -- NOTE --
    #   All headings are in nautical format
    #                 0
    #                 |
    #          270 -- . -- 90
    #                 |
    #                180
    #
    # This can be converted the standard heading format that is counterclockwise
    # using the heading_angle_conversion(deg) function found in utils.py
    #
    #
    ## Each custom reward function should have the following arguments ##
    Args:
        agent_id (int): Agent ID we are computing the sparse reward for
        agents (list): List of agent ID's (this is used to find the index of our agent_id)
        state (dict):
            'agent_position' (list): List of agent positions in the order of (agents list)
                                    Each Index in the list are positions [x,y]

                        Ex. Usage: Get agent_id's current position
                        agent_id = 'agent_1'
                        position = state['agent_position'][agents.index(agent_id)]

            'prev_agent_position' (list): Contains the prev position [x,y] for agent at the specified index
                        Ex. Usage: Get agent_id's previous position
                        agent_id = 'agent_1'
                        prev_position = state['prev_agent_position'][agents.index(agent_id)]

            'agent_speed' (list): Each element represents the speed of the agent in the agents list order

                        Ex. Usage: Get agent_id's speed
                        agent_id = 'agent_1'
                        speed = state

            'agent_heading' (list): List of headings for each agent in the agents list order

                        Ex. Usage: Get agent_id's heading
                        agent_id = 'agent_1'
                        heading = state['agent_heading'][agents.index(agent_id)]

            'agent_oob' (list): List of all agents and the number of times they have                                driven OOB in order of agents list
                        
                        Ex. Usage: Check the number of times agent_id has gone OOB
                        agent_id = 'agent_1'
                        num_oob = state['agent_oob'][agents.index(agent_id)]
            
            'agent_has_flag' (list):


                        Ex. Usage: Check if agent_id has opponents flag
                        agent_id = 'agent_1'
                        has_flag = state['agent_has_flag'][agents.index(agent_id)] == 1
            'agent_made_tag' (list):  list of all agents and if that agent tagged something at the current timestep (will be index of tagged agent if so) otherwise None
                        Ex. Usage: Check of agent_id has tagged an agent
                        agent_id = 'agent_1'
                        tagged_opponent = state['agent_made_tag'][agents.index(agent_id)]

            'agent_tagging_cooldown (list): Current tagging cooldown for all agents in the agents list order
                        Note: 0.0 means agent currently has a tag available                    
    
                        Ex. Usage: Get agent_id current tagging cooldown
                        agent_id = 'agent_1'
                        cooldown = self.state['agent_tagging_cooldown'][agents.index(agent_id)]
                        

            'dist_bearing_to_obstacles' (dict): For each agent in game list out distances and bears to all obstacles in game in order of obstacles list
                        Note: Not Used for the 2025 MCTF competition
            
            TODO: Add Team mapping for each agent to the state

            'flag_home' (list):

            'flag_position' (list):

            'flag_taken' (list):

            'team_has_flag' (list):

            'captures' (list):

            'tags' (list):

            'grabs' (list):

        prev_state (dict): Contains the state information from the previous step

"""

import math
#Sparse Rewards for all game events that occur
def sparse(self, agent_id, agents, state, prev_state):
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
def caps_and_grabs(self, agent_id, agents, state, prev_state):
    reward = 0.0
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
    return reward
