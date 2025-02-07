import numpy as np
import os
from ray.rllib.policy.policy import Policy
#Need an added import for codalab competition submission?
#Post an issue to the github and we will work to get it added into the system!

#NOTE: You are only allowed to change the gen_config OBS params specified
# Changing additional variables will result in disqualification of that entry

#YOUR CODE HERE

#Load in your trained model and return the corresponding agent action based on the information provided in step()
class solution:
	#Add Variables required for solution
	
    def __init__(self):
        

        #Load in policy or anything else you want to load/do here
        #NOTE: You can only load from files that are in the same directory as the solution.py or a subdirectory
        
        #Load in learned policies see examples below:
        self.policy_one = Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')
        self.policy_two = Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')
        self.policy_three = Policy.from_checkpoint(os.path.dirname(os.path.realpath(__file__))+ '<Your Policy Path Here>')

	#Given an observation return a valid action agent_id is agent that needs an action, observation space is the current normalized observation space for the specific agent
    def compute_action(self,agent_id:int, full_obs_normalized:dict, full_obs:dict, global_state:dict):
        #WARNING: If using global state you must ensure your entry can run on both RED and BLUE sides
        # State includes actual coordinate positions which are not the same on each side
        if agent_id == 0:
            return self.policy_one.compute_single_action(observation_normalized[agent_id], explore=False)[0]
        elif agent_id == 2:
            return self.policy_two.compute_single_action(observation_normalized[agent_id], explore=False)[0]
        else:
            return self.policy_three.compute_single_action(observation_normalized[agent_id], explore=False)[0]

#END OF CODE SECTION
