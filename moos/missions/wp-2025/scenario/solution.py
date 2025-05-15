import argparse
import math
import numpy as np
#Need an added import for competition submission?
#Post an issue to the github and we will work to get it added into the system!

#NOTE: You are only allowed to change the gen_config OBS params specified
# Changing additional variables will result in disqualification of that entry

#YOUR CODE HERE
moos_role_map = {
    'b1': 'blue_one',
    'b2': 'blue_two',
    'b3': 'blue_three',
    'r1': 'red_one',
    'r2': 'red_two',
    'r3': 'red_three'
}
moos_port_map = {
    'b1': 9015,
    'b2': 9016,
    'b3': 9017,
    'r1': 9011,
    'r2': 9012,
    'r3': 9013
}
agent_id_moos_map = {
    'agent_0': 'blue_one',
    'agent_1': 'blue_two',
    'agent_2': 'blue_three',
    'agent_3': 'red_one',
    'agent_4': 'red_two',
    'agent_5': 'red_three'
}
BLUE_IDS = ["blue_one", "blue_two", "blue_three"]
RED_IDS = ["red_one", "red_two", "red_three"]


def angle180(deg):
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg

def heading_angle_conversion(deg):
    """
    Converts a world-frame angle to a heading and vice-versa
    The transformation is its own inverse
    Args:
        deg: the angle (heading) in degrees
    Returns:
        float: the heading (angle) in degrees.
    """
    return (90 - deg) % 360

def vector_to(A, B, unit=False):
    """
    Returns a vector from A to B
    if unit is true, will scale the vector.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    vec = B - A

    if unit:
        norm = np.linalg.norm(vec)
        if norm >= 1e-5:
            vec = vec / norm

    return vec

def vec_to_mag_heading(vec):
    """Converts a vector to a magnitude and heading (deg)."""
    mag = np.linalg.norm(vec)
    angle = math.degrees(math.atan2(vec[1], vec[0]))
    return mag, angle180(heading_angle_conversion(angle))

def mag_bearing_to(A, B, relative_hdg=None):
    """
    Returns magnitude and bearing vector between two points.

    if relative_hdg provided then return bearing relative to that

    Returns
    -------
        float: distance
        float: bearing in [-180, 180] which is relative
               to relative_hdg if provided and otherwise
               is global
    """
    mag, hdg = vec_to_mag_heading(vector_to(A, B))
    if relative_hdg is not None:
        hdg = (hdg - relative_hdg) % 360
    return mag, angle180(hdg)


class Defender:
    def __init__(self, agent_id, max_speed, protect_id, defender_ids, opp_ids, tag_radius, protect_buffer):
        self.agent_id = agent_id
        self.max_speed = max_speed
        self.protect_id = protect_id
        self.defender_ids = defender_ids
        self.opp_ids = opp_ids
        self.tag_radius = tag_radius
        self.protect_buffer = protect_buffer
    
    def compute_action(self, global_state):
        if not global_state[(agent_id_moos_map[self.agent_id], "is_tagged")]:
            pos = global_state[(agent_id_moos_map[self.agent_id], "pos")]
            heading = global_state[(agent_id_moos_map[self.agent_id], "heading")]

            defenders_tagged = np.array([global_state[(agent_id_moos_map[agent_id], "is_tagged")] for agent_id in self.defender_ids])
            valid_defender_ids = np.array([agent_id for i, agent_id in enumerate(self.defender_ids) if not defenders_tagged[i]])
            defender_poses = np.array([global_state[(agent_id_moos_map[agent_id], "pos")] for agent_id in valid_defender_ids])

            protect_pos = global_state[(agent_id_moos_map[self.protect_id], "pos")]
            protect_dist = np.linalg.norm(protect_pos - pos)

            opp_tagged = np.array([global_state[(agent_id_moos_map[agent_id], "is_tagged")] for agent_id in self.opp_ids])
            valid_opp_ids = np.array([agent_id for i, agent_id in enumerate(self.opp_ids) if not opp_tagged[i]])
            opp_poses = np.array([global_state[(agent_id_moos_map[agent_id], "pos")] for agent_id in valid_opp_ids])

            if not np.all(opp_tagged):
                opp_dists_to_targ = np.linalg.norm(protect_pos - opp_poses, axis=-1)
                opp_idx_closest_to_targ = np.argmin(opp_dists_to_targ)

                defender_dists_to_opp_closest_to_targ = np.linalg.norm(opp_poses[opp_idx_closest_to_targ] - defender_poses, axis=-1)
                defender_id_closest_to_opp_closest_to_targ = valid_defender_ids[np.argmin(defender_dists_to_opp_closest_to_targ)]

                if defender_id_closest_to_opp_closest_to_targ == self.agent_id:
                    if len(valid_defender_ids) > 1: 
                        goal_dist, goal_bearing = mag_bearing_to(pos, opp_poses[opp_idx_closest_to_targ], relative_hdg=heading)
                        tag = (goal_dist <= self.tag_radius) and (protect_dist > self.protect_buffer)
                    else:
                        opp_poses = np.array([opp_pos for i, opp_pos in enumerate(opp_poses)])
                        goal_pos = np.mean(0.5*protect_pos + 0.5*opp_poses, axis=0)
                        goal_dist, goal_bearing = mag_bearing_to(pos, goal_pos, relative_hdg=heading)

                        opp_dists = np.linalg.norm(pos - opp_poses, axis=-1)
                        tag = np.all(opp_dists <= self.tag_radius) and (protect_dist > self.protect_buffer)
                else:
                    #go to midpoint between target and other agents
                    other_opp_poses = np.array([opp_pos for i, opp_pos in enumerate(opp_poses) if i != opp_idx_closest_to_targ])
                    if len(other_opp_poses) > 0:
                        if len(other_opp_poses) > 1:
                            goal_pos = np.mean(0.5*protect_pos + 0.5*other_opp_poses, axis=0)
                            goal_dist, goal_bearing = mag_bearing_to(pos, goal_pos, relative_hdg=heading)

                            other_opp_dists = np.linalg.norm(pos - other_opp_poses, axis=-1)
                            tag = np.all(other_opp_dists <= self.tag_radius) and (protect_dist > self.protect_buffer)
                        else:
                            goal_pos = 0.5*protect_pos + 0.5*other_opp_poses[0]
                            goal_dist, goal_bearing = mag_bearing_to(pos, goal_pos, relative_hdg=heading)

                            opp_dist = np.linalg.norm(pos - other_opp_poses[0])
                            tag = (opp_dist <= self.tag_radius) and (protect_dist > self.protect_buffer)
                    else:
                        goal_pos = 0.5*protect_pos + 0.5*opp_poses[opp_idx_closest_to_targ]
                        goal_dist, goal_bearing = mag_bearing_to(pos, goal_pos, relative_hdg=heading)

                        opp_dist = np.linalg.norm(pos - opp_poses[opp_idx_closest_to_targ])
                        tag = (opp_dist <= self.tag_radius) and (protect_dist > self.protect_buffer)

                if goal_dist <= self.tag_radius:
                    return [0., goal_bearing], tag
                else:
                    return [self.max_speed, goal_bearing], tag
            else:
                return [0., 0.], False
        else:
            return [0., 0.], False

class A:
    def __init__(self, agent_id, max_speed):
        self.agent_id = agent_id
        self.max_speed = max_speed
    
    def compute_action(self, global_state):
        pos = global_state[(agent_id_moos_map[self.agent_id], "pos")]
        heading = global_state[(agent_id_moos_map[self.agent_id], "heading")]

        goal_pos = global_state["red_flag_home"]
        goal_dist, goal_bearing = mag_bearing_to(pos, goal_pos, relative_hdg=heading)

        return [self.max_speed, goal_bearing], False


class Attacker:
    def __init__(self, agent_id, max_speed, targ_agent_id, tag_radius):
        self.agent_id = agent_id
        self.max_speed = max_speed
        self.targ_agent_id = targ_agent_id
        self.tag_radius = tag_radius
    
    def compute_action(self, global_state):
        pos = global_state[(agent_id_moos_map[self.agent_id], "pos")]
        heading = global_state[(agent_id_moos_map[self.agent_id], "heading")]

        goal_pos = global_state[(agent_id_moos_map[self.targ_agent_id], "pos")]
        goal_dist, goal_bearing = mag_bearing_to(pos, goal_pos, relative_hdg=heading)

        tag = False
        if goal_dist <= self.tag_radius:
            tag = True

        return [self.max_speed, goal_bearing], tag


#Load in your trained model and return the corresponding agent action based on the information provided in step()
class solution:
	#Add Variables required for solution
	
    def __init__(self):
        ### load policies ###############################################################
        self.policy_dict  = {
            "agent_0": A("agent_0", 1.0),
            "agent_1": Defender(
                agent_id=f"agent_1",
                max_speed=2.0,
                protect_id="agent_0",
                defender_ids=["agent_1", "agent_2"],
                opp_ids=["agent_3", "agent_4", "agent_5"],
                tag_radius=14,
                protect_buffer=15
            ),
            "agent_2": Defender(
                agent_id=f"agent_2",
                max_speed=2.0,
                protect_id="agent_0",
                defender_ids=["agent_1", "agent_2"],
                opp_ids=["agent_3", "agent_4", "agent_5"],
                tag_radius=14,
                protect_buffer=15
            ),
            "agent_3": Attacker(
                agent_id=f"agent_3",
                max_speed=2.0,
                targ_agent_id="agent_0",
                tag_radius=14
            ),
            "agent_4": Attacker(
                agent_id=f"agent_4",
                max_speed=2.0,
                targ_agent_id="agent_0",
                tag_radius=14
            ),
            "agent_5": Attacker(
                agent_id=f"agent_5",
                max_speed=2.0,
                targ_agent_id="agent_0",
                tag_radius=14
            ), 
        }

	#Given an observation return a valid action agent_id is agent that needs an action, observation space is the current normalized observation space for the specific agent
    def compute_action(self,agent_id:str, full_obs_normalized:dict, full_obs:dict, global_state:dict):
        action, tag = self.policy_dict[agent_id].compute_action(global_state)
        return [action[0], action[1], tag]

#END OF CODE SECTION
