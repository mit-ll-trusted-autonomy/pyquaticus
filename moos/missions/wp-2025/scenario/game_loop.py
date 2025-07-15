import argparse
import math
import numpy as np
from pyquaticus.moos_bridge.config import WestPoint2025, pyquaticus_config_std
from pyquaticus.moos_bridge.pyquaticus_moos_bridge_ext import PyQuaticusMoosBridgeFullObs


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

def dist(A, B):
    return np.linalg.norm(B - A)

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

def get_avoid_vect(avoid_pos, avoid_threshold=10.0):
    """
    This function finds the vector most pointing away to all enemy agents.

    Args:
        agent: An agents position
        avoid_pos: All other agent (polar) positions we potentially need to avoid.
        avoid_threshold: The threshold that, when the agent is closer than this range,
            is attempted to be avoided.

    Returns
    -------
        np.array vector that points away from as many agents as possible
    """
    avoid_vects = []
    need_avoid = False
    for avoid_ag in avoid_pos:
        if avoid_ag[0] < avoid_threshold:
            coeff = np.divide(avoid_threshold, avoid_ag[0])
            ag_vect = rel_bearing_to_local_unit_rect(avoid_ag[1])
            avoid_vects.append([coeff * ag_vect[0], coeff * ag_vect[1]])
            need_avoid = True
    av_x = 0.0
    av_y = 0.0
    if need_avoid:
        for vects in avoid_vects:
            av_x += vects[0]
            av_y += vects[1]
        norm = np.linalg.norm(np.array([av_x, av_y]))
        final_avoid_unit_vect = np.array(
            [-1.0 * np.divide(av_x, norm), -1.0 * np.divide(av_y, norm)]
        )
    else:
        final_avoid_unit_vect = np.array([0, 0])

    return final_avoid_unit_vect

def unit_vect_between_points(start: np.ndarray, end: np.ndarray):
    """Calculates the unit vector between two rectangular points."""
    return (end - start) / np.linalg.norm(end - start)

def global_rect_to_abs_bearing(vec):
    """Calculates the absolute bearing of a rectangular vector in the global frame."""
    return angle180(90 - np.degrees(np.arctan2(vec[1], vec[0])))

def dist_rel_bearing_to_local_rect(distance, rel_bearing):
    """Calculates the local frame rectangular vector given a distance and relative bearing."""
    return distance * rel_bearing_to_local_unit_rect(rel_bearing)

def rel_bearing_to_local_unit_rect(rel_bearing):
    """Calculates the local frame rectangular unit vector in the direction of the given relative bearing."""
    rad = np.deg2rad(rel_bearing)
    return np.array((np.sin(rad), np.cos(rad)))

def local_rect_to_rel_bearing(vec):
    """Calculates the relative bearing of a rectangular vector in the local frame."""
    return global_rect_to_abs_bearing(vec)

def global_rect_to_local_rect(global_vec, ego_pos, ego_heading):
    """Converts a rectangular vector in the global frame to one in the local frame."""
    distance = dist(global_vec, ego_pos)
    abs_bearing = global_rect_to_abs_bearing(global_vec - ego_pos)
    rel_bearing = abs_bearing - ego_heading
    return dist_rel_bearing_to_local_rect(distance, rel_bearing)


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
    """This is a Policy class that contains logic for capturing the flag."""

    def __init__(self, agent_id, max_speed, tag_radius, env_size, team: str, targ_agent_id: list, opp_ids: list):
        self.id = agent_id
        self.max_speed = max_speed
        self.env_size = env_size
        self.team = team
        self.tag_radius = tag_radius

        self.targ_id = targ_agent_id
        self.opponent_ids = opp_ids

    def compute_action(self, global_state):
        """
        Compute an action from the given observation and global state.

        Args:
            obs: observation from the gym
            info: info from the gym

        Returns
        -------
            action: if continuous, a tuple containing desired speed and heading error.
            if discrete, an action index corresponding to ACTION_MAP in config.py
        """

        self.update_state(global_state)

        # If I'm close to a wall, add the closest point to the wall as an obstacle to avoid
        if self.wall_distances[0] < 10 and (-90 < self.wall_bearings[0] < 90):
            self.opp_team_pos.append(
                (
                    self.wall_distances[0],
                    self.wall_bearings[0],
                )
            )
        elif self.wall_distances[2] < 10 and (-90 < self.wall_bearings[2] < 90):
            self.opp_team_pos.append(
                (
                    self.wall_distances[2],
                    self.wall_bearings[2],
                )
            )
        if self.wall_distances[1] < 10 and (-90 < self.wall_bearings[1] < 90):
            self.opp_team_pos.append(
                (
                    self.wall_distances[1],
                    self.wall_bearings[1],
                )
            )
        elif self.wall_distances[3] < 10 and (-90 < self.wall_bearings[3] < 90):
            self.opp_team_pos.append(
                (
                    self.wall_distances[3],
                    self.wall_bearings[3],
                )
            )

        # Increase the avoidance threshold to start avoiding when farther away
        avoid_thresh = 30.0

        # Otherwise go get the flag
        goal_vect = rel_bearing_to_local_unit_rect(self.opp_flag_bearing)
        avoid_vect = get_avoid_vect(
            self.opp_team_pos, avoid_threshold=avoid_thresh
        )
        if (not np.any(goal_vect + (avoid_vect))) or (
            np.allclose(
                np.abs(np.abs(goal_vect) - np.abs(avoid_vect)),
                np.zeros(np.array(goal_vect).shape),
                atol=1e-01,
                rtol=1e-02,
            )
        ):
            # Special case where a player is closely in line with the goal
            # vector such that the calculated avoid vector nearly negates the
            # action (the player is in a spot that causes the agent to just go
            # straight into them). In this case just start going towards the top
            # or bottom boundary, whichever is farthest.

            top_dist = self.wall_distances[3]
            bottom_dist = self.wall_distances[1]

            # Some bias towards the bottom boundary to force it to stick with a
            # direction.
            if top_dist > 1.25 * bottom_dist:
                my_action = dist_rel_bearing_to_local_rect(
                    top_dist, self.wall_bearings[0]
                )
            else:
                my_action = dist_rel_bearing_to_local_rect(
                    bottom_dist, self.wall_bearings[2]
                )
        else:
            my_action = np.multiply(1.25, goal_vect) + avoid_vect

        tag = False
        if self.opp_flag_distance <= self.tag_radius:
            tag = True

        return self.action_from_vector(my_action, 1), tag

    def action_from_vector(self, vector, desired_speed_normalized):
        if desired_speed_normalized == 0:
            return (0, 0)
        rel_bearing = local_rect_to_rel_bearing(vector)

        return (desired_speed_normalized * self.max_speed, rel_bearing)

    def update_state(self, global_state) -> None:
        """
        Method to convert the gym obs and info into data more relative to the
        agent.

        Note: all rectangular positions are in the ego agent's local coordinate frame.
        Note: all bearings are relative, measured in degrees clockwise from the ego agent's heading.

        Args:
            obs: observation from gym
            info: info from gym
        """

        my_pos = global_state[(agent_id_moos_map[self.id], "pos")]
        my_heading = global_state[(agent_id_moos_map[self.id], "heading")]

        self.has_flag = global_state[(agent_id_moos_map[self.id], "has_flag")]
        self.is_tagged = global_state[(agent_id_moos_map[self.id], "is_tagged")]

        # Calculate the rectangular coordinates for the flags location relative to the agent.
        team_str = self.team
        opp_str = "red" if team_str == "blue" else "blue"

        self.opp_flag_distance = dist(my_pos, global_state[(agent_id_moos_map[self.targ_id], "pos")])
        self.opp_flag_bearing = angle180(
            global_rect_to_abs_bearing(global_state[(agent_id_moos_map[self.targ_id], "pos")] - my_pos)
            - my_heading
        )
        self.opp_flag_loc = dist_rel_bearing_to_local_rect(
            self.opp_flag_distance, self.opp_flag_bearing
        )

        home_distance = dist(my_pos, global_state[team_str + "_flag_home"])
        self.home_bearing = angle180(
            global_rect_to_abs_bearing(global_state[team_str + "_flag_home"] - my_pos)
            - my_heading
        )
        self.home_loc = dist_rel_bearing_to_local_rect(home_distance, self.home_bearing)

        self.opp_team_pos = []
        for id in self.opponent_ids:
            distance = dist(my_pos, global_state[(agent_id_moos_map[id], "pos")])
            bearing = angle180(
                global_rect_to_abs_bearing(global_state[(agent_id_moos_map[id], "pos")] - my_pos)
                - my_heading
            )
            self.opp_team_pos.append(np.array((distance, bearing)))

        self.wall_distances = [my_pos[0], my_pos[1], self.env_size[0] - my_pos[0], self.env_size[1] - my_pos[1]]
        self.wall_bearings = [
            angle180(global_rect_to_abs_bearing(np.array([0, my_pos[1]]) - my_pos) - my_heading),
            angle180(global_rect_to_abs_bearing(np.array([my_pos[0], 0]) - my_pos) - my_heading),
            angle180(global_rect_to_abs_bearing(np.array([self.env_size[0], my_pos[1]]) - my_pos) - my_heading),
            angle180(global_rect_to_abs_bearing(np.array([my_pos[0], self.env_size[1]]) - my_pos) - my_heading)
        ]

# class Attacker:
#     def __init__(self, agent_id, max_speed, targ_agent_id, tag_radius):
#         self.agent_id = agent_id
#         self.max_speed = max_speed
#         self.targ_agent_id = targ_agent_id
#         self.tag_radius = tag_radius
    
#     def compute_action(self, global_state):
#         pos = global_state[(agent_id_moos_map[self.agent_id], "pos")]
#         heading = global_state[(agent_id_moos_map[self.agent_id], "heading")]

#         goal_pos = global_state[(agent_id_moos_map[self.targ_agent_id], "pos")]
#         goal_dist, goal_bearing = mag_bearing_to(pos, goal_pos, relative_hdg=heading)

#         tag = False
#         if goal_dist <= self.tag_radius:
#             tag = True

#         return [self.max_speed, goal_bearing], tag

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MOOS bridge for a given mission")
    parser.add_argument('-a', '--agent', default='h', choices=['b1','b2','b3', 'r1', 'r2', 'r3'], help='Select a pyquaticus base policy')
    args = parser.parse_args()

    agent_name = moos_role_map[args.agent]
    agent_port = moos_port_map[args.agent]

    if agent_name.startswith('blue'):
        team_names = [name for name in BLUE_IDS if name != agent_name]
        opponent_names = RED_IDS
    else:
        team_names = [name for name in RED_IDS if name != agent_name]
        opponent_names = BLUE_IDS

    env = PyQuaticusMoosBridgeFullObs(
        server="localhost",
        agent_name=agent_name,
        agent_port=agent_port,
        team_names=team_names,
        opponent_names=opponent_names,
        all_agent_names=BLUE_IDS + RED_IDS,
        moos_config=WestPoint2025(),
        pyquaticus_config=pyquaticus_config_std,
        timewarp=3,
        quiet=False
    )

    ### load policy ###############################################################
    if agent_name == "blue_one":
        policy = A("agent_0", 1.0)
    elif agent_name.startswith("blue"):
        policy = Defender(
            agent_id=f"agent_{int(args.agent[-1]) - 1}",
            max_speed=2.0,
            protect_id="agent_0",
            defender_ids=["agent_1", "agent_2"],
            opp_ids=["agent_3", "agent_4", "agent_5"],
            tag_radius=14,
            protect_buffer=15

        )
    else:
        policy = Attacker(
            agent_id=f"agent_{int(args.agent[-1]) + 2}",
            max_speed=2.0,
            env_size=[160, 80],
            team="red",
            targ_agent_id="agent_0",
            opp_ids=["agent_1", "agent_2"],
            tag_radius=14
        )


    ### run game loop #############################################################
    obs, info = env.reset()

    while True:
        action, tag = policy.compute_action(info["global_state"])
        _, _, _, _, info = env.step(action, tag)