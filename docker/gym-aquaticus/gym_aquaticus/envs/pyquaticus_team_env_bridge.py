# external imports
try:
    from pyquaticus.utils.obs_utils import ObsNormalizer
    from pyquaticus.envs.pyquaticus import ACTION_MAP
except ImportError as e:
    print("Missing Pyquaticus library. Please install from the repository with pip install -e .")
    raise e

from collections import OrderedDict
from gymnasium.spaces import Discrete, Dict
import numpy as np

# internal imports
from gym_aquaticus.envs.aquaticus_team_env import AquaticusTeamEnv
from gym_aquaticus.utils import zero_centered_heading, mag_heading_to, \
    closest_point_on_line, ArtificialPotentialField, mag_heading_to_vec, vec_to_mag_heading

class PyquaticusBridge(AquaticusTeamEnv):
    """
    This class is an AquaticusTeamEnv with modifications to run Pyquaticus policies.

    It communicates with MOOS using a pymoos client.
    """

    def __init__(self, env_config):
        """
        See AquaticusTeamEnv for documentation on env_config
        """
        if 'action_type' in env_config:
            assert env_config.action_type == 'discrete'
        else:
            env_config['action_type'] = 'discrete'
        super().__init__(env_config)
        self.agent_obs_normalizer = self._register_state_elements()
        # overwrite observation and action space
        self.action_space = Dict([
            (agent, self._define_agent_action_space()) for agent in self._agents
        ])
        self.observation_space = Dict([
            (agent, self._define_agent_observation_space(agent)) for agent in self._agents
        ])

    def step(self, raw_action):
        obs, reward, done, info = super().step(raw_action)
        if not isinstance(done, bool):
            # Pyquaticus doesn't split out done per-agent
            # just have a single done signal
            done = done['__all__']
        return obs, reward, done, info

    def _define_agent_observation_space(self, agent_name):
        """Overridden method inherited from `Gym`."""
        return self.agent_obs_normalizer.normalized_space

    def _define_agent_action_space(self):
        """Overridden method inherited from `Gym`."""
        return Discrete(len(ACTION_MAP))

    def state_to_obs(self, state):
        obs = {agent: self._agent_state_to_obs(state, agent) for agent in self._agents}
        return obs

    def _agent_state_to_obs(self, state, agent_name):
        """
        Returns a local observation space. These observations are
        based entirely on the agent local coordinate frame rather
        than the world frame.
        This was originally designed so that observations can be
        easily shared between different teams and agents.
        Without this the world frame observations from the blue and
        red teams are flipped (e.g., the goal is in the opposite
        direction)
        Observation Space (per agent):
            Retrieve flag relative bearing (clockwise degrees)
            Retrieve flag distance (meters)
            Home flag relative bearing (clockwise degrees)
            Home flag distance (meters)
            Wall 1 relative bearing (clockwise degrees)
            Wall 1 distance (meters)
            Wall 2 relative bearing (clockwise degrees)
            Wall 2 distance (meters)
            Wall 3 relative bearing (clockwise degrees)
            Wall 3 distance (meters)
            Wall 4 relative bearing (clockwise degrees)
            Wall 4 distance (meters)
            Own speed (meters per second)
            Own flag status (boolean)
            On side (boolean)
            Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            Is tagged (boolean)
            For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
                Bearing from you (clockwise degrees)
                Distance (meters)
                Heading of other agent relative to the vector to you (clockwise degrees)
                Speed (meters per second)
                Has flag status (boolean)
                On their side status (boolean)
                Tagging cooldown (seconds)
                Is tagged (boolean)
        Note 1 : the angles are 0 when the agent is pointed directly at the object
                 and increase in the clockwise direction
        Note 2 : the wall distances can be negative when the agent is out of bounds
        Note 3 : the boolean args Tag/Flag status are -1 false and +1 true
        Developer Note 1: changes here should be reflected in _register_state_elemnts.
        """
        agent = self._agents[agent_name]
        agent_states = state['agents']

        own_team = 'red' if agent_name in self._red_team else 'blue'
        other_team = 'blue' if own_team == 'red' else 'red'

        pnt, has_flag, on_side = agent_states[agent_name]
        np_pos = np.array([pnt.x, pnt.y], dtype=np.float32)

        opponent_home_loc, own_home_loc = self.get_team_goal_defend(own_team)

        obs = OrderedDict()
        # Goal flag
        opponent_home_dist, opponent_home_bearing = mag_heading_to(
            np_pos, opponent_home_loc, pnt.heading
        )
        # Defend flag
        own_home_dist, own_home_bearing = mag_heading_to(
            np_pos, own_home_loc, pnt.heading
        )

        # TODO: consider swapping goal location once flag is retrieved
        #       especially if we're bringing the flag all the way back

        obs["opponent_home_bearing"] = opponent_home_bearing
        obs["opponent_home_distance"] = opponent_home_dist
        obs["own_home_bearing"] = own_home_bearing
        obs["own_home_distance"] = own_home_dist

        # Walls
        wall_0_closest_point = closest_point_on_line(
            self.boundary_ul, self.boundary_ur, np_pos
        )
        wall_0_dist, wall_0_bearing = mag_heading_to(
            np_pos, wall_0_closest_point, pnt.heading
        )
        obs["wall_0_bearing"] = wall_0_bearing
        obs["wall_0_distance"] = wall_0_dist

        wall_1_closest_point = closest_point_on_line(
            self.boundary_ur, self.boundary_lr, np_pos
        )
        wall_1_dist, wall_1_bearing = mag_heading_to(
            np_pos, wall_1_closest_point, pnt.heading
        )
        obs["wall_1_bearing"] = wall_1_bearing
        obs["wall_1_distance"] = wall_1_dist

        wall_2_closest_point = closest_point_on_line(
            self.boundary_lr, self.boundary_ll, np_pos
        )
        wall_2_dist, wall_2_bearing = mag_heading_to(
            np_pos, wall_2_closest_point, pnt.heading
        )
        obs["wall_2_bearing"] = wall_2_bearing
        obs["wall_2_distance"] = wall_2_dist

        wall_3_closest_point = closest_point_on_line(
            self.boundary_ll, self.boundary_ul, np_pos
        )
        wall_3_dist, wall_3_bearing = mag_heading_to(
            np_pos, wall_3_closest_point, pnt.heading
        )
        obs["wall_3_bearing"] = wall_3_bearing
        obs["wall_3_distance"] = wall_3_dist

        # Own speed
        obs["speed"] = pnt.speed
        # Own flag status
        obs["has_flag"] = has_flag
        # True if on your own side
        obs["on_side"] = on_side
        # Resets to zero after you tag another agent then counts up to max (when you can tag again)
        obs["tagging_cooldown"] = pnt.tagging_cooldown
        # True if you are currently tagged
        obs["is_tagged"] = pnt.tagged

        # Relative observations to other agents
        # teammates first
        # TODO: consider sorting these by some metric
        #       in an attempt to get permutation invariance
        #       distance or maybe flag status (or some combination?)
        #       i.e. sorted by perceived relevance
        for team in [own_team, other_team]:
            dif_agents = filter(lambda name: name != agent_name, 
                                [n for n, _ in self.get_agents_gen(team)])
            for i, dif_agent_name in enumerate(dif_agents):
                entry_name = f"teammate_{i}" if team == own_team else f"opponent_{i}"


                dif_pnt, dif_has_flag, dif_on_side = agent_states[dif_agent_name]
                dif_np_pos = np.array([dif_pnt.x, dif_pnt.y], dtype=np.float32)

                dif_agent_dist, dif_agent_bearing = mag_heading_to(
                    np_pos, dif_np_pos, pnt.heading
                )
                _, hdg_to_agent = mag_heading_to(dif_np_pos, np_pos)
                hdg_to_agent = hdg_to_agent % 360
                # bearing relative to the bearing to you
                obs[(entry_name, "bearing")] = dif_agent_bearing
                obs[(entry_name, "distance")] = dif_agent_dist
                obs[(entry_name, "relative_heading")] = zero_centered_heading(
                    (dif_pnt.heading - hdg_to_agent) % 360
                )
                obs[(entry_name, "speed")] = dif_pnt.speed
                obs[(entry_name, "has_flag")] = dif_has_flag
                obs[(entry_name, "on_side")] = dif_on_side
                obs[(entry_name, "tagging_cooldown")] = dif_pnt.tagging_cooldown
                obs[(entry_name, "is_tagged")] = dif_pnt.tagged

        return self.agent_obs_normalizer.normalized(obs).flatten()

    def _process_actions(self, last_state, raw_act):
        '''
        Process a dictionary of actions.
        If raw_act is a single action meant for self._agent_to_train,
        then create a dictionary so it's in a standard format
        '''
        actions = OrderedDict()
        for name in raw_act:
            spd, hdg = ACTION_MAP[raw_act[name]]
            agent = self._agents[name]
            hdg = (agent.pnt.heading + hdg) % 360
            actions[name] = (spd, hdg)
        return actions

    def _register_state_elements(self):
        """Initializes the normalizer."""
        agent_obs_normalizer = ObsNormalizer(True)
        max_bearing = [180]
        max_dist = [np.linalg.norm(self.world_size) + 10]  # add a ten meter buffer
        min_dist = [0.0]
        max_bool, min_bool = [1.0], [0.0]
        min_speed, max_speed = self._moos_config.speed_bounds
        max_speed += 2 # add some padding to the max speed
        min_speed, max_speed = [min_speed], [max_speed]
        agent_obs_normalizer.register("opponent_home_bearing", max_bearing)
        agent_obs_normalizer.register("opponent_home_distance", max_dist, min_dist)
        agent_obs_normalizer.register("own_home_bearing", max_bearing)
        agent_obs_normalizer.register("own_home_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_0_bearing", max_bearing)
        agent_obs_normalizer.register("wall_0_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_1_bearing", max_bearing)
        agent_obs_normalizer.register("wall_1_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_2_bearing", max_bearing)
        agent_obs_normalizer.register("wall_2_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_3_bearing", max_bearing)
        agent_obs_normalizer.register("wall_3_distance", max_dist, min_dist)
        agent_obs_normalizer.register("speed", max_speed, min_speed)
        agent_obs_normalizer.register("has_flag", max_bool, min_bool)
        agent_obs_normalizer.register("on_side", max_bool, min_bool)
        agent_obs_normalizer.register(
            "tagging_cooldown", [self._moos_config.tagging_cooldown], [0.0]
        )
        agent_obs_normalizer.register("is_tagged", max_bool, min_bool)

        assert len(self._red_team) == len(self._blue_team), "Expecting symmetric team sizes so that policies are interchangeable"
        num_on_team = len(self._blue_team)

        for i in range(num_on_team - 1):
            teammate_name = f"teammate_{i}"
            agent_obs_normalizer.register((teammate_name, "bearing"), max_bearing)
            agent_obs_normalizer.register(
                (teammate_name, "distance"), max_dist, min_dist
            )
            agent_obs_normalizer.register(
                (teammate_name, "relative_heading"), max_bearing
            )
            agent_obs_normalizer.register(
                (teammate_name, "speed"), max_speed, min_speed
            )
            agent_obs_normalizer.register(
                (teammate_name, "has_flag"), max_bool, min_bool
            )
            agent_obs_normalizer.register(
                (teammate_name, "on_side"), max_bool, min_bool
            )
            agent_obs_normalizer.register(
                (teammate_name, "tagging_cooldown"), [self.tagging_cooldown], [0.0]
            )
            agent_obs_normalizer.register(
                (teammate_name, "is_tagged"), max_bool, min_bool
            )

        for i in range(num_on_team):
            opponent_name = f"opponent_{i}"
            agent_obs_normalizer.register((opponent_name, "bearing"), max_bearing)
            agent_obs_normalizer.register(
                (opponent_name, "distance"), max_dist, min_dist
            )
            agent_obs_normalizer.register(
                (opponent_name, "relative_heading"), max_bearing
            )
            agent_obs_normalizer.register(
                (opponent_name, "speed"), max_speed, min_speed
            )
            agent_obs_normalizer.register(
                (opponent_name, "has_flag"), max_bool, min_bool
            )
            agent_obs_normalizer.register(
                (opponent_name, "on_side"), max_bool, min_bool
            )
            agent_obs_normalizer.register(
                (opponent_name, "tagging_cooldown"), [self._moos_config.tagging_cooldown], [0.0]
            )
            agent_obs_normalizer.register(
                (opponent_name, "is_tagged"), max_bool, min_bool
            )

        return agent_obs_normalizer

    
