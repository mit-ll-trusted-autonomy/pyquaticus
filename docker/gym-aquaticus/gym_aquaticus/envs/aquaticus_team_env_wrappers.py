from collections import OrderedDict
import gym
import math
import numpy as np
import random

from gym_aquaticus.envs.aquaticus_team_env import AquaticusTeamEnv

from gym_aquaticus.utils import ArtificialPotentialField, mag_heading_to_vec, vec_to_mag_heading

class AquaticusTeamDefender(AquaticusTeamEnv):
    '''
    This class wraps the AquaticusTeamEnv Gym environment.
    It allows taking multiple steps in the underlying environment
    and creates a dedicated action space that consists of
    handwritten heuristic actions.
    '''
    def __init__(self, config_dict):
        self._defender_frame = config_dict['frame']
        config_dict['frame'] = 'world'
        config_dict['return_raw_state'] = True
        super().__init__(config_dict)
        # this should be the number of steps to execute a subpolicy
        # this is only interrupted when the boolean state projection changes
        self._meta_steps = config_dict['meta_steps']
        self._team = config_dict['team']
        self._other_team = 'red' if self._team == 'blue' else 'blue'
        self._agent_to_train = config_dict['agent_to_train']

        assert self._agent_to_train is not None, 'Expecting a specific agent'
        assert self._single_agent is None, 'Need to be able to take actions for all agents in each step'

        # this entry should be a dictionary mapping integers to a policy to execute
        self._policies = config_dict['policies']
        self._opponent_policies = config_dict['opponent_policies']
        self.observation_space = self._frame_observation_space[self._defender_frame][self._agent_to_train]
        self.action_space = gym.spaces.Discrete(len(self._policies))
        # instantiate the policies
        for k in self._policies:
           self._policies[k] = self._policies[k](self, self._agent_to_train, self._team)

        scrimmage_right = np.asarray(self._moos_config.blue_zone_ur, dtype=np.float32)
        scrimmage_left = np.asarray(self._moos_config.blue_zone_ul, dtype=np.float32)
        self._fields = [
            ArtificialPotentialField(self.steptime, 8, self.boundary_lr, self.boundary_ur),
            ArtificialPotentialField(self.steptime, 8, self.boundary_ur, self.boundary_ul),
            ArtificialPotentialField(self.steptime, 8, self.boundary_ul, self.boundary_ll),
            ArtificialPotentialField(self.steptime, 8, self.boundary_ll, self.boundary_lr),
            ArtificialPotentialField(self.steptime, 8, scrimmage_right, scrimmage_left)
            ]

        self._defend_total_reward = 0
        self._sorted_agents = list(sorted([n for n, _ in self.get_agents_gen()]))
        self._bool_statevec = None

    def reset(self):
        self.pause()
        if self.episode_number:
            print(f'Defend Total Reward: {self._defend_total_reward}')
            self._defend_total_reward = 0
        state = super().reset()
        self._bool_statevec = self.get_bool_statevec(self._state)
        self._last_tagged = {a: False for a in self._sorted_agents}
        self._opponent_choose_action = dict()
        for n in self._opponent_policies:
            self._opponent_choose_action[n] = self._opponent_policies[n](self.episode_number)
        self.unpause()
        return self.state_to_obs(state, frame=self._defender_frame)[self._agent_to_train]

    def step(self, meta_action):
        policy = self._policies[meta_action]
        total_reward = 0
        action_dict = {}
        for i in range(self._meta_steps):
            # the previous state is always saved in self._state after the parent class step
            action = policy.choose_action(self._state)
            action_vec = mag_heading_to_vec(action[0], action[1])
            pnt, _, _ = self._state['agents'][self._agent_to_train]
            edit = np.zeros((len(action_vec),))
            for field in self._fields:
                edit += field(pnt, action_vec)
            action_vec += edit
            spd, hdg = vec_to_mag_heading(action_vec)
            action = [spd, hdg]
            action_dict[self._agent_to_train] = action

            # get actions for other agents
            for name, _ in self.get_agents_gen(self._other_team):
                if name in self._opponent_policies:
                    action_dict[name] = self._opponent_choose_action[name](self._state)
            state, _, done, info = super().step(action_dict)

            reward = self.compute_reward(self._agent_to_train, info['last_state'], action, self._state, self._last_tagged, info['tagged'])
            # penalize for hitting a potential field
            edit_norm = np.linalg.norm(edit)
            if info['tagged'][self._agent_to_train]:
                # don't want to get tagged
                reward += -2
            elif edit_norm >= 1e-5:
                reward += -min(edit_norm, 0.05)
            total_reward += reward
            if done['__all__']:
                break

            new_bool_statevec = self.get_bool_statevec(self._state)
            boolstate_changed = new_bool_statevec != self._bool_statevec
            self._bool_statevec = new_bool_statevec

            self._last_tagged = info['tagged']

            if boolstate_changed:
                break

        obs = self.state_to_obs(state, frame=self._defender_frame)[self._agent_to_train]
        self._defend_total_reward += total_reward

        return obs, total_reward, done['__all__'], info


    def get_bool_statevec(self, state):
        vec = []
        for name in self._sorted_agents:
            pnt, has_flag, on_side = state['agents'][name]
            vec.extend([pnt.tagged, has_flag, on_side])
        return vec


    def compute_reward(self, agent_name, last_state, action, state, last_tagged, tagged):
        close_opps = self.get_agents_in_range(agent_name, last_state, \
                                              self.get_agents_gen(self._other_team), \
                                              radius=1.1*self._moos_config.tag_radius)
        _, defend_flag = self.get_team_goal_defend(self._team)
        pnt, _, _ = state['agents'][self._agent_to_train]
        ego_pos = np.array([pnt.x, pnt.y], dtype=np.float32)
        ego_dist_to_flag = np.linalg.norm(defend_flag - ego_pos)
        for a in close_opps:
            _, lhas_flag, lon_side = last_state['agents'][a]
            a_pnt, has_flag, on_side = state['agents'][a]
            a_pos = np.array([a_pnt.x, a_pnt.y], dtype=np.float32)
            if not last_tagged[a] and tagged[a]:
                # positive reward if tagged
                rew = 1.
                # get extra reward if facing each other
                # (want to discourage tail chasing)
                unit_ego_vec = mag_heading_to_vec(pnt.speed, pnt.heading, unit=True)
                unit_a_vec = mag_heading_to_vec(a_pnt.speed, a_pnt.heading, unit=True)
                rew += max(0., -np.dot(unit_ego_vec, unit_a_vec))
                return rew

        # if not getting anything for close opponents, consider other cases
        for a, _ in self.get_agents_gen(self._other_team):
            _, lhas_flag, lon_side = last_state['agents'][a]
            a_pnt, has_flag, on_side = state['agents'][a]
            a_pos = np.array([a_pnt.x, a_pnt.y], dtype=np.float32)
            a_dist_to_flag = np.linalg.norm(defend_flag - a_pos)
            if not tagged[a] and ego_dist_to_flag > a_dist_to_flag:
                # bad to let agent slip past you and get closer to the flag
                return -0.1
            elif not lon_side and has_flag and on_side:
                # large negative reward if they get flag to their side
                return -5.
            elif not lhas_flag and has_flag:
                # negative reward for letting other agent get the flag
                return -2

        return 0.


    def get_agents_in_range(self, agent_name, state, other_agents, radius):
        close_agents = set()
        # this uses a particular state (might be last_state not current)
        ego_pnt, _, _ = state['agents'][agent_name]
        for n, _ in other_agents:
            if n == agent_name:
                continue
            agent_pnt, has_flag, on_side = state['agents'][n]
            dist = math.hypot(agent_pnt.x - ego_pnt.x,
                              agent_pnt.y - ego_pnt.y)
            if dist < radius:
                close_agents.add(n)
        return close_agents


class AquaticusTeamAttacker(AquaticusTeamEnv):
    def __init__(self, config_dict):
        self._attacker_frame = config_dict['frame']
        config_dict['frame'] = 'world'
        config_dict['return_raw_state'] = True
        super().__init__(config_dict)
        self._team = config_dict['team']
        self._other_team = 'red' if self._team == 'blue' else 'blue'
        self._agent_to_train = config_dict['agent_to_train']

        assert self._agent_to_train is not None, 'Expecting a specific agent'
        assert self._single_agent is None, 'Need to be able to take actions for all agents in each step'

        # opponent policies maps an agent name to a function that takes a state and produces an action
        self._opponent_policies = config_dict['opponent_policies']
        self.observation_space = self._frame_observation_space[self._attacker_frame][self._agent_to_train]
        self.action_space = gym.spaces.Discrete(9)

        self._fields = [
            ArtificialPotentialField(self.steptime, 8, self.boundary_lr, self.boundary_ur),
            ArtificialPotentialField(self.steptime, 8, self.boundary_ur, self.boundary_ul),
            ArtificialPotentialField(self.steptime, 8, self.boundary_ul, self.boundary_ll),
            ArtificialPotentialField(self.steptime, 8, self.boundary_ll, self.boundary_lr)
            ]

        self._attack_total_reward = 0.
        self._sorted_agents = list(sorted([n for n, _ in self.get_agents_gen()]))

    def reset(self):
        self.pause()
        if self.episode_number:
            print(f'Attack Total Reward: {self._attack_total_reward}')
            self._attack_total_reward = 0
        state = super().reset()
        self._last_tagged = {a: False for a in self._sorted_agents}

        # get actions for other agents
        # self._opponent_choose_action = dict()
        for n in self._opponent_policies:
            self._agents[n].take_action(self._opponent_policies[n](self.episode_number))
            # self._opponent_choose_action[n] = self._opponent_policies[n](self.episode_number)

        self.unpause()
        return self.state_to_obs(state, frame=self._attacker_frame)[self._agent_to_train]

    def step(self, meta_action):
        action = self.process_action(self._agent_to_train, self._state, meta_action)
        action_vec = mag_heading_to_vec(action[0], action[1])
        edit = np.zeros((len(action_vec),))
        pnt = self._state['agents'][self._agent_to_train][0]
        for field in self._fields:
            edit += field(pnt, action_vec)
        action_vec += edit

        action_dict = OrderedDict()
        # avoid opponent
        for name, _ in self.get_agents_gen(self._other_team):
            opp_pnt = self._state['agents'][name][0]
            opp_pos = np.array([opp_pnt.x, opp_pnt.y], dtype=np.float32)
            field = ArtificialPotentialField(self.steptime, self._moos_config.tag_radius*1.6, opp_pos)
            edit += field(pnt, action_vec)

        action_vec += edit
        spd, hdg = vec_to_mag_heading(action_vec)
        action = [min(spd, self._moos_config.speed_bounds[1]), hdg]
        action_dict[self._agent_to_train] = action

        # get actions for other agents
        # for name in self._opponent_policies:
        #     action_dict[name] = self._opponent_choose_action[name](self._state)

        state, _, done, info = super().step(action_dict)
        reward = self.compute_reward(self._agent_to_train, info['last_state'], action, self._state, info)
        # penalize for hitting a potential field
        edit_norm = np.linalg.norm(edit)
        if edit_norm >= 1e-5:
            reward += -min(edit_norm, 0.5)
        self._attack_total_reward += reward
        self._last_tagged = info['tagged']
        obs = self.state_to_obs(state, frame=self._attacker_frame)[self._agent_to_train]

        return obs, reward, done['__all__'], info


    def process_action(self, agent_name, state, raw_action):
        '''
        0-7 are relative heading differences of 45 degrees
        8   means to stop in place
        '''
        if isinstance(raw_action, str):
            return raw_action
        assert isinstance(raw_action, int) or isinstance(raw_action, np.integer)
        pnt = state['agents'][agent_name][0]
        spd = self._moos_config.speed_bounds[1]
        if raw_action == 8:
            return [0, pnt.heading]
        else:
            return [spd, (pnt.heading + 45*raw_action)%360]

    def compute_reward(self, agent_name, last_state, action, state, info):
        lpnt, lhas_flag, lon_side = last_state['agents'][agent_name]
        pnt, has_flag, on_side = state['agents'][agent_name]

        tagged = info['tagged']
        if tagged[self._agent_to_train]:
            # penalty for getting tagged
            # is proportional to how many steps it took to return to your flag location
            return -info['tag_steps']
        elif not lhas_flag and has_flag:
            # grab flag
            return 5.
        elif lhas_flag and not has_flag:
            assert not tagged[self._agent_to_train]
            # score
            return 10.
        else:
            # no directional reward: trying to avoid having it go straight into tagger
            # negative reward to encourage getting the flag as soon as possible
            reward = -0.01
            if pnt.speed < 0.6*self._moos_config.speed_bounds[1]:
                reward -= 0.1
            return reward
            # reward = -1.05
            # # toward flag reward
            # goal = None
            # if not lhas_flag:
            #     # getting flag
            #     goal = self._agents[agent_name]._goal_flag
            # else:
            #     # returning flag
            #     goal = self._agents[agent_name]._defend_flag
            # goal_direc = np.array([goal[0] - lpnt.x, goal[1] - lpnt.y], dtype=np.float32)
            # goal_norm = np.linalg.norm(goal_direc)
            # if goal_norm > 1e-6:
            #     goal_direc = goal_direc / goal_norm
            # mvmt_x = pnt.x - lpnt.x
            # mvmt_y = pnt.y - lpnt.y
            # mvmt_vec = np.array([mvmt_x, mvmt_y], dtype=np.float32)
            # max_mvmt = self._moos_config.speed_bounds[1]*self.steptime
            # mvmt_vec = mvmt_vec / max_mvmt

            # # directional reward
            # reward += min(np.dot(mvmt_vec, goal_direc), 1.)
            # return reward


        return 0.