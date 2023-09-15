from collections import OrderedDict
import copy
from logging.config import DEFAULT_LOGGING_CONFIG_PORT
import gym
import itertools
import math
import numpy as np
import pymoos
import subprocess
import sys
import time


from gym_aquaticus.envs.aquaticus_agent import AquaticusAgent
from gym_aquaticus.utils import mag_heading_to, closest_point_on_line, \
     mag_heading_to_vec, vec_to_mag_heading, zero_centered_heading

DEFAULT_CONFIG = {
    'sim_script': None,
    'max_score': 1,
    'frame': 'world',
    'max_steps': 400,
    'single_agent': None, # if None then allows the whole team, otherwise flattens env for this agent
    'return_raw_state': False, # if set to true, reset/step return raw state dict
    'normalize_obs': True,
}

FRAME_OPTIONS = {'world', 'local'}

class AquaticusTeamEnv(gym.Env):
    def __init__(self, env_config):
        '''
        Args:
            env_config:            a dictionary containing the following
                moos_config:       a gym_aquaticus.config object
                red_team_params:   list of tuples (name, port) for red team agents
                blue_team_params:  list of tuples (name, port) for blue team agents
                sim_script:        bash script to launch MOOS
                max_score:         maximum score before game ends
                frame:             whether observations and actions are in the world or local frame
                                    world frame assumes central control of the whole team
                                    local frame assumes individual policies for each agent
        '''
        self.set_config(env_config)

        self._red_team = []
        self._blue_team = []
        self._agents = dict()
        # initialize these agents
        self._initialize_agents()

        if self._single_agent is not None:
            if self._single_agent in self._red_team:
                self._team = 'red'
            elif self._single_agent in self._blue_team:
                self._team = 'blue'
            else:
                raise RuntimeError('Agent to train is of unknown team')

        self._score = {
            'red':  0,
            'blue': 0
        }
        self._done = {n: False for _, _, n in itertools.chain(self._red_team_params, self._blue_team_params)}
        self._done['__all__'] = False

        # state elements for keeping track of flag location
        self._has_flag = {n: False for _, _, n in itertools.chain(self._red_team_params, self._blue_team_params)}

        # set up the expected informational attributes for a Gym Env
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()

        self.total_steps = 0
        self.episode_steps = 0
        self.episode_number = 0
        self.episode_reward = 0.0
        # Re-initialize these on reset()
        self.start_time = 0.0   # start of simulation (moostime)
        self.obs_time = 0.0     # time of last call to step()


    def reset(self):
        if self.total_steps == 0:
            self.hard_reset()

        if self._agent_to_train is not None:
            self.unpause() # make sure the sim is running
            new_state = self.get_raw_state()
            # don't want to start the episode with the agent tagged
            while new_state['agents'][self._agent_to_train][0].tagged:
                # tell agent to stop (for when control is returned)
                self._agents[self._agent_to_train].take_action([0., 0.])
                time.sleep(self.steptime/self.timewarp)
                new_state = self.get_raw_state()
        return self.soft_reset()


    def hard_reset(self):
        # stop any currently running MOOS processes
        self.close()


        # start the simulation script
        self._run_sim_script()
        self._setup_shoreside_comm()
        self._connect_agents()
        time.sleep(3)
        if not self._check_connection():
            # Try once more
            if not self._check_connection():
                raise RuntimeError('Failed to connect to MOOS.')

        return self.soft_reset()


    def soft_reset(self):
        # print stats for the previous episode
        if self.total_steps > 0:
            self.episode_number += 1
            print(f'\n=========================================================')
            print(f'episode: {self.episode_number}')
            print(f'episode steps: {self.episode_steps}')
            # print(f'episode reward: {self.episode_reward:.1f}')
            print(f'total steps: {self.total_steps}')
            print(f'=========================================================\n')
            self.episode_steps = 0
            self.episode_reward = 0.0
        self.start_time = 0.0
        self.obs_time = 0.0
        self.dead_time_n = 0
        self.dead_time_mean  = 0.0

        for team in self._score:
            self._score[team] = 0
        for n in self._done:
            self._done[n] = False

        self.unpause()

        state = self.get_raw_state()
        self._state = state
        if self._return_raw_state:
            return state
        else:
            return self.state_to_obs(state)


    def state_to_obs(self, state, frame=None):
        if frame is None:
            frame = self._frame
        if frame == 'local':
            obs = self.get_localframe_observation(state['agents'])
        else:
            obs = self.get_worldframe_observation(state['agents'])
        if self._normalize_obs:
            for key in obs:
                obs[key] = (obs[key] - self._obs_center[frame][key]) / self._obs_mag[frame][key]
        return self._extract_agent(obs)


    def get_worldframe_observation(self, agent_states):
        '''
        Observation space (per agent):
            x_goal_flag
            y_goal_flag
            x_defend_flag
            y_defend_flag
            Repeating for each agent of team (starting with yourself) then each agent of enemy team
                x_pos
                y_pos
                has_flag
        '''
        obs_args = []
        for team in ['red', 'blue']:
            other_team = 'red' if team == 'blue' else 'blue'
            for name, _ in self.get_agents_gen(team):
                obs = []
                goal_flag, defend_flag = self.get_team_goal_defend(team)
                obs.append(goal_flag[0])
                obs.append(goal_flag[1])
                obs.append(defend_flag[0])
                obs.append(defend_flag[1])
                # Own info
                pnt, has_flag, _ = agent_states[name]
                obs.append(pnt.x)
                obs.append(pnt.y)
                obs.append(1. if has_flag else 0.)
                # Teammate info
                for other_name, _ in self.get_agents_gen(team):
                    if other_name == name:
                        continue
                    pnt, has_flag, _ = agent_states[other_name]
                    obs.append(pnt.x)
                    obs.append(pnt.y)
                    obs.append(1. if has_flag else 0.)
                for other_name, _ in self.get_agents_gen(other_team):
                    pnt, has_flag, _ = agent_states[other_name]
                    obs.append(pnt.x)
                    obs.append(pnt.y)
                    obs.append(1. if has_flag else 0.)
                obs_args.append((name, obs))
        return OrderedDict(obs_args)


    def get_localframe_observation(self, agent_states):
        '''
        Returns a local observation space. These observations are
        based entirely on the agent local coordinate frame rather
        than the world frame.

        This was originally designed so that observations can be
        easily shared between different teams and agents.
        Without this the world frame observations from the blue and
        red teams are flipped (e.g., the goal is in the opposite
        direction)

        Observation Space (per agent):
            Goal location heading (degrees)  (opponent flag then home region after grabbing flag)
            Goal location distance
            Home flag heading (degrees)
            Home flag distance
            Wall 1 heading (degrees)
            Wall 1 distance
            Wall 2 heading (degrees)
            Wall 2 distance
            Wall 3 heading (degrees)
            Wall 3 distance
            Wall 4 heading (degrees)
            Wall 4 distance
            Own speed
            Own flag status
            On side (1. if on your own side, otherwise -1)
            For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
                Bearing from you (degrees)
                Heading of other agent relative to the vector to you (degrees)
                Distance
                Speed
                Tag status
                Has flag status

        Note 1 : the angles are 0 when the agent is pointed directly at the object
                 and increase in the clockwise direction
        Note 2 : the wall distances can be negative when the agent is out of bounds
        Note 3 : the boolean args Tag/Flag status are -1 false and +1 true

        Developer Note 1: changes here should be reflected in _define_localframe_observation_space
        '''
        obs_dict = OrderedDict()
        for team in ['red', 'blue']:
            opp_flag, defend_flag = self.get_team_goal_defend(team)
            other_team = 'blue' if team == 'red' else 'red'
            for name, _ in self.get_agents_gen(team):
                obs = []
                pnt, has_flag, on_side = agent_states[name]
                np_pos = np.array([pnt.x, pnt.y], dtype=np.float32)

                # Goal flag
                opp_flag_dist, opp_flag_heading = mag_heading_to(np_pos, opp_flag, pnt.heading)
                # Defend flag
                defend_flag_dist, defend_flag_heading = mag_heading_to(np_pos, defend_flag, pnt.heading)

                # Note: changing the goal rather than just relying on the
                #       existing defend flag info so that observations look
                #       sufficiently different after getting the flag
                #       need the defend flag region all the time so it knows
                #       what to guard if defending
                if has_flag:
                    goal_flag_heading = defend_flag_heading
                    goal_flag_dist = defend_flag_dist
                else:
                    goal_flag_heading = opp_flag_heading
                    goal_flag_dist = opp_flag_dist

                obs.append(goal_flag_heading)
                obs.append(goal_flag_dist)
                obs.append(defend_flag_heading)
                obs.append(defend_flag_dist)
                # Walls
                wall_1_closest_point = closest_point_on_line(self.boundary_ul, self.boundary_ur, np_pos)
                wall_1_dist, wall_1_heading = mag_heading_to(np_pos, wall_1_closest_point, pnt.heading)
                obs.append(wall_1_heading)
                obs.append(wall_1_dist)

                wall_2_closest_point = closest_point_on_line(self.boundary_ur, self.boundary_lr, np_pos)
                wall_2_dist, wall_2_heading = mag_heading_to(np_pos, wall_2_closest_point, pnt.heading)
                obs.append(wall_2_heading)
                obs.append(wall_2_dist)

                wall_3_closest_point = closest_point_on_line(self.boundary_lr, self.boundary_ll, np_pos)
                wall_3_dist, wall_3_heading = mag_heading_to(np_pos, wall_3_closest_point, pnt.heading)
                obs.append(wall_3_heading)
                obs.append(wall_3_dist)

                wall_4_closest_point = closest_point_on_line(self.boundary_ll, self.boundary_ul, np_pos)
                wall_4_dist, wall_4_heading = mag_heading_to(np_pos, wall_4_closest_point, pnt.heading)
                obs.append(wall_4_heading)
                obs.append(wall_4_dist)

                # Own speed
                obs.append(pnt.speed)
                # Own flag status
                obs.append(1. if has_flag else -1.)
                # On side
                obs.append(1. if on_side else -1.)

                # Relative observations to other agents
                # teammates first
                # TODO: consider sorting these by some metric
                #       in an attempt to get permutation invariance
                #       distance or maybe flag status (or some combination?)
                #       i.e. sorted by perceived relevance
                for other_agent_team in [team, other_team]:
                    for other_name, _ in self.get_agents_gen(other_agent_team):
                        if other_name == name:
                            continue
                        other_pnt, other_flag_status, other_on_side = agent_states[other_name]
                        other_np_pos = np.array([other_pnt.x, other_pnt.y], dtype=np.float32)
                        other_pnt_dist, other_pnt_heading = mag_heading_to(np_pos, other_np_pos, pnt.heading)
                        obs.append(other_pnt_heading)
                        _, hdg_to_pnt = mag_heading_to(other_np_pos, np_pos)
                        hdg_to_pnt = hdg_to_pnt%360
                        # bearing relative to the bearing to you
                        obs.append(zero_centered_heading((other_pnt.heading - hdg_to_pnt)%360))
                        obs.append(other_pnt_dist)
                        obs.append(other_pnt.speed)

                        obs.append(1. if other_pnt.tagged else -1.)
                        obs.append(1. if other_flag_status else -1.)

                obs_dict[name] = np.array(obs)
        return obs_dict


    def step(self, raw_action):
        '''
        Take an environment step with the given action

        If self._single_agent is set, then expect a single action applied to that agent
        Otherwise, takes a dictionary mapping agent names to actions

        Note: if the agent is tagged, step will wait until the agent is returned
              to its flag region (which may take a while)
        '''
        self.total_steps += 1
        self.episode_steps += 1
        # time for each step
        tau = self.steptime/self.timewarp

        if self._single_agent is None and \
           not isinstance(raw_action, dict) and not isinstance(raw_action, gym.spaces.Dict):
            raise RuntimeError('AquaticusTeamEnv: If passing single actions, must specify single_agent')

        # Take the action(s)
        acting_agents = set(raw_action) if self._single_agent is None else {self._single_agent}
        action = self._process_actions(self._state, raw_action)
        for name, agent in self._agents.items():
            if name in acting_agents:
                agent.take_action(action[name])
            # NOTE: this is not perfect, because it counts time separately from the underlying moos process
            #       ideally this would be reported from MOOS
            #       would have to modify the uFldTagManager from moos-aquaticus
            if agent.pnt.tagging_cooldown != agent.tagging_cooldown:
                # agent is still under a cooldown from tagging, advance their cooldown timer, clip at the configured tagging cooldown
                agent.pnt.tagging_cooldown = self._min(
                    (agent.pnt.tagging_cooldown + tau), agent.tagging_cooldown
                )

        # Let the simulation run for steptime seconds of moostime
        time.sleep(tau)
        new_state = self.get_raw_state()

        tagged = {n: pnt.tagged for n, (pnt, _, _) in new_state['agents'].items()}
        tag_steps = 0
        # TODO: unify single_agent and agent_to_train
        #       separated so wrappers could have a single agent but still access all agents
        #       but then don't get the tag semantics
        if self._agent_to_train is not None and tagged[self._agent_to_train]:
            # if agent to train is tagged, loop until MOOS control has returned agent to flag region
            while new_state['agents'][self._agent_to_train][0].tagged:
                # tell agent to stop (for when control is returned)
                self._agents[self._agent_to_train].take_action([0., 0.])
                time.sleep(self.steptime/self.timewarp)
                new_state = self.get_raw_state()
                tag_steps += 1

        if self._return_raw_state:
            obs = new_state
        else:
            obs = self.state_to_obs(new_state)
        done = self._check_done(self._state, action, new_state, tagged)
        reward = self._compute_reward(self._state, action, new_state, tagged)
        info = {'tagged': tagged, 'last_state': self._state, 'tag_steps': tag_steps}
        self._state = new_state
        self.episode_reward += reward if self._single_agent is not None else sum(reward.values())
        return obs, reward, done, info


    def render(self, mode='human'):
        pass


    def get_raw_state(self):
        '''
        Returns the raw state variables (PNT objects) and a boolean
        indicating flag posession for each agent.

        Note: the agents' state is updated asynchronously at 10Hz
        This function caches the state at a particular time for processing,
        ensuring that the relative positions of the agents are consistent
        with a snapshot

        Return:
            dict: a dictionary
                  agent name -> (PNT object, boolean for has_flag)
        '''
        state = dict()
        agent_states = {n:(copy.deepcopy(a.pnt), self._has_flag[n]) for n, a in self.get_agents_gen()}
        agent_states = dict()
        for team in ['red', 'blue']:
            for n, a in self.get_agents_gen(team):
                pnt = copy.deepcopy(a.pnt)
                has_flag = self._has_flag[n]
                on_side = self._check_on_side(pnt, team)
                agent_states[n] = (pnt, has_flag, on_side)
        state['agents'] = agent_states
        state['blue_flag'] = self._blue_flag
        state['red_flag'] = self._red_flag
        return state


    def close(self):
        if self.sim_script is None:
            return
        self._red_team.clear()
        self._blue_team.clear()
        for a in self._agents.values():
            a.close(False)
        self._agents.clear()
        if hasattr(self, 'shoreside_comm'):
            self.shoreside_comm.close(False)
        # This is the quickest way to end the simulation (don't use ktm)
        subprocess.call('killAllMOOS.sh')
        # We must wait a few seconds to allow procs to exit and sockets to
        # close before restarting the script
        time.sleep(3)


    def _process_actions(self, last_state, raw_act):
        '''
        Process a dictionary of actions.
        If raw_act is a single action meant for self._single_agent,
        then create a dictionary so it's in a standard format
        '''
        if self._single_agent is not None:
            return {self._single_agent:
                    self._process_single_action(last_state['agents'][self._single_agent][0],
                                               raw_act)}
        else:
            actions = OrderedDict()
            for name in raw_act:
                actions[name] = self._process_single_action(last_state['agents'][name][0],
                                                            raw_act[name])
            return actions


    def _process_single_action(self, last_pnt, raw_act):
        '''
        Does any processing of the action before passing to the agent.
        In particular, it takes delta action spaces for the local frame
        version, and produces the actual desired speed and heading

        Args:
            last_pnt:    the last PNT of the agent before taking this action
            raw_act:     the raw action

        Returns:
            (float, float): a (speed, heading) action in the world frame

        Note: agents always take actions in the world frame
                This function will translate agent-local frame action headings
                to world-frame headings as needed
        '''
        action = None
        if isinstance(raw_act, str):
            # this is a MOOS command -- no processing to do
            return raw_act
        elif self._frame == 'world':
            action = raw_act
        else:
            # interpret the heading as relative to the last reported state
            spd, delta_hdg = raw_act
            if spd > self._moos_config.speed_bounds[1]:
                spd = self._moos_config.speed_bounds[1]
            elif spd < self._moos_config.speed_bounds[0]:
                spd = self._moos_config.speed_bounds[0]
            hdg = (last_pnt.heading + delta_hdg) % 360
            action = [spd, hdg]

        return action


    #----------------------------------------------------------------------
    # _check_connection: verify that the agent MOOSCommClients are connected
    def _check_connection(self):
        disconnected_agents = self._get_disconnected_agents()
        if self.shoreside_comm.is_connected() and len(disconnected_agents) == 0:
            return True

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f'AquaticusTeamEnv ({",".join([a.name for a in disconnected_agents])}) lost connection - restarting')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        with open('aquaticus_env.log', 'a') as f:
            f.write(f'restarting {",".join([a.name for a in disconnected_agents])} after {self.episode_number} episodes\n')

        self.close()
        self._run_sim_script()
        self._connect_agents()
        self._setup_shoreside_comm()
        time.sleep(3)

        return self.shoreside_comm.is_connected() and len(self._get_disconnected_agents()) == 0


    def _get_disconnected_agents(self):
        disconnected_agents = []
        all_connected = True
        for _, agent in self.get_agents_gen():
            if not agent.is_connected():
                disconnected_agents.append(agent)
                all_connected = False

        if all_connected:
            assert not len(disconnected_agents)
            return []

        # give the clients a few more seconds to connect on their own
        time.sleep(3)
        disconnected_agents.clear()
        all_connected = True
        for _, agent in self.get_agents_gen():
            if not agent.is_connected():
                disconnected_agents.append(agent)
                all_connected = False
        return disconnected_agents


    def _connect_agents(self):
        if not self._agents:
            self._initialize_agents()
        for _, agent in self.get_agents_gen():
            agent.connect_to_moos()


    def _initialize_agents(self):
        for server, port, name in self._red_team_params:
            self._red_team.append(name)
            if name in self._agents:
                raise ValueError(f'Agents should have distinct names (even between teams) but got "{name}" twice.')
            self._agents[name] = AquaticusAgent(self._moos_config, str(server), int(port), str(name), 'red')

        for server, port, name in self._blue_team_params:
            self._blue_team.append(name)
            if name in self._agents:
                raise ValueError(f'Agents should have distinct names (even between teams) but got "{name}" twice.')
            self._agents[name] = AquaticusAgent(self._moos_config, str(server), int(port), str(name), 'blue')


    #----------------------------------------------------------------------
    # _run_sim_script: start the simulation script and handle the occasional
    #     exception. Give up after max_retries.
    #     If sim_script is None, this function is a NO-OP.
    def _run_sim_script(self):
        if self.sim_script is None:
            return
        max_retries = 10
        for retries in range(max_retries):
            try:
                subprocess.call(self.sim_script, timeout=15)
                return
            except (subprocess.TimeoutExpired, OSError) as e:
                self.close()
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(f'Error: {e}')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                with open('aquaticus_env.log', 'a') as f:
                    f.write(f'Error: {e} after {self.episode_number} episodes\n')
        # not much else we can do if script keeps failing
        print(f'{self.sim_script} failed to run after {max_retries} attempts.')
        sys.exit(1)

    def _on_mail(self):
        for msg in self.shoreside_comm.fetch():
            if 'HAS_FLAG_' in msg.key():
                name = msg.key()[len('HAS_FLAG_'):].lower()
                val = msg.string().lower()
                assert val in {'true', 'false'}
                hflag = (val == 'true')
                agent = self._agents[name]
                agent.notify_flag_status(hflag)
                time.sleep(0.02) # make sure agent tagged status has been updated
                if self.has_flag(name) and not hflag and not agent.pnt.tagged:
                    print('Score', agent._team)
                    self._score[agent._team] += 1
                self._has_flag[name] = hflag
            elif "TAG_RESULT" in msg.key():
                tagger = None
                rejected = False
                for item in msg.string().split(','):
                    n, val = item.split('=')
                    if n == 'src':
                        tagger = val
                    elif n == 'rejected':
                        rejected = True
                if tagger is None:
                    raise Warning(f'Unknown tag message format {msg.string()}')
                if not rejected:
                    agent = self._agents[tagger]
                    # reset the tagging cooldown
                    # can tag again when this is maxed out
                    # you can see the interval in TagManager.cpp (of uFldTagManager)
                    # it was hardcoded to 10.0 at the time of this comment
                    # and is replicated in our config file
                    agent.pnt.tagging_cooldown = 0.0
        return True


    def has_flag(self, name):
        return self._has_flag.get(name, False)


    # TODO: make reward amounts configurable
    def _compute_reward(self, last_state, action, state, tagged):
        '''
        Args:
            last_state: the previous state
            action:     the executed action
            state:      the resulting state
            tagged:     whether each agent was tagged
                        Note: last_state will be the state just before it was tagged
                              and state will be after the agent to train
                              is returned to its flag region
        '''
        reward_dict = OrderedDict()
        for n in self._agents:
            reward = -2
            lpnt, lhas_flag, _ = last_state['agents'][n]
            pnt, has_flag, _ = state['agents'][n]
            spd, hdg = pnt.speed, pnt.heading

            # toward flag reward
            goal = None
            if not lhas_flag:
                # getting flag
                goal = self._agents[n]._goal_flag
            else:
                # returning flag
                goal = self._agents[n]._defend_flag
            # TODO: revisit this
            #       it's an easy solution for now to avoid giving
            #       good reward when not actually moving (fast enough)
            #       but might encourage undesirable behavior
            #       (slow speed when avoiding)
            goal_direc = np.array([goal[0] - lpnt.x, goal[1] - lpnt.y], dtype=np.float32)
            goal_norm = np.linalg.norm(goal_direc)
            if goal_norm > 1e-6:
                goal_direc = goal_direc / goal_norm
            mvmt_x = pnt.x - lpnt.x
            mvmt_y = pnt.y - lpnt.y
            mvmt_vec = np.array([mvmt_x, mvmt_y], dtype=np.float32)
            max_mvmt = self._moos_config.speed_bounds[1]*self.steptime
            mvmt_vec = mvmt_vec / max_mvmt

            # directional reward
            reward += np.dot(mvmt_vec, goal_direc)

            # negatively reward for going slow
            reward += spd - self._moos_config.speed_bounds[1]

            # reward for getting flag
            if not lhas_flag and has_flag:
                reward += 10

            # reward for scoring
            if lhas_flag and not has_flag and \
                not tagged[n]:
                reward += 10

            # negative reward for getting tagging
            if tagged[n]:
                reward += -10

            reward_dict[n] = reward

        return self._extract_agent(reward_dict)


    def _check_done(self, last_state, action, state, tagged):
        '''
        Sets self._done and returns either
          a) a Boolean if self._single_agent is set
          b) the whole dictionary if it is not set
        '''
        all_done = any([s >= self._max_score for s in self._score.values()])
        all_done |= self.episode_steps > self._max_steps
        # uncomment if you want to end the episode when tagged
        # for n in tagged:
        #     self._done[n] |= tagged[n]
        if all_done:
            for n in self._done:
                self._done[n] = True
        return self._extract_agent(self._done)


    def _define_observation_space(self):
        to_def_fun = {
            'local': self._define_localframe_observation_space,
            'world': self._define_worldframe_observation_space
        }

        self._orig_frame_observation_space = dict()
        for frame in FRAME_OPTIONS:
            self._orig_frame_observation_space[frame] = to_def_fun[frame]()
        self._frame_observation_space = copy.deepcopy(self._orig_frame_observation_space)

        # cache information used to normalize observations (if normalization requested)
        # it is stored for all possible frames so that we can generate normalized
        # observations for each (regardless of what the env is set to)
        if self._normalize_obs:
            # maps from frame type to a dictionary of observation keys to range and centers of Box observations
            self._obs_mag = {}
            self._obs_center = {}
            for frame in FRAME_OPTIONS:
                obs = self._orig_frame_observation_space[frame]
                self._obs_mag[frame] = dict()
                self._obs_center[frame] = dict()
                normalized_obs_space = OrderedDict()
                for key in obs:
                    box = obs[key]
                    self._obs_mag[frame][key] = (box.high - box.low) / 2
                    self._obs_center[frame][key] = (box.low + box.high) / 2
                    normalized_box = gym.spaces.Box(low=-1, high=1, shape=box.high.shape)
                    normalized_obs_space[key] = normalized_box
                self._frame_observation_space[frame] = normalized_obs_space
            obs = gym.spaces.Dict(self._frame_observation_space[self._frame])
        else:
            obs = self._orig_frame_observation_space[frame]

        return self._extract_agent(obs)


    def _define_worldframe_observation_space(self):
        odict = OrderedDict()
        for team in ['red', 'blue']:
            other_team = 'blue' if team == 'red' else 'red'
            # Get flag bounds with some cushion
            flag_low_x = min([self._moos_config.blue_flag[0], self._moos_config.red_flag[0]]) - 1
            flag_low_y = min([self._moos_config.blue_flag[1], self._moos_config.red_flag[1]]) - 1
            flag_high_x = max([self._moos_config.blue_flag[0], self._moos_config.red_flag[0]]) + 1
            flag_high_y = max([self._moos_config.blue_flag[1], self._moos_config.red_flag[1]]) + 1
            obs_low = np.array([flag_low_x, flag_low_y, flag_low_x, flag_low_y])
            obs_high = np.array([flag_high_x, flag_high_y, flag_high_x, flag_high_y])
            for _, agent in self.get_agents_gen(team):
                obs_low = np.append(obs_low, agent.get_obs_low_limit())
                obs_high = np.append(obs_high, agent.get_obs_high_limit())
            for _, agent in self.get_agents_gen(other_team):
                obs_low = np.append(obs_low, agent.get_obs_low_limit())
                obs_high = np.append(obs_high, agent.get_obs_high_limit())
            box = gym.spaces.Box(low=obs_low, high=obs_high)
            for name, _ in self.get_agents_gen(team):
                odict[name] = box
        return gym.spaces.Dict(odict)


    def _define_localframe_observation_space(self):
        '''
        See comment of get_localframe_observation for the observation space definition

        Observation Space (per agent):
            Goal flag angle (degrees)
            Goal flag distance
            Defend flag angle (degrees)
            Defend flag distance
            Upper wall angle (degrees)
            Upper wall distance
            Right wall angle (degrees)
            Right wall distance
            Lower wall angle (degrees)
            Lower wall distance
            Left wall angle (degrees)
            Left wall distance
            Own speed
            Own flag status
            On side (1. if on your own side, otherwise 0)
            For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
                Angle (degrees)
                Angle (degrees)
                Distance
                Speed
                Tag status
                Has flag status
        '''
        angle_bounds = [-180., 180.]
        distance_low = 0
        padding = 80 # padding to ensure we don't exceed the distance
        distance_high = math.hypot(abs(self.boundary_ll[0] - self.boundary_ur[0]) + padding,
                                   abs(self.boundary_lr[1] - self.boundary_ul[1]) + padding)
        distance_bounds = [distance_low, distance_high]
        speed_bounds = list(self._moos_config.speed_bounds)
        # give some buffer in case it gets going faster
        speed_bounds[1] = speed_bounds[1]*1.7
        bool_bounds  = [-1., 1.]

        low, high = [], []
        for i, l in enumerate([low, high]):
            # flag and wall angle / distance
            l.extend([angle_bounds[i], distance_bounds[i]]*6)
            # speed
            l.append(speed_bounds[i])
            # flag status and on side
            l.append(bool_bounds[i])
            l.append(bool_bounds[i])
            # other agents
            other_agent_bounds = [angle_bounds[i],
                                  angle_bounds[i],
                                  distance_bounds[i],
                                  speed_bounds[i],
                                  bool_bounds[i],
                                  bool_bounds[i]]
            l.extend(other_agent_bounds*(len(self._agents)-1))
        agent_obs_space = gym.spaces.Box(low=np.array(low, dtype=np.float32),
                                         high=np.array(high, dtype=np.float32))
        obs_space_dict = OrderedDict()
        for name, _ in self.get_agents_gen():
            obs_space_dict[name] = agent_obs_space
        return gym.spaces.Dict(obs_space_dict)


    def _define_action_space(self):
        '''
        Defines the action space based on the agents.
        Note: this environment will accept actions for every agent (either team)
        but does not require it.
        '''
        to_def_fun = {
            'local': self._define_localframe_action_space,
            'world': self._define_worldframe_action_space
        }

        self._frame_action_space = dict()
        for frame in FRAME_OPTIONS:
            self._frame_action_space[frame] = to_def_fun[frame]()
        return self._extract_agent(self._frame_action_space[self._frame])


    def _define_worldframe_action_space(self):
        odict = OrderedDict()
        for name, agent in self.get_agents_gen():
            odict[name] = gym.spaces.Box(low=agent.get_action_low_limit(),
                                         high=agent.get_action_high_limit())
        return gym.spaces.Dict(odict)


    def _define_localframe_action_space(self):
        # heading action space is -180, 180 so that a network outputting 0 is
        # mapped to a heading of 0 in the agent's frame
        odict = OrderedDict()
        speed_bounds = self._moos_config.speed_bounds
        for n in sorted(self._agents):
            odict[n] = gym.spaces.Box(low=np.array([speed_bounds[0], -180], dtype=np.float32),
                                      high=np.array([speed_bounds[1], 180], dtype=np.float32))
        return gym.spaces.Dict(odict)


    def get_agents_gen(self, team=None):
        '''
        Returns a generator over all agents in a given team

        Args:
            team: 'red', 'blue', or None

        Return:
            generator: the corresponding agents or red agents followed
                       by blue agents if team is None
        '''
        if team is None:
            for name in itertools.chain(self._red_team, self._blue_team):
                yield (name, self._agents[name])
        elif team not in {'red', 'blue'}:
            raise ValueError(f'Expecting "red" or "blue" for team but got {team}')
        else:
            agent_names = self._red_team if team == 'red' else self._blue_team
            for name in agent_names:
                yield (name, self._agents[name])


    def get_team_goal_defend(self, team):
        goal_flag = self._blue_flag if team == 'red' else self._red_flag
        defend_flag = self._red_flag if team == 'red' else self._blue_flag
        return goal_flag, defend_flag


    def get_priority_agent(self, agent_name, agents, prioritize_flag=False):
        '''
        Gets the priority agent agent_name out of the agents in agents
        Typically priority means the closest agent, but the flag
        prioritize_flag can be used to instead prioritize any agent
        holding the flag.

        Args:
            agent_name: the name of the agent this is relative to
            agents: a list or generator of name, agent tuples
            prioritize_flag: if true, then will return the agent holding the flag regardless of distance

        Returns:
            str: the name of the priority agent
        '''
        ego_pnt, _, _ = self._state['agents'][agent_name]
        # get the priority target
        priority_agent = None
        min_dist = 100000
        for n, _ in agents:
            if n == agent_name:
                continue
            agent_pnt, has_flag, on_side = self._state['agents'][n]
            if prioritize_flag and has_flag:
                priority_agent = n
                break
            elif priority_agent is None:
                priority_agent = n
                continue
            dist = math.hypot(agent_pnt.x - ego_pnt.x,
                              agent_pnt.y - ego_pnt.y)
            if dist < min_dist:
                min_dist = dist
                priority_agent = n
        assert priority_agent is not None
        return priority_agent


    def pause(self):
        self.shoreside_comm.notify('DEPLOY_ALL', 'false', -1)


    def unpause(self):
        self.shoreside_comm.notify('DEPLOY_ALL', 'true', -1)


    def _setup_shoreside_comm(self):
        if hasattr(self, 'shoreside_comm'):
            del self.shoreside_comm

        # Communication object for shoreside
        self.shoreside_comm = pymoos.comms()
        def _on_connect():
            for _, _, name in itertools.chain(self._red_team_params, self._blue_team_params):
                self.shoreside_comm.register(f'HAS_FLAG_{name.upper()}', 0)
                self.shoreside_comm.register(f'TAG_RESULT_{name.upper()}', 0)
            self.shoreside_comm.register('DEPLOY_ALL', 0)
        self.shoreside_comm.set_on_connect_callback(_on_connect)
        self.shoreside_comm.set_on_mail_callback(self._on_mail)
        server, port, name = self._shoreside_params
        self.shoreside_comm.run(str(server), int(port), str(name))
        self.shoreside_comm.set_quiet(True)


    def _extract_agent(self, d):
        '''
        If a specific agent is being trained, index by that agent

        Args:
            d: a dictionary representing observations or actions
        '''
        if self._single_agent is None or \
           (not isinstance(d, dict) and not isinstance(d, gym.spaces.Dict)):
            return d
        else:
            return d[self._single_agent]


    def set_config(self, env_config):
        if hasattr(self, '_config_set') and self._config_set:
            return

        for k, v in DEFAULT_CONFIG.items():
            if k not in env_config:
                env_config[k] = v

        self._moos_config = env_config['moos_config']
        self._shoreside_params = env_config['shoreside_params']
        self._red_team_params = env_config['red_team_params']
        self._blue_team_params = env_config['blue_team_params']
        self.sim_script = env_config['sim_script']
        self._max_score = env_config['max_score']
        self._max_steps = env_config['max_steps']
        self._frame = env_config['frame']
        self._return_raw_state = env_config['return_raw_state']
        self._normalize_obs = env_config['normalize_obs']
        if self._frame not in FRAME_OPTIONS:
            raise ValueError(f'Expecting frame to be in {repr(FRAME_OPTIONS)} but got: {self._frame}')
        self._single_agent = env_config['single_agent'] # the agent to train
        if 'agent_to_train' in env_config:
            self._agent_to_train = env_config['agent_to_train']
        else:
            self._agent_to_train = None

        self._blue_flag = np.asarray(self._moos_config.blue_flag, dtype=np.float32)
        self._red_flag = np.asarray(self._moos_config.red_flag, dtype=np.float32)

        self.scrimmage_pnts = np.asarray(self._moos_config.scrimmage_pnts, dtype=np.float32)
        # define function for checking which side an agent is on
        if abs(self.scrimmage_pnts[0][0] - self.scrimmage_pnts[1][0]) < 1e-2:
            if self._red_flag[0] > self.scrimmage_pnts[0][0]:
                def check_side(pnt, team):
                    if team == 'red':
                        return pnt.x > self.scrimmage_pnts[0][0]
                    else:
                        return pnt.x < self.scrimmage_pnts[0][0]
            else:
                def check_side(pnt, team):
                    if team == 'red':
                        return pnt.x < self.scrimmage_pnts[0][0]
                    else:
                        return pnt.x > self.scrimmage_pnts[0][0]

        elif abs(self.scrimmage_pnts[0][1] - self.scrimmage_pnts[1][1]) < 1e-2:
            raise RuntimeError('Horizontal scrimmage lines not yet supported')
        else:
            m = (self.scrimmage_pnts[1][1] - self.scrimmage_pnts[0][1]) / (self.scrimmage_pnts[1][0] - self.scrimmage_pnts[0][1])
            b = self.scrimmage_pnts[0][1] - m*self.scrimmage_pnts[0][0]

            if self._red_flag[1] > m*self._red_flag[0] + b:
                def check_side(pnt, team):
                    if team == 'red':
                        return pnt.y > m*pnt.x + b
                    else:
                        return pnt.y < m*pnt.x + b
            else:
                def check_side(pnt, team):
                    if team == 'red':
                        return pnt.y < m*pnt.x + b
                    else:
                        return pnt.y > m*pnt.x + b

        self._check_on_side = check_side

        # The operating boundary is defined in shoreside/meta_shoreside.moos
        self.boundary_ul = np.asarray(self._moos_config.boundary_ul, dtype=np.float32)
        self.boundary_ur = np.asarray(self._moos_config.boundary_ur, dtype=np.float32)
        self.boundary_ll = np.asarray(self._moos_config.boundary_ll, dtype=np.float32)
        self.boundary_lr = np.asarray(self._moos_config.boundary_lr, dtype=np.float32)
        self.world_size  = np.array([np.linalg.norm(self.boundary_lr - self.boundary_ll),
                                     np.linalg.norm(self.boundary_ul - self.boundary_ll)])
        if 'timewarp' in env_config and env_config['timewarp'] is not None:
            self._moos_config.moos_timewarp = env_config['timewarp']
            self._moos_config.sim_timestep = self._moos_config.moos_timewarp / 10.0
        self.steptime = self._moos_config.sim_timestep
        self.time_limit = self._moos_config.sim_time_limit
        self.timewarp = self._moos_config.moos_timewarp

        # mark this function called already
        # if called again, nothing will happen
        self._config_set = True

    def _min(self, a, b) -> bool:
        """Convenience method for determining a minimum value. The standard `min()` takes much longer to run."""
        if a < b:
            return a
        else:
            return b

