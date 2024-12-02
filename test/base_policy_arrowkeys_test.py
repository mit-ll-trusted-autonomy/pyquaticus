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

import argparse
import gymnasium as gym
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE
import sys
from pyquaticus.config import get_std_config, ACTION_MAP
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus import pyquaticus_v0
from pygame import QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_UP, K_RIGHT
from collections import OrderedDict

RENDER_MODE = 'human'

runtime = 120 # seconds

class KeyTest:

    def __init__(self, env, quittable=True):
        '''
        Args:
            env:        the pyquaticus environment
            red_policy: if set to None, then red doesn't move unless controlled
                        if passed a policy, then controls the red team with the policy
        '''
        reset_opts = {'normalize': False}
        self.obs = env.reset(options=reset_opts)
        self.env = env
        
        self.blue_policy = BaseAttacker(env.players[0].id, env.players[0].team, mode='competition_easy')
        

        self.quittable = quittable
        self.no_op_action = 16
        straight = 4
        left = 6
        right = 2
        straightleft = 5
        straightright = 3

        self.red_keys_to_action={0              : self.no_op_action,
                                 K_UP           : straight,
                                 K_LEFT         : left,
                                 K_RIGHT        : right,
                                 K_UP + K_LEFT  : straightleft,
                                 K_UP + K_RIGHT : straightright
                                }

        self.blue_agent_id = self.env.agents_of_team[Team.BLUE_TEAM][0].id
        self.red_agent_id  = self.env.agents_of_team[Team.RED_TEAM][0].id


    def begin(self):
        while True:
            action_dict = self.process_event(self.quittable)
            
            self.obs, rewards, terminated, truncated, info = self.env.step(action_dict)
            # time.sleep(1.)
            for k in terminated:
                if terminated[k] == True or truncated[k]==True:
                    self.env.reset()
                    break

    def process_event(self, quittable):

        if quittable:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    self.env.close()
                    sys.exit()

        action_dict = OrderedDict([(player_id, self.no_op_action) for player_id in self.env.players])
        is_key_pressed = pygame.key.get_pressed()

        # blue policy
        blue_action = self.blue_policy.compute_action(self.obs)
        action_dict[self.blue_agent_id] = blue_action

        # red keys
        red_keys = K_RIGHT*is_key_pressed[K_RIGHT] + K_LEFT*is_key_pressed[K_LEFT]*(is_key_pressed[K_LEFT] - is_key_pressed[K_RIGHT]) + K_UP*is_key_pressed[K_UP]
        red_action = self.red_keys_to_action[red_keys]
        action_dict[self.red_agent_id] = red_action

        return action_dict
    
if __name__ == '__main__':
    config_dict = get_std_config()
    config_dict['normalize'] = False

    #PyQuaticusEnv is a Parallel Petting Zoo Environment
    env = pyquaticus_v0.PyQuaticusEnv(render_mode='human', team_size=1)
    kt = KeyTest(env)
    kt.begin()
