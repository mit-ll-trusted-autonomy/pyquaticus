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
from collections import OrderedDict
import gymnasium as gym
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE, K_SPACE, K_LEFT, K_UP, K_RIGHT, K_a, K_w, K_d, K_TAB
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus.config
import copy
from pyquaticus import pyquaticus_v0


class KeyTest:

    def __init__(self, env, quittable=True):
        '''
        Args:
            env: the pyquaticus environment
        '''
        self.obs, _ = env.reset()
        # render initial frame for env 0 so pygame display initializes
        env.render(0)
        self.env = env

        self.quittable = quittable

        # support multiple parallel envs and keep track of selected env
        self.n_envs = getattr(env, 'n_envs', 1)
        self.selected_env = 0

        no_op = 16
        straight = 4
        left = 6
        right = 2
        straightleft = 5
        straightright = 3

        self.no_op_action = no_op

        self.blue_keys_to_action = {
            0              : no_op,
            K_UP           : straight,
            K_LEFT         : left,
            K_RIGHT        : right,
            K_UP + K_LEFT  : straightleft,
            K_UP + K_RIGHT : straightright
        }
        self.red_keys_to_action = {
            0         : no_op,
            K_w       : straight,
            K_a       : left,
            K_d       : right,
            K_w + K_a : straightleft,
            K_w + K_d : straightright
        }

        self.blue_agent_id = self.env.agents_of_team[Team.BLUE_TEAM][0].id
        self.red_agent_id  = self.env.agents_of_team[Team.RED_TEAM][0].id


    def begin(self):
        while True:
            # render the currently-selected environment
            self.env.render(self.selected_env)

            action_dict = self.process_event(self.quittable)

            # step all environments concurrently
            self.obs, rewards, terminated, truncated, info = self.env.step(action_dict)

            # determine which envs are done (terminated or truncated) and reset only those
            done_mask = np.zeros(self.n_envs, dtype=bool)
            for k in terminated:
                try:
                    done_mask |= np.asarray(terminated[k], dtype=bool)
                except Exception:
                    pass
            for k in truncated:
                try:
                    done_mask |= np.asarray(truncated[k], dtype=bool)
                except Exception:
                    pass

            done_envs = np.where(done_mask)[0]
            if done_envs.size > 0:
                time.sleep(1.)
                self.env.reset(env_idxs=done_envs.tolist())

            # render updated state for the selected environment
            self.env.render(self.selected_env)

    def process_event(self, quittable):

        if quittable:
            for event in pygame.event.get():
                if event.type == QUIT or (
                    event.type == KEYDOWN and event.key == K_ESCAPE
                ):
                    self.env.close()
                    sys.exit()

                # switch which parallel env we're controlling
                if event.type == KEYDOWN and event.key == K_TAB:
                    self.selected_env = (self.selected_env + 1) % max(1, self.n_envs)

        # initialize per-agent actions for all envs as no-op
        action_dict = OrderedDict()
        for player_id in self.env.players:
            action_dict[player_id] = np.full(self.n_envs, self.no_op_action, dtype=int)

        is_key_pressed = pygame.key.get_pressed()

        # blue keys (compute desired action for the selected env)
        blue_keys = K_RIGHT*is_key_pressed[K_RIGHT] + K_LEFT*is_key_pressed[K_LEFT]*(is_key_pressed[K_LEFT] - is_key_pressed[K_RIGHT]) + K_UP*is_key_pressed[K_UP]
        blue_action = self.blue_keys_to_action[blue_keys]
        action_dict[self.blue_agent_id][self.selected_env] = blue_action

        # red keys
        red_keys = K_d*is_key_pressed[K_d] + K_a*is_key_pressed[K_a]*(is_key_pressed[K_a] - is_key_pressed[K_d]) + K_w*is_key_pressed[K_w]
        red_action = self.red_keys_to_action[red_keys]
        action_dict[self.red_agent_id][self.selected_env] = red_action

        return action_dict

def main():
    config = {}
    config["timewarp"] = 8
    config["max_time"] = 1000
    config["render_agent_ids"] = True
    config["tag_on_oob"] = True


    
    #PyQuaticusEnv is a Parallel Petting Zoo Environment
    env = pyquaticus_v0.PyQuaticusEnv(render_mode='human', n_envs=10, team_size=1, config_dict=config)
    kt = KeyTest(env)
    kt.begin()

if __name__ == "__main__":
    main()
