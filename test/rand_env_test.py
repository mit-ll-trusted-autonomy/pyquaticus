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
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE
import sys
import time
from pyquaticus.envs.pyquaticus import Team
from pyquaticus import pyquaticus_v0
import pyquaticus.utils.rewards as rew

RENDER_MODE = 'human'

runtime = 120 # seconds

def run_one_episode(env, quittable=True, render=RENDER_MODE):
    env.reset()
    full_action = {agent_id:env.action_space(agent_id).sample() for agent_id in  env.players}
    for i in range(int(runtime/env.tau)):
        if quittable and render:
            for event in pygame.event.get():
                if event.type == QUIT or (
                    event.type == KEYDOWN and event.key == K_ESCAPE
                ):
                    env.close()
                    sys.exit()

        full_action = {agent_id:env.action_space(agent_id).sample() for agent_id in env.players}
        #full_action[0] = "PM"
        new_obs, reward, terminated, truncated, info = env.step(full_action)
        for k in terminated:
            if terminated[k] == True or truncated[k] == True:
                time.sleep(1.)
                break
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run random actions in a CTF environment')

    reward_config = {1:rew.sparse, 2:rew.custom_v1, 3:None, 5:None}
    args = parser.parse_args()
    env = pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, team_size=3)
    run_one_episode(env)
