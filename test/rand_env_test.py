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
import copy
import gymnasium as gym
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE
import sys
import time
from pyquaticus.envs.pyquaticus import Team
from pyquaticus import pyquaticus_v0
import pyquaticus.utils.rewards as rew
import pyquaticus.config

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

    config = copy.deepcopy(pyquaticus.config.config_dict_std)
    config["gps_env"] = True
    config["env_bounds"] = "auto"
    # config["blue_flag_home"] = (41.3504170, -74.0614643)
    # config["red_flag_home"] = (41.3512143, -74.0608635)
    config["blue_flag_home"] = (42.352229714597705, -70.99992567997114)
    config["red_flag_home"] = (42.32710627259394, -70.96739585043458)
    config["flag_homes_unit"] = "ll"
    config["sim_speedup_factor"] = 4
    config["max_time"] = 700
    # config["screen_frac"] = 0.75 
    config["lidar_obs"] = True
    config["num_lidar_rays"] = 100
    config["lidar_range"] = 120
    config["render_agent_ids"] = True
    config["render_lidar"] = True
    # config["render_traj_mode"] = "traj_agent"
    config["render_traj_freq"] = 10
    config["render_traj_cutoff"] = 55
    # config["record_render"] = True
    config["recording_format"] = "mp4"
    # config["render_fps"] = 10
    # config["normalize"] = False

    env = pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, team_size=3, config_dict=config)
    run_one_episode(env)
