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
from pygame import (
    KEYDOWN,
    QUIT,
    K_ESCAPE,
    K_SPACE,
    K_LEFT,
    K_UP,
    K_RIGHT,
    K_a,
    K_w,
    K_d,
)
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus.config
import copy
from pyquaticus import pyquaticus_v0
import pyquaticus.utils.rewards as reward


class KeyTest:

    def __init__(self, env, red_policy=None, quittable=True):
        """
        Args:
            env:        the pyquaticus environment
            red_policy: if set to None, then red doesn't move unless controlled
                        if passed a policy, then controls the red team with the policy
        """
        self.obs = env.reset()
        self.env = env
        self.policy = None
        if red_policy is not None:
            self.policy = red_policy(env.observation_space, env.action_space, {})

        self.quittable = quittable

        if "PyQuaticusEnv" in str(env):
            no_op = 16
            straight = 4
            left = 6
            right = 2
            straightleft = 5
            straightright = 3
        else:
            assert "ctf" in str(env)
            no_op = 0
            straight = 1
            left = 2
            right = 3
            straightleft = 4
            straightright = 5

        self.no_op_action = no_op

        self.blue_keys_to_action = {
            0: no_op,
            K_w: straight,
            K_a: left,
            K_d: right,
            K_w + K_a: straightleft,
            K_w + K_d: straightright,
        }

        self.red_keys_to_action = {
            0: no_op,
            K_UP: straight,
            K_LEFT: left,
            K_RIGHT: right,
            K_UP + K_LEFT: straightleft,
            K_UP + K_RIGHT: straightright,
        }

        self.blue_agent_id = self.env.agents_of_team[Team.BLUE_TEAM][0].id
        self.red_agent_id = self.env.agents_of_team[Team.RED_TEAM][0].id

    def begin(self):
        while True:
            action_dict = self.process_event(self.quittable)
            self.obs, rewards, terminated, truncated, info = self.env.step(action_dict)
            k = list(terminated.keys())
            if terminated[k[0]] == True or truncated[k[0]] == True:
                break
            # for k in terminated:
            #     if terminated[k] == True or truncated[k] == True:
            #         # time.sleep(1.0)
            #         # self.env.reset()
            #         break

    def process_event(self, quittable):

        if quittable:
            for event in pygame.event.get():
                if event.type == QUIT or (
                    event.type == KEYDOWN and event.key == K_ESCAPE
                ):
                    self.env.close()
                    sys.exit()

        action_dict = OrderedDict(
            [(player_id, self.no_op_action) for player_id in self.env.players]
        )
        is_key_pressed = pygame.key.get_pressed()

        # blue keys
        blue_keys = (
            K_d * is_key_pressed[K_d]
            + K_a * is_key_pressed[K_a] * (is_key_pressed[K_a] - is_key_pressed[K_d])
            + K_w * is_key_pressed[K_w]
        )
        blue_action = self.blue_keys_to_action[blue_keys]
        action_dict[self.red_agent_id] = blue_action

        if self.policy is not None:
            action_dict[self.red_agent_id] = self.policy.compute_action(
                self.obs[self.red_agent_id]
            )

        else:
            # red keys
            red_keys = (
                K_RIGHT * is_key_pressed[K_RIGHT]
                + K_LEFT
                * is_key_pressed[K_LEFT]
                * (is_key_pressed[K_LEFT] - is_key_pressed[K_RIGHT])
                + K_UP * is_key_pressed[K_UP]
            )
            red_action = self.red_keys_to_action[red_keys]
            action_dict[self.blue_agent_id] = red_action
        return action_dict


def main():
    parser = argparse.ArgumentParser(
        description="Play CTF manually (optionally against a policy)"
    )
    parser.add_argument(
        "--red-policy",
        default=None,
        choices=[],
        help="Select a red policy to play against.",
    )
    args = parser.parse_args()

    # Setup for 1 blue player and 1 red player with sparse rewards
    reward_config = {0: None, 1: None}
    # Alternative
    # reward_config = {0:reward.sparse, 1:reward.sparse}

    config = copy.deepcopy(pyquaticus.config.config_dict_std)
    # config["normalize"] = False
    config["max_time"] = 600.0
    config["max_score"] = 3
    config["tau"] = 0.1
    config["sim_speedup_factor"] = 4
    config["render_agent_ids"] = True
    config["render_traj_mode"] = "traj"
    config["render_traj_cutoff"] = 100
    # config["arena_buffer_frac"] = 0.25
    # config["render_fps"] = 10
    # config["dynamics"] = ["heron", "heron"]
    # config["lidar_obs"] = True
    # config["render_lidar_mode"] = "detection"
    # config["lidar_range"] = 20
    # config["render_traj_mode"] = "traj_history"
    # config["render_traj_freq"] = 50
    # config["short_hist_length"] = 4
    # config["long_hist_length"] = 5
    # config["long_hist_interval"] = 20
    # config["render_traj_cutoff"] = 300
    config["obstacles"] = {
        # "circle": [(4*2, (6*2, 5*2))],
        # "polygon": [((70, 10), (85, 21), (83, 51), (72, 35))]
    }

    # PyQuaticusEnv is a Parallel Petting Zoo Environment
    env = pyquaticus_v0.PyQuaticusEnv(
        render_mode="human", team_size=1, config_dict=config
    )
    red_policy = args.red_policy

    kt = KeyTest(env, red_policy)
    kt.begin()
    # env.buffer_to_video()
    env.close()


if __name__ == "__main__":
    main()
