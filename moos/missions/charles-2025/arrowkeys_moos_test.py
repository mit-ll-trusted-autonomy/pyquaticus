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
import sys
from pyquaticus.moos_bridge.config import MITConfig2025, pyquaticus_config_std
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.moos_bridge.pyquaticus_moos_bridge_ext import PyQuaticusMoosBridgeFullObs

def main():
    no_op = 16
    straight = 4
    left = 6
    right = 2
    straightleft = 5
    straightright = 3
    keys_to_action={
        0: no_op,
        'w': straight,
        'a': left,
        'd': right
    }
    env = PyQuaticusMoosBridgeFullObs(
        server="localhost",
        agent_name="red_one",
        agent_port=9011,
        team_names=["red_two", "red_three"],
        opponent_names=["blue_one", "blue_two", "blue_three"],
        all_agent_names=["blue_one", "blue_two", "blue_three", "red_one", "red_two", "red_three"],
        moos_config=MITConfig2025(),
        pyquaticus_config=pyquaticus_config_std,
        timewarp=1,
        quiet=False
    )
    env.reset()
    try:
        from pynput import keyboard

        class KeyListener(keyboard.Listener):

            def __init__(self, env):
                self.env = env
                self.action = no_op
                # Collect events until released
                listener = keyboard.Listener(
                    on_press=self.on_press)
                listener.start()

                while True:
                    obs, reward, terminated, truncated, info = self.env.step(self.action)

            def on_press(self, key):
                try:
                    if key.char in keys_to_action:
                        self.action = keys_to_action[key.char]
                except AttributeError as e:
                    print('special key {0} pressed'.format(
                        key))

        kl = KeyListener(env)

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        print("Running finally block")
        env.close()

if __name__ == "__main__":
    main()