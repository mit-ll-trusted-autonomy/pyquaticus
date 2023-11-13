import argparse
import os
import time

from pyquaticus.moos.pyquaticus_moos_bridge import PyQuaticusMoosBridge
# Note: can use pyquaticus.moos.pyquaticus_moos_bridge_ext import PyQuaticusMoosBridgeFullObs
# if you want to see observations from all agents in the field
from pyquaticus.moos.config import FieldReaderConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MOOS bridge for a given mission")
    parser.add_argument("mission_dir", help="The path to the mission directory")
    args = parser.parse_args()

    # Assuming 2v2
    # NOTE: timewarp MUST match how you've started shoreside/vehicles
    env = PyQuaticusMoosBridge("localhost", "red_one", 9011,
                      ["red_two"], ["blue_one", "blue_two"], moos_config=FieldReaderConfig(args.mission_dir),
                      timewarp=1, quiet=False)

    try:
        obs = env.reset()
        action_space = env.action_space
        for i in range(800):
            obs, _, _, _, _ = env.step(action_space.sample())
        print("Finished loop")
    finally:
        env.close()
