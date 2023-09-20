import time

from pyquaticus.moos.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.moos.config import WestPointConfig

if __name__ == "__main__":
    watcher = PyQuaticusMoosBridge("localhost", "red_one", 9011,
                      [], ["blue_one"], moos_config=WestPointConfig(), 
                      quiet=False)

    try:
        obs = watcher.reset()
        action_space = watcher.action_space
        for i in range(800):
            obs, _, _, _, _ = watcher.step(action_space.sample())
        print("Finished loop")
    finally:
        watcher.close()