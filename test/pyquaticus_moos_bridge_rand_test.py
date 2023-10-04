import time

from pyquaticus.moos.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.moos.config import WestPointConfig, JervisBayConfig

if __name__ == "__main__":
    # NOTE must pick the corresponding configuration object based on which mission you're running
    # and must choose the correct teammates/opponents and timewarp
    env = PyQuaticusMoosBridge("localhost", "red_one", 9011,
                      [], ["blue_one"], moos_config=WestPointConfig(),
                      timewarp=4, quiet=False)

    try:
        obs = env.reset()
        action_space = env.action_space
        for i in range(800):
            obs, _, _, _, _ = env.step(action_space.sample())
        print("Finished loop")
    finally:
        env.close()
