import time

from pyquaticus.moos.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.moos.config import WestPointConfig

if __name__ == "__main__":
    watcher = PyQuaticusMoosBridge("localhost", "red_one", 9011,
                      [], ["blue_one"], moos_config=WestPointConfig(), 
                      quiet=False)

    try:
        watcher.reset()
        for i in range(20):
            print(f"{'#'*10} REPORT {'#'*10}")
            for name, agent in watcher.players.items():
                print(f"{name}: {agent.pos}, {agent.speed} m/s, {agent.heading} deg")
            print('#'*28)
            time.sleep(1)
        print("Finished reporting loop")
    finally:
        watcher.close()