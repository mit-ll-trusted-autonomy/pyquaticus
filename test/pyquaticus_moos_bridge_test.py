import time

from pyquaticus.envs.pyquaticus_moos_bridge import Watcher

if __name__ == "__main__":
    watcher = Watcher("localhost", "red_one", 9011, 
                      [], ["blue_one"], quiet=False)

    for i in range(60):
        print(f"{'#'*10} REPORT {'#'*10}")
        for name, agent in watcher.players.items():
            print(f"{name}: {agent.pos}, {agent.speed} m/s, {agent.heading} deg")
        print('#'*28)
        time.sleep(1)