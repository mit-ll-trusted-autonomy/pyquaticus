import argparse
import time


from pyquaticus.moos.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.moos.config import WestPointConfig, JervisBayConfig,FieldReaderConfig
from solution import solution

boat_ports = {
    'blue_one': 9015,
    'blue_two': 9016,
    'blue_three': 9017,
    'red_one': 9011,
    'red_two': 9012,
    'red_three': 9013,
}

boat_ips = {
    's': "192.168.1.12",
    't': "192.168.1.22",
    'u': "192.168.1.32",
    'v': "192.168.1.42",
    'w': "192.168.1.52",
    'x': "192.168.1.62",
    'y': "192.168.1.72",
    'z': "192.168.1.82"
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the simulation with trained agents.")
    # parser.add_argument('--sim', required=True, choices=['true', 'false'], help="Specify if simulation or not.")
    parser.add_argument('--sim', action='store_true', help="Specify if simulation or not.")
    parser.add_argument('--color', required=True, choices=['red', 'blue'], help="Specify if red or blue team is the trained agent.")
    parser.add_argument('--boat_id', required=True, choices=["blue_one", "blue_two", "red_one", "red_two"], help="Specify the boat id.")
    parser.add_argument('--boat_name', required=False, choices=['s', 't', 'u', 'v', 'w', 'x', 'y', 'z'], help="Specify the boat name.")
    parser.add_argument('--num-players', required=True, type=int, help="Specify the number of players on each team.")
    parser.add_argument('--timewarp', required=True, type=int, default=4, help='Specify the timewarp.')
    args = parser.parse_args()

    if args.sim:
        print("Simulation mode")
        server = "localhost"
    else:
        server = boat_ips[args.boat_name]
    
    boat_ids = list(boat_ports.keys())
    teammates = [bid for bid in boat_ids if bid.startswith(args.color) and bid != args.boat_id][:args.num_players-1]  # Exclude the specified boat_id and limit to num_players
    opponents = [bid for bid in boat_ids if not bid.startswith(args.color)][:args.num_players]

    print(f"Teammates: {teammates}")
    print(f"Opponents: {opponents}")
    print(f"Connecting to {server}:{boat_ports[args.boat_id]}")
    print(f"Boat id: {args.boat_id}")
    print(f"Boat name: {args.boat_name}")
    print(f"Num players: {args.num_players}")
    env = PyQuaticusMoosBridge(server, args.boat_id, boat_ports[args.boat_id],
                      teammates, opponents, moos_config=FieldReaderConfig('/home/john/moos-ivp-aquaticus/missions/wp_2024'), timewarp=args.timewarp,
                      quiet=False)

    # Catch Ctrl-C

    # Write here a function that we can call to pass the actions into the watcher
    # def run_moos_agent(policy):
    #policy(obs) -> action
    #sol = solution()
    try:
        agent_id = 0 if "one" in args.boat_id else 1
        obs_norm = env.reset()
        env.normalize = False
        obs = env.state_to_obs(args.boat_id)
        env.normalize = True
        action_space = env.action_space
        while True:
            #Get action from learned policy
            #action = sol.compute_action(agent_id, obs_norm, obs)
            obs_norm, _, _, _, _ = env.step(-1)
            env.normalize = False
            obs = env.state_to_obs(args.boat_id)
            env.normalize = True
        print("Finished loop")

    finally:
        print("Interrupted by user")
        env.close()
