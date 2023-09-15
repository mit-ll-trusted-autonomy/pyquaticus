# external imports
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.envs.pyquaticus import Team

# internal imports
from gym_aquaticus.envs.pyquaticus_team_env_bridge import PyquaticusBridge
from gym_aquaticus.envs.config import WestPointConfig
from ray.rllib.algorithms.ppo import PPOConfig as ppo
from ray import air

from ray.rllib.policy.policy import Policy
import argparse

#This file launchs a single trained agent in simulation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the simulation with trained agents.")
    parser.add_argument('--color', required=True, choices=['red', 'blue'], help="Specify if red or blue team is the trained agent.")
    parser.add_argument('--policy-dir', required=True, help="Directory containing policy file.")
    args = parser.parse_args()

    config_dict = {
        'moos_config': WestPointConfig(),
        'shoreside_params': ('localhost', 9000, 'shoreside'),
        'red_team_params': [('localhost', 9011, 'red_one')], # ('localhost', 9012, 'red_two')
        'blue_team_params': [('localhost', 9015, 'blue_one')], # ('localhost', 9016, 'blue_two')
        'return_raw_state': False,
        'max_steps': 1000000
    }

    
    # Set the right script to use
    if args.color == 'red':
        config_dict['sim_script'] = './red.sh'
    elif args.color == 'blue':
        config_dict['sim_script'] = './blue.sh'

    env = PyquaticusBridge(config_dict)
    obs_normalizer = env.agent_obs_normalizer

    # Use the `from_checkpoint` utility of the Policy class:
    policy = Policy.from_checkpoint(args.policy_dir)


    obs = env.reset()

    done = False

    while not done:
        unnormalized_obs = {name: obs_normalizer.unnormalized(obs[name]) for name in env._agents}

        if args.color == "blue":
            blue_action = defend_policy.compute_single_action(obs['blue_one'])[0]
            obs, reward, done, info = env.step({'blue_one': blue_action})

        else:
            red_action = policy.compute_single_action(obs['red_one'])[0]
            obs, reward, done, info = env.step({'red_one': red_action})

        if done:
            break

    print('Done')
