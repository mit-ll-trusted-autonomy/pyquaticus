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

if __name__ == "__main__":
    config_dict = {
        'moos_config': WestPointConfig(),
        'shoreside_params': ('localhost', 9000, 'shoreside'),
        'red_team_params': [('localhost', 9011, 'red_one')], # ('localhost', 9012, 'red_two')
        'blue_team_params': [('localhost', 9015, 'blue_one')], # ('localhost', 9016, 'blue_two')
        'sim_script': './demo_1v1.sh',
        'return_raw_state': False,
        'max_steps': 1000000
    }

    env = PyquaticusBridge(config_dict)
    obs_normalizer = env.agent_obs_normalizer

    
    # Use the `from_checkpoint` utility of the Policy class:
    attack_policy = Policy.from_checkpoint("./rllib_policies/attacker-1-policy")
    defend_policy = Policy.from_checkpoint("./rllib_policies/defender-0-policy")
    base_attacker = BaseAttacker('red_one', Team.RED_TEAM, mode='hard')
    base_defender = BaseDefender('blue_one', Team.BLUE_TEAM, mode='medium')

    obs = env.reset()

    done = False
    print(env._agents)
    for i in range(500):
        unnormalized_obs = {name: obs_normalizer.unnormalized(obs[name]) for name in env._agents}
        # blue_action = base_defender.compute_action(unnormalized_obs)
        blue_action = defend_policy.compute_single_action(obs['blue_one'])[0]
        # print(blue_action)
        # print(obs['red_one'])
        # red_action = attack_policy.compute_single_action(obs['red_one'])[0]
        red_action = base_attacker.compute_action(unnormalized_obs)
        # print(red_action)
        obs, reward, done, info = env.step({'red_one': red_action,
                                            'blue_one': blue_action})
        if done:
            break

    print('Done')
