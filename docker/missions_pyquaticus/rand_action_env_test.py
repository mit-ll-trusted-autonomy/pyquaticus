# internal imports
from gym_aquaticus.envs.aquaticus_team_env import AquaticusTeamEnv
from gym_aquaticus.envs.config import WestPointConfig


if __name__ == "__main__":
    config_dict = {
        'moos_config': WestPointConfig(),
        'shoreside_params': ('localhost', 9000, 'shoreside'),
        'red_team_params': [('localhost', 9011, 'red_one')],
        'blue_team_params': [('localhost', 9015, 'blue_one')],
        'sim_script': './demo_1v1.sh',
        'return_raw_state': False,
        'max_steps': 1000
    }

    env = AquaticusTeamEnv(config_dict)
    obs = env.reset()

    done = False
    for i in range(500):
        obs, reward, done, info = env.step(env.action_space.sample())
        # AquaticusTeamEnv returns a dictionary with done marked for each agent by default
        if done['__all__']:
            break

    print('Done')
