from gym.envs.registration import register

register(
    id='aquaticus-v0',
    entry_point='gym_aquaticus.envs:AquaticusTeamEnv',
)
