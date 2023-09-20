from gym_aquaticus.envs.aquaticus_team_env import AquaticusTeamEnv
try:
    import pyquaticus
    PYQUATICUS_AVAILABLE=True
except ImportError:
    PYQUATICUS_AVAILABLE=False

if PYQUATICUS_AVAILABLE:
    from gym_aquaticus.envs.pyquaticus_team_env_bridge import PyquaticusBridge

