"""Quick test to verify observation normalization."""
import numpy as np
from pyquaticus.envs.dynamic_pyquaticus import DynamicPyQuaticusEnv
from pyquaticus.envs.graph_obs_wrapper import GraphObsWrapper
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import pyquaticus.utils.rewards as rew

# Create environment
cfg = {
    "sim_speedup_factor": 4,
    "max_score": 3,
    "max_time": 240,
    "tagging_cooldown": 60,
    "tag_on_oob": True,
}

reward_config = {
    "agent_0": rew.caps_and_grabs, "agent_1": rew.caps_and_grabs, "agent_2": rew.caps_and_grabs,
    "agent_3": rew.caps_and_grabs, "agent_4": rew.caps_and_grabs, "agent_5": rew.caps_and_grabs,
}

env = DynamicPyQuaticusEnv(
    team_size_range=(1, 3),
    tag_removes_agent=False,
    reinforcement_interval=0,
    config_dict=cfg,
    reward_config=reward_config,
)

# Wrap with graph wrapper
env = GraphObsWrapper(env, flatten_for_fc=True)
env = ParallelPettingZooWrapper(env)

# Reset and check observations
print("Resetting environment...")
obs, info = env.reset()

print(f"\nNumber of agents: {len(obs)}")
print(f"Agent IDs: {list(obs.keys())}")

# Check each observation
for agent_id, observation in obs.items():
    obs_array = np.asarray(observation)
    print(f"\n{agent_id}:")
    print(f"  Shape: {obs_array.shape}")
    print(f"  Min: {obs_array.min():.3f}")
    print(f"  Max: {obs_array.max():.3f}")
    print(f"  Mean: {obs_array.mean():.3f}")
    
    # Check for values outside [-1, 1]
    out_of_range = np.abs(obs_array) > 1.0
    if np.any(out_of_range):
        print(f"  ⚠️  WARNING: {np.sum(out_of_range)} values outside [-1, 1]!")
        print(f"  Out of range indices: {np.where(out_of_range)[0]}")
        print(f"  Out of range values: {obs_array[out_of_range]}")
    else:
        print(f"  ✓ All values in [-1, 1]")

# Step a few times
print("\n\nStepping environment 10 times...")
# Get action space info
first_aid = list(obs.keys())[0]
if hasattr(env, "action_space") and callable(env.action_space):
    test_space = env.action_space(first_aid)
elif hasattr(env, "action_spaces") and isinstance(env.action_spaces, dict):
    test_space = env.action_spaces[first_aid]
else:
    par_env = getattr(env, "par_env", env)
    if hasattr(par_env, "action_space") and callable(par_env.action_space):
        test_space = par_env.action_space(first_aid)
    else:
        test_space = par_env.action_spaces[first_aid] if hasattr(par_env, "action_spaces") else None

print(f"Action space type: {type(test_space).__name__}")
if test_space:
    print(f"Action space: {test_space}")
    # Try sampling once to see what we get
    try:
        sample_action = test_space.sample()
        print(f"Sample action type: {type(sample_action)}, value: {sample_action}")
    except Exception as e:
        print(f"Error sampling action: {e}")

for i in range(10):
    # Use simple discrete actions (no-op = 16, or random 0-16)
    actions = {aid: np.random.randint(0, 17) for aid in obs.keys()}
    
    try:
        obs, rewards, term, trunc, info = env.step(actions)
    except Exception as e:
        print(f"\nError on step {i}: {e}")
        print(f"Actions: {actions}")
        import traceback
        traceback.print_exc()
        break
    
    # Check observations and rewards after step
    if i == 9:  # Check last step
        print("\nAfter 10 steps:")
        for agent_id, observation in obs.items():
            obs_array = np.asarray(observation)
            out_of_range = np.abs(obs_array) > 1.0
            if np.any(out_of_range):
                print(f"  {agent_id}: ⚠️  {np.sum(out_of_range)} values outside [-1, 1]")
                print(f"    Values: {obs_array[out_of_range]}")
            else:
                print(f"  {agent_id}: ✓ OK")
        
        # Check rewards
        print("\nRewards:")
        for agent_id, reward in rewards.items():
            if np.isnan(reward) or np.isinf(reward):
                print(f"  {agent_id}: ⚠️  Invalid reward: {reward}")
            else:
                print(f"  {agent_id}: {reward:.3f}")
        
        # Check termination/truncation
        print(f"\nTerminated: {any(term.values())}")
        print(f"Truncated: {any(trunc.values())}")

env.close()
print("\nTest complete!")
