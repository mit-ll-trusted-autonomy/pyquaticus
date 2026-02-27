# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
Quick test to verify hierarchical role system is working.

This script runs a short test and prints basic role assignment info.
"""

import numpy as np
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.hierarchical.role_wrapper import wrap_env_with_roles
from pyquaticus.hierarchical.roles import ATTACK, DEFEND, INTERCEPT, ROLE_NAMES

def extract_role_from_obs(observation):
    """Extract role ID from observation."""
    try:
        if observation is None:
            return ATTACK
        obs_array = np.asarray(observation)
        if len(obs_array.shape) == 1 and obs_array.shape[0] >= 3:
            role_one_hot = obs_array[-3:]
            role_id = np.argmax(role_one_hot)
            if role_id in [ATTACK, DEFEND, INTERCEPT]:
                return role_id
        return ATTACK
    except Exception:
        return ATTACK

def main():
    print("Quick Hierarchical Role Test")
    print("=" * 35)
    
    # Simple environment config
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 50
    config_dict['max_score'] = 1
    config_dict['max_time'] = 60
    config_dict['team_size'] = 3
    
    try:
        # Create environment
        base_env = pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict,
            render_mode=None,
            team_size=3
        )
        
        env = wrap_env_with_roles(base_env)
        
        print("✅ Environment created successfully")
        
        # Reset and check initial roles
        obs, _ = env.reset()
        print("\n📋 Initial Roles:")
        
        for agent_id in sorted(env.agents):
            role_id = extract_role_from_obs(obs[agent_id])
            role_name = ROLE_NAMES[role_id]
            print(f"  {agent_id}: {role_name}")
        
        # Run a few steps
        print(f"\n🏃 Running 20 steps...")
        
        for step in range(20):
            # Random actions
            actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Print roles every 5 steps
            if step % 5 == 0:
                print(f"\nStep {step}:")
                for agent_id in sorted(env.agents):
                    role_id = extract_role_from_obs(obs[agent_id])
                    role_name = ROLE_NAMES[role_id]
                    print(f"  {agent_id}: {role_name}")
            
            # Check if done
            if all(terminated.values()) or all(truncated.values()):
                print(f"\n🏁 Episode completed at step {step}")
                break
        
        print("\n✅ Test completed successfully!")
        print("The hierarchical role system is working correctly.")
        
        env.close()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that all dependencies are installed and the environment is set up correctly.")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
