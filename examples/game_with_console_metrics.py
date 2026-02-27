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
# Part 252.227-7013 or 7024 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
Game window + console metrics - NO FLASHING OVERLAY.

See robots move in pygame window, metrics in console.
"""

import pygame
import numpy as np
import time
import os
import threading
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

def clear_console():
    """Clear console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_metrics(roles, game_state, step):
    """Print metrics to console."""
    clear_console()
    
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 18 + "HIERARCHICAL ROLES METRICS" + " " * 18 + "║")
    print("╠" + "═" * 58 + "╣")
    
    # Blue team
    print("║ 🔵 BLUE TEAM:                                                    ║")
    blue_roles = {ATTACK: 0, DEFEND: 0, INTERCEPT: 0}
    for agent_id in sorted(roles.keys()):
        if agent_id.startswith('agent_0') or agent_id.startswith('agent_1') or agent_id.startswith('agent_2'):
            role_id = roles[agent_id]
            role_name = ROLE_NAMES[role_id]
            blue_roles[role_id] += 1
            print(f"║   {agent_id:12} → {role_name:12}                                ║")
    
    print("║   Summary: A:{blue_roles[ATTACK]} D:{blue_roles[DEFEND]} I:{blue_roles[INTERCEPT]:<13}                ║")
    
    print("╠" + "═" * 58 + "╣")
    
    # Red team
    print("║ 🔴 RED TEAM:                                                     ║")
    red_roles = {ATTACK: 0, DEFEND: 0, INTERCEPT: 0}
    for agent_id in sorted(roles.keys()):
        if agent_id.startswith('agent_3') or agent_id.startswith('agent_4') or agent_id.startswith('agent_5'):
            role_id = roles[agent_id]
            role_name = ROLE_NAMES[role_id]
            red_roles[role_id] += 1
            print(f"║   {agent_id:12} → {role_name:12}                                ║")
    
    print("║   Summary: A:{red_roles[ATTACK]} D:{red_roles[DEFEND]} I:{red_roles[INTERCEPT]:<13}                ║")
    
    print("╠" + "═" * 58 + "╣")
    
    # Game state
    print("║ 📊 GAME STATE:                                                   ║")
    if 'team_has_flag' in game_state:
        blue_has = game_state['team_has_flag'][0]
        red_has = game_state['team_has_flag'][1]
        print(f"║   Flags: Blue {'✓' if blue_has else '✗'}  Red {'✓' if red_has else '✗'}                     ║")
    
    if 'captures' in game_state:
        blue_score = game_state['captures'][0]
        red_score = game_state['captures'][1]
        print(f"║   Score:  Blue {blue_score}  -  {red_score}  Red                     ║")
    
    print("║                                                                   ║")
    print(f"║   Step: {step:6d}  |  Game Window: Arrow/WASD | ESC: Quit         ║")
    print("╚" + "═" * 58 + "╝")

class GameWithConsoleMetrics:
    """Game with console metrics instead of overlay."""
    
    def __init__(self, env):
        self.env = env
        self.obs, _ = env.reset()
        
        # Action mappings
        self.no_op = 16
        self.straight = 4
        self.left = 6
        self.right = 2
        
        # Find controllable agents
        self.blue_agent = None
        self.red_agent = None
        
        for agent_id in env.agents:
            if agent_id.startswith('agent_'):
                agent_num = int(agent_id.split('_')[1])
                if agent_num == 0 and self.blue_agent is None:
                    self.blue_agent = agent_id
                elif agent_num == 3 and self.red_agent is None:
                    self.red_agent = agent_id
        
        print(f"Controlling Blue: {self.blue_agent}")
        print(f"Controlling Red: {self.red_agent}")
    
    def get_current_roles(self):
        """Get current roles for all agents."""
        roles = {}
        for agent_id in self.env.agents:
            if agent_id in self.obs:
                roles[agent_id] = extract_role_from_obs(self.obs[agent_id])
            else:
                roles[agent_id] = ATTACK
        return roles
    
    def get_game_state(self):
        """Get current game state."""
        try:
            if hasattr(self.env, 'par_env') and hasattr(self.env.par_env, 'state'):
                return self.env.par_env.state
        except Exception:
            pass
        return {}
    
    def run(self):
        """Run the game with console metrics."""
        clock = pygame.time.Clock()
        running = True
        step_count = 0
        last_console_update = 0
        
        print("Game started! Controls:")
        print("  Blue Team: Arrow Keys")
        print("  Red Team: WASD")
        print("  ESC: Quit")
        print("  Metrics in console (updates every 15 steps)")
        print()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Get keys
            keys = pygame.key.get_pressed()
            
            # Create actions
            actions = {}
            
            # Blue agent control
            if self.blue_agent:
                if keys[pygame.K_UP]:
                    actions[self.blue_agent] = self.straight
                elif keys[pygame.K_LEFT]:
                    actions[self.blue_agent] = self.left
                elif keys[pygame.K_RIGHT]:
                    actions[self.blue_agent] = self.right
                else:
                    actions[self.blue_agent] = self.no_op
            
            # Red agent control
            if self.red_agent:
                if keys[pygame.K_w]:
                    actions[self.red_agent] = self.straight
                elif keys[pygame.K_a]:
                    actions[self.red_agent] = self.left
                elif keys[pygame.K_d]:
                    actions[self.red_agent] = self.right
                else:
                    actions[self.red_agent] = self.no_op
            
            # Random actions for other agents
            for agent_id in self.env.agents:
                if agent_id not in actions:
                    actions[agent_id] = self.env.action_space(agent_id).sample()
            
            # Step environment
            self.obs, rewards, terminated, truncated, info = self.env.step(actions)
            step_count += 1
            
            # Update console metrics every 15 steps
            if step_count - last_console_update >= 15:
                roles = self.get_current_roles()
                game_state = self.get_game_state()
                print_metrics(roles, game_state, step_count)
                last_console_update = step_count
            
            # Check if episode ended
            if all(terminated.values()) or all(truncated.values()):
                print("Episode ended! Resetting...")
                self.obs, _ = self.env.reset()
                step_count = 0
                last_console_update = 0
            
            # Render game (NO OVERLAY - CLEAN GAME WINDOW)
            self.env.render()
            pygame.display.flip()
            
            clock.tick(15)  # 15 FPS
        
        print("Game completed!")

def main():
    print("🎮 GAME + CONSOLE METRICS - NO FLASHING")
    print("=" * 50)
    print("✅ Clean game window (see robots move)")
    print("✅ Metrics in console (no overlay)")
    print("✅ Zero flashing issues")
    print()
    
    # Environment config
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['team_size'] = 3
    config_dict['render_agent_ids'] = True
    config_dict['render_lidar_mode'] = "off"  # Cleaner view
    
    try:
        # Create environment with rendering
        base_env = pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict,
            render_mode='human',  # Game window!
            team_size=3
        )
        
        env = wrap_env_with_roles(base_env)
        
        print("✅ Environment created!")
        print("Starting game with console metrics...")
        print()
        
        # Run game
        game = GameWithConsoleMetrics(env)
        game.run()
        
        env.close()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
