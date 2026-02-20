# SPDX-License-Identifier: BSD-3-Clause
"""
Dynamic PyQuaticus Environment

Extends PyQuaticusEnv to support:
- Variable team sizes (1-3 per team) randomized at each reset
- Mid-game agent removal (captured/destroyed) via disabled_agents
- Mid-game agent spawning (reinforcements)
- disabled_agents state for tracking active agents
"""

import random
from typing import Optional, Union

import numpy as np

from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.structs import Team


class DynamicPyQuaticusEnv(PyQuaticusEnv):
    """
    PyQuaticus environment with dynamic team sizes and mid-game agent changes.

    - team_size_range: (min, max) agents per team at reset (default (1, 3))
    - tag_removes_agent: if True, tagged agents are disabled (removed from game)
    - reinforcement_interval: steps between reinforcement spawns (0 = disabled)
    - reinforcement_prob: probability of spawning a reinforcement when interval hits
    """

    def __init__(
        self,
        team_size_range: tuple[int, int] = (1, 3),
        tag_removes_agent: bool = False,
        reinforcement_interval: int = 0,
        reinforcement_prob: float = 0.5,
        action_space: Union[str, list[str], dict[str, str]] = "discrete",
        reward_config: dict = None,
        config_dict=None,
        render_mode: Optional[str] = None,
    ):
        max_team_size = team_size_range[1]
        super().__init__(
            team_size=max_team_size,
            action_space=action_space,
            reward_config=reward_config,
            config_dict=config_dict or {},
            render_mode=render_mode,
        )

        self.team_size_range = team_size_range
        self.tag_removes_agent = tag_removes_agent
        self.reinforcement_interval = reinforcement_interval
        self.reinforcement_prob = reinforcement_prob
        self.num_blue_active = max_team_size
        self.num_red_active = max_team_size
        self._step_count = 0

        for player in self.players.values():
            if not hasattr(player, "is_disabled"):
                player.is_disabled = False

    def _register_state_elements(self, num_on_team, num_obstacles):
        """Override to add is_disabled to observation space."""
        agent_obs_normalizer, global_state_normalizer = super()._register_state_elements(num_on_team, num_obstacles)
        max_bool, min_bool = [1.0], [0.0]
        agent_obs_normalizer.register("is_disabled", max_bool, min_bool)
        for i in range(num_on_team - 1):
            agent_obs_normalizer.register((f"teammate_{i}", "is_disabled"), max_bool, min_bool)
        for i in range(num_on_team):
            agent_obs_normalizer.register((f"opponent_{i}", "is_disabled"), max_bool, min_bool)
        for player in self.players.values():
            global_state_normalizer.register((player.id, "is_disabled"), max_bool, min_bool)
        return agent_obs_normalizer, global_state_normalizer

    def state_to_obs(self, agent_id, normalize=True):
        """Override to add is_disabled to observations."""
        obs, unnorm = super().state_to_obs(agent_id, normalize=False)
        disabled = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))
        agent = self.players[agent_id]
        obs["is_disabled"] = float(disabled[agent.idx])
        for team in [agent.team, (Team.RED_TEAM if agent.team == Team.BLUE_TEAM else Team.BLUE_TEAM)]:
            dif_agents = [a for a in self.agents_of_team[team] if a.id != agent.id]
            for i, dif_agent in enumerate(dif_agents):
                entry_name = f"teammate_{i}" if team == agent.team else f"opponent_{i}"
                obs[(entry_name, "is_disabled")] = float(disabled[dif_agent.idx])
        if normalize:
            return self.agent_obs_normalizer.normalized(obs), obs
        return obs, None

    def _set_initial_disabled(self, num_blue_active: int, num_red_active: int):
        """Disable agents not in the active set."""
        disabled = np.ones(self.num_agents, dtype=bool)
        for i in range(num_blue_active):
            disabled[i] = False
        for i in range(self.num_blue, self.num_blue + num_red_active):
            disabled[i] = False
        self.state["disabled_agents"] = disabled
        for i, player in enumerate(self.players.values()):
            player.is_disabled = bool(disabled[i])

    def reset(self, seed=None, options: Optional[dict] = None):
        """Reset with randomized team sizes."""
        super().reset(seed=seed, options=options)
        self._step_count = 0

        min_size, max_size = self.team_size_range
        self.num_blue_active = random.randint(min_size, max_size)
        self.num_red_active = random.randint(min_size, max_size)

        if "disabled_agents" not in self.state:
            self.state["disabled_agents"] = np.zeros(self.num_agents, dtype=bool)
        self._set_initial_disabled(self.num_blue_active, self.num_red_active)
        self.state["num_blue_active"] = self.num_blue_active
        self.state["num_red_active"] = self.num_red_active

        obs = {aid: self._history_to_obs(aid, "obs_hist_buffer") for aid in self.players}
        global_state = self._history_to_state()
        info = {
            aid: {
                "global_state": global_state,
                "num_blue_active": self.num_blue_active,
                "num_red_active": self.num_red_active,
                "disabled_agents": self.state["disabled_agents"],
            }
            for aid in self.players
        }
        for aid in self.agents:
            if self.normalize_obs:
                info[aid]["unnorm_obs"] = self._history_to_obs(aid, "unnorm_obs_hist_buffer")
        return obs, info

    def step(self, raw_action_dict):
        """Step with no-op for disabled agents, tag removal, and reinforcements."""
        # Patch actions for disabled agents (discrete no-op = 16, continuous = [0,0])
        patched = dict(raw_action_dict)
        for i, player in enumerate(self.players.values()):
            if getattr(player, "is_disabled", False):
                if self.act_space_str.get(player.id, "discrete") == "continuous":
                    patched[player.id] = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    patched[player.id] = 16  # no-op in ACTION_MAP

        obs, rewards, terminated, truncated, info = super().step(patched)
        self._step_count += 1

        # Tag removes agent (disable)
        if self.tag_removes_agent:
            for i, player in enumerate(self.players.values()):
                if player.is_tagged and not getattr(player, "is_disabled", False):
                    self.state["disabled_agents"][i] = True
                    player.is_disabled = True
                    if i < self.num_blue:
                        self.num_blue_active = max(0, self.num_blue_active - 1)
                    else:
                        self.num_red_active = max(0, self.num_red_active - 1)
                    self.state["num_blue_active"] = self.num_blue_active
                    self.state["num_red_active"] = self.num_red_active

        # Reinforcement spawn
        if self.reinforcement_interval > 0 and self._step_count % self.reinforcement_interval == 0:
            if self._step_count > 0 and random.random() < self.reinforcement_prob:
                self._spawn_reinforcement()

        for aid in self.agents:
            info[aid]["num_blue_active"] = self.state.get("num_blue_active", self.num_blue_active)
            info[aid]["num_red_active"] = self.state.get("num_red_active", self.num_red_active)
            info[aid]["disabled_agents"] = self.state.get("disabled_agents", np.zeros(self.num_agents, dtype=bool))

        return obs, rewards, terminated, truncated, info

    def _spawn_reinforcement(self):
        """Re-enable one disabled agent per team."""
        disabled = self.state["disabled_agents"]
        for team in [Team.BLUE_TEAM, Team.RED_TEAM]:
            inds = self.agent_inds_of_team[team]
            disabled_team = [i for i in inds if disabled[i]]
            if disabled_team and random.random() < 0.5:
                idx = random.choice(disabled_team)
                self.state["disabled_agents"][idx] = False
                self.players[self.agents[idx]].is_disabled = False
                flag_home = np.array(self.flags[int(team)].home)
                offset = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
                new_pos = np.clip(flag_home + offset, [0, 0], self.env_size)
                self.state["agent_position"][idx] = new_pos
                p = self.players[self.agents[idx]]
                p.pos = self.state["agent_position"][idx].tolist()
                p.prev_pos = p.pos.copy()
                p.is_tagged = False
                self.state["agent_is_tagged"][idx] = False
                if team == Team.BLUE_TEAM:
                    self.num_blue_active = min(self.num_blue, self.num_blue_active + 1)
                else:
                    self.num_red_active = min(self.num_red, self.num_red_active + 1)
                self.state["num_blue_active"] = self.num_blue_active
                self.state["num_red_active"] = self.num_red_active
                break
