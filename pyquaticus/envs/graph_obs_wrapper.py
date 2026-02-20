# SPDX-License-Identifier: BSD-3-Clause
"""
Graph Observation Wrapper for PyQuaticus

Converts vector observations to graph format:
- node_features: (max_agents, feat_dim) per-agent features
- edge_index: (2, num_edges) for all-to-all or distance-based edges
- mask: (max_agents,) 1.0 if agent active, 0.0 if disabled
- num_nodes: int (for variable-size handling)

Output is a Dict that can be flattened for standard policies or used with GNNs.
"""

from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Dict

from pettingzoo import ParallelEnv

# Try importing RLLib classes for compatibility
try:
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ParallelPettingZooEnv = None


# Node feature dim: x, y, heading_norm, speed_norm, has_flag, is_tagged, on_side, team, is_disabled, cooldown_norm
NODE_FEAT_DIM = 10
MAX_AGENTS = 6


def _state_to_node_features(env, agent_id: str, global_state: dict, disabled: np.ndarray, env_size, max_speed) -> np.ndarray:
    """Build node features for all agents (max 6). All values normalized to [-1, 1]."""
    features = np.zeros((MAX_AGENTS, NODE_FEAT_DIM), dtype=np.float32)
    env_size = np.asarray(env_size)
    max_speed = float(max_speed) if max_speed > 0 else 1.0
    
    for i, (aid, player) in enumerate(env.players.items()):
        if i >= MAX_AGENTS:
            break
        
        # Position (normalize to [-1, 1])
        pos = global_state.get((aid, "pos"), np.zeros(2))
        pos = np.asarray(pos, dtype=np.float32)
        if len(pos) >= 2:
            # Check if already normalized (in [-1, 1])
            if np.all((-1.0 <= pos) & (pos <= 1.0)):
                features[i, 0] = float(pos[0])
                features[i, 1] = float(pos[1])
            else:
                # Normalize from meters to [-1, 1]
                features[i, 0] = np.clip(2.0 * pos[0] / env_size[0] - 1.0, -1.0, 1.0)
                features[i, 1] = np.clip(2.0 * pos[1] / env_size[1] - 1.0, -1.0, 1.0)
        
        # Heading (normalize degrees to [-1, 1], assuming range [-180, 180] or [0, 360])
        h_raw = global_state.get((aid, "heading"), 0)
        # Convert to scalar float, handling numpy arrays
        if isinstance(h_raw, np.ndarray):
            h = float(h_raw.item() if h_raw.size == 1 else h_raw[0])
        else:
            h = float(h_raw)
        # Always normalize if abs value > 1 (treat as degrees)
        if abs(h) > 1.0:
            # Normalize: heading in degrees -> [-1, 1]
            # Handle both [-180, 180] and [0, 360] ranges
            if h > 180:
                h = h - 360  # Convert [180, 360] to [-180, 0]
            elif h < -180:
                h = h + 360  # Convert [-360, -180] to [0, 180]
            h_norm = h / 180.0
            features[i, 2] = float(np.clip(h_norm, -1.0, 1.0))
        else:
            # Already normalized, just clip to be safe
            features[i, 2] = float(np.clip(h, -1.0, 1.0))
        
        # Speed (normalize to [0, 1] then scale to [-1, 1])
        s = global_state.get((aid, "speed"), 0)
        s = float(s)
        # Always normalize if abs value > 1 (treat as m/s)
        if abs(s) > 1.0:
            # Normalize: speed / max_speed -> [0, 1] -> [-1, 1]
            norm_speed = np.clip(s / max_speed, 0.0, 1.0)
            features[i, 3] = 2.0 * norm_speed - 1.0  # [0, 1] -> [-1, 1]
        else:
            # Already normalized, just clip to be safe
            features[i, 3] = np.clip(s, -1.0, 1.0)
        
        # Boolean features (already -1 or 1)
        features[i, 4] = 1.0 if global_state.get((aid, "has_flag"), False) else -1.0
        features[i, 5] = 1.0 if global_state.get((aid, "is_tagged"), False) else -1.0
        features[i, 6] = 1.0 if global_state.get((aid, "on_side"), True) else -1.0
        features[i, 7] = 1.0 if player.team.value == 0 else -1.0
        features[i, 8] = 1.0 if (i < len(disabled) and disabled[i]) else -1.0
        
        # Cooldown (normalize to [-1, 1])
        cooldown = global_state.get((aid, "tagging_cooldown"), 0)
        cooldown = float(cooldown)
        # Always normalize if abs value > 1 (treat as seconds)
        if abs(cooldown) > 1.0:
            # Normalize: cooldown / max_cooldown -> [0, 1] -> [-1, 1]
            max_cooldown = 60.0  # Default tagging cooldown
            norm_cooldown = np.clip(cooldown / max_cooldown, 0.0, 1.0)
            features[i, 9] = 2.0 * norm_cooldown - 1.0
        else:
            # Already normalized, just clip to be safe
            features[i, 9] = np.clip(cooldown, -1.0, 1.0)
    
    # Final safety check: ensure all features are in [-1, 1]
    # Convert to float32 and clip aggressively
    features = np.asarray(features, dtype=np.float32)
    features = np.clip(features, -1.0, 1.0)
    # Double-check: if any value is still outside [-1, 1], force clip
    if np.any(np.abs(features) > 1.0):
        features = np.clip(features, -1.0, 1.0)
    return features


def _build_edge_index(max_agents: int = MAX_AGENTS, fully_connected: bool = True) -> np.ndarray:
    """All-to-all edges."""
    if fully_connected:
        edges = []
        for i in range(max_agents):
            for j in range(max_agents):
                if i != j:
                    edges.append([i, j])
        return np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return np.zeros((2, 0), dtype=np.int64)


class GraphObsWrapper(ParallelEnv):
    """Wraps a PyQuaticus env to provide graph observations.
    
    If wrapping a ParallelPettingZooEnv, preserves MultiAgentEnv compatibility.
    """

    def __init__(self, env: ParallelEnv, flatten_for_fc: bool = True):
        # Check if we should inherit from RLLib's wrapper for compatibility
        if HAS_RAY and isinstance(env, ParallelPettingZooEnv):
            # Don't double-wrap - this shouldn't happen, but handle it
            super().__init__()
        else:
            super().__init__()
        self.env = env
        self.flatten_for_fc = flatten_for_fc
        self.par_env = getattr(env, "par_env", env)
        # Get agents from wrapped env - try multiple ways
        self.possible_agents = getattr(env, "possible_agents", None)
        if self.possible_agents is None:
            # Try getting from par_env
            self.possible_agents = getattr(self.par_env, "possible_agents", [f"agent_{i}" for i in range(6)])
        self.agents = getattr(env, "agents", None)
        if self.agents is None:
            self.agents = getattr(self.par_env, "agents", self.possible_agents.copy())
        self.metadata = getattr(env, "metadata", {})

        # Observation space: Dict with node_features, edge_index, mask, self_node_idx (for GNN policy)
        node_space = Box(low=-1, high=1, shape=(MAX_AGENTS, NODE_FEAT_DIM), dtype=np.float32)
        mask_space = Box(low=0, high=1, shape=(MAX_AGENTS,), dtype=np.float32)
        edge_space = Box(low=0, high=MAX_AGENTS - 1, shape=(2, MAX_AGENTS * (MAX_AGENTS - 1)), dtype=np.int64)
        self_node_space = Box(low=0, high=MAX_AGENTS - 1, shape=(1,), dtype=np.int64)  # which node is the acting agent
        self._graph_obs_space = Dict({
            "node_features": node_space,
            "edge_index": edge_space,
            "mask": mask_space,
            "self_node_idx": self_node_space,
        })
        # Initialize spaces - use default agents if not available yet
        default_agents = self.possible_agents if self.possible_agents else [f"agent_{i}" for i in range(6)]
        if flatten_for_fc:
            flat_dim = MAX_AGENTS * NODE_FEAT_DIM + MAX_AGENTS
            self.observation_spaces = {aid: Box(low=-1, high=1, shape=(flat_dim,), dtype=np.float32) for aid in default_agents}
        else:
            self.observation_spaces = {aid: self._graph_obs_space for aid in default_agents}
        
        # Get action spaces from wrapped env
        wrapped_action_spaces = getattr(env, "action_spaces", None)
        if wrapped_action_spaces is None:
            # Try method-based access
            if hasattr(env, "action_space") and callable(env.action_space):
                self.action_spaces = {aid: env.action_space(aid) for aid in default_agents}
            else:
                # Fallback: get from wrapped env's action_space dict or single space
                if hasattr(env, "action_space") and isinstance(env.action_space, dict):
                    self.action_spaces = {aid: env.action_space.get(aid, list(env.action_space.values())[0]) for aid in default_agents}
                else:
                    single_space = getattr(env, "action_space", None)
                    if single_space is None:
                        single_space = getattr(env, "action_spaces", None)
                    if single_space is None:
                        # Last resort: try to get from par_env
                        par_action = getattr(self.par_env, "action_space", None)
                        if par_action is None:
                            par_action = getattr(self.par_env, "action_spaces", None)
                        if par_action and callable(par_action):
                            self.action_spaces = {aid: par_action(aid) for aid in default_agents}
                        elif isinstance(par_action, dict):
                            self.action_spaces = {aid: par_action.get(aid, list(par_action.values())[0]) for aid in default_agents}
                        else:
                            self.action_spaces = {aid: par_action for aid in default_agents} if par_action else {}
                    else:
                        self.action_spaces = {aid: single_space for aid in default_agents}
        elif isinstance(wrapped_action_spaces, dict):
            self.action_spaces = {aid: wrapped_action_spaces.get(aid, list(wrapped_action_spaces.values())[0]) for aid in default_agents}
        else:
            self.action_spaces = {aid: wrapped_action_spaces for aid in default_agents}
        
        # PettingZoo expects observation_space as a method, not a dict
        # Store as method for compatibility with ParallelPettingZooWrapper
        def observation_space(agent_id: str):
            if agent_id in self.observation_spaces:
                return self.observation_spaces[agent_id]
            # Fallback: return first space or create default
            if self.observation_spaces:
                return list(self.observation_spaces.values())[0]
            # Create default if empty
            flat_dim = MAX_AGENTS * NODE_FEAT_DIM + MAX_AGENTS if self.flatten_for_fc else self._graph_obs_space
            if self.flatten_for_fc:
                return Box(low=-1, high=1, shape=(flat_dim,), dtype=np.float32)
            return self._graph_obs_space
        self.observation_space = observation_space
        
        def action_space(agent_id: str):
            if agent_id in self.action_spaces:
                return self.action_spaces[agent_id]
            if self.action_spaces:
                return list(self.action_spaces.values())[0]
            # Fallback: get from wrapped env
            if hasattr(self.env, "action_space") and callable(self.env.action_space):
                return self.env.action_space(agent_id)
            return getattr(self.env, "action_space", None) or getattr(self.env, "action_spaces", {}).get(agent_id)
        self.action_space = action_space

    def _obs_to_graph(self, agent_id: str, vec_obs, info: dict) -> dict:
        """Convert vector obs + info to graph format. Includes self_node_idx for GNN (agent_0->0, agent_1->1, ...)."""
        env = self.par_env
        global_state = info.get("global_state", {})
        disabled = info.get("disabled_agents", np.zeros(MAX_AGENTS, dtype=bool))
        env_size = getattr(env, "env_size", np.array([160, 80]))
        max_speed = max(getattr(env, "max_speeds", [2.0]))
        node_features = _state_to_node_features(env, agent_id, global_state, disabled, env_size, max_speed)
        edge_index = _build_edge_index(MAX_AGENTS)
        mask = 1.0 - np.array([float(disabled[i]) if i < len(disabled) else 1.0 for i in range(MAX_AGENTS)], dtype=np.float32)
        # Index of the acting agent in the node list (agent_0 -> 0, agent_1 -> 1, ...)
        try:
            self_node_idx = int(next(i for i, aid in enumerate(env.players) if aid == agent_id))
        except StopIteration:
            self_node_idx = 0
        self_node_idx = np.clip(self_node_idx, 0, MAX_AGENTS - 1).astype(np.int64)
        return {"node_features": node_features, "edge_index": edge_index, "mask": mask, "self_node_idx": np.array([self_node_idx], dtype=np.int64)}

    def _graph_to_flat(self, g: dict) -> np.ndarray:
        """Flatten graph for FC policy."""
        nf = g["node_features"].flatten()
        m = g["mask"]
        flat = np.concatenate([nf, m]).astype(np.float32)
        # Ensure all values are in [-1, 1] (node features should already be, but clip mask too)
        flat = np.clip(flat, -1.0, 1.0)
        return flat

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Update agents and spaces - always refresh from actual agents
        self.agents = list(obs.keys())
        self.possible_agents = self.agents.copy()
        
        # Rebuild observation_spaces for actual agents
        if self.flatten_for_fc:
            flat_dim = MAX_AGENTS * NODE_FEAT_DIM + MAX_AGENTS
            self.observation_spaces = {aid: Box(low=-1, high=1, shape=(flat_dim,), dtype=np.float32) for aid in self.possible_agents}
        else:
            self.observation_spaces = {aid: self._graph_obs_space for aid in self.possible_agents}
        
        # Update methods for PettingZoo compatibility (must be callable)
        def observation_space(agent_id: str):
            return self.observation_spaces.get(agent_id, list(self.observation_spaces.values())[0])
        self.observation_space = observation_space
        
        def action_space(agent_id: str):
            return self.action_spaces.get(agent_id, list(self.action_spaces.values())[0])
        self.action_space = action_space
        
        # Always update action_spaces from wrapped env
        wrapped_action_spaces = getattr(self.env, "action_spaces", None)
        if wrapped_action_spaces is None:
            wrapped_action_spaces = getattr(self.env, "action_space", None)
        if wrapped_action_spaces is None:
            # Try method-based access
            if hasattr(self.env, "action_space") and callable(self.env.action_space):
                self.action_spaces = {aid: self.env.action_space(aid) for aid in self.possible_agents}
            else:
                # Last resort: use first agent's space from wrapped env
                first_aid = self.possible_agents[0] if self.possible_agents else "agent_0"
                if hasattr(self.env, "action_space") and isinstance(self.env.action_space, dict):
                    first_space = self.env.action_space.get(first_aid, list(self.env.action_space.values())[0])
                else:
                    first_space = getattr(self.env, "action_space", None)
                self.action_spaces = {aid: first_space for aid in self.possible_agents} if first_space else {}
        elif isinstance(wrapped_action_spaces, dict):
            # Use wrapped env's action spaces
            self.action_spaces = {aid: wrapped_action_spaces.get(aid, list(wrapped_action_spaces.values())[0]) for aid in self.possible_agents}
        else:
            # Single space for all agents
            self.action_spaces = {aid: wrapped_action_spaces for aid in self.possible_agents}
        
        # Update methods for PettingZoo compatibility
        def observation_space(agent_id: str):
            return self.observation_spaces.get(agent_id, list(self.observation_spaces.values())[0])
        self.observation_space = observation_space
        
        def action_space(agent_id: str):
            return self.action_spaces.get(agent_id, list(self.action_spaces.values())[0])
        self.action_space = action_space
            
        out_obs = {}
        for aid in obs:
            g = self._obs_to_graph(aid, obs[aid], info.get(aid, {}))
            out_obs[aid] = self._graph_to_flat(g) if self.flatten_for_fc else g
        return out_obs, info

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        out_obs = {}
        for aid in obs:
            g = self._obs_to_graph(aid, obs[aid], info.get(aid, {}))
            out_obs[aid] = self._graph_to_flat(g) if self.flatten_for_fc else g
        return out_obs, rewards, term, trunc, info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()
