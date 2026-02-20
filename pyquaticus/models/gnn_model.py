# SPDX-License-Identifier: BSD-3-Clause
"""
GNN policy model for graph observations.

Processes graph (node_features, edge_index, mask, self_node_idx) with message passing,
then uses the self node's embedding (after message passing) for action logits and value.

Agent embedding flow:
  1. Node embedding: raw node features (B, N, F) -> linear+ReLU+linear -> (B, N, H).
  2. Message passing: for each edge (src,dst), message = ReLU(Linear([h_src, h_dst]));
     each node aggregates incoming messages by mean; then residual: h_new = h_old + agg.
  3. Mask: disabled nodes get embedding zeroed (h *= mask).
  4. Self embedding: for the acting agent, take the node at self_node_idx -> vector (B, H).
  5. Policy: self_emb -> MLP -> action logits. Value: self_emb -> MLP -> scalar.
  So each agent's action is decided only from its own node's embedding (which already
  encodes neighborhood info from message passing).
"""

import numpy as np
import torch
import torch.nn as nn

from pyquaticus.envs.graph_obs_wrapper import MAX_AGENTS, NODE_FEAT_DIM

try:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.annotations import override
    from ray.rllib.utils.framework import try_import_torch
    _HAS_RAY = True
except ImportError:
    TorchModelV2 = object
    override = lambda f: f
    try_import_torch = lambda: (None, None)
    _HAS_RAY = False

torch, _ = try_import_torch()

# Flattened size when Dict is flattened (key order: node_features, edge_index, mask, self_node_idx)
_NUM_EDGES = MAX_AGENTS * (MAX_AGENTS - 1)
_FLAT_NODE = MAX_AGENTS * NODE_FEAT_DIM
_FLAT_EDGE = 2 * _NUM_EDGES
_FLAT_MASK = MAX_AGENTS
_FLAT_SELF = 1
_FLAT_TOTAL = _FLAT_NODE + _FLAT_EDGE + _FLAT_MASK + _FLAT_SELF


def _scatter_mean(src, index, dim_size):
    """Aggregate messages to nodes (mean). src (E, F), index (E,) -> (dim_size, F)."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    index = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, index, src)
    count.scatter_add_(0, index[:, :1], torch.ones_like(src[:, :1]))
    count = count.clamp(min=1)
    return out / count


class GNNModel(TorchModelV2, nn.Module):
    """
    Graph Neural Network that:
    1. Embeds node features
    2. Runs message passing (mean aggregation over edges)
    3. Selects the self node's embedding
    4. Policy head: self_embedding -> action logits
    5. Value head: self_embedding -> value
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)
        self.num_outputs = num_outputs
        self._value_out = None

        hidden = int(model_config.get("custom_model_config", {}).get("gnn_hidden", 64))
        num_layers = int(model_config.get("custom_model_config", {}).get("gnn_layers", 2))

        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(NODE_FEAT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Linear(hidden * 2, hidden) for _ in range(num_layers)
        ])
        # Policy head (from self embedding)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_outputs),
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _unflatten_obs(self, obs):
        """Unflatten preprocessor output: order is node_features, edge_index, mask, self_node_idx (Dict order)."""
        B = obs.shape[0]
        o = obs.view(B, -1)
        i = 0
        nf_flat = o[:, i : i + _FLAT_NODE]
        i += _FLAT_NODE
        edge_flat = o[:, i : i + _FLAT_EDGE]
        i += _FLAT_EDGE
        mask = o[:, i : i + _FLAT_MASK]
        i += _FLAT_MASK
        self_idx = o[:, i : i + _FLAT_SELF].long()
        node_features = nf_flat.reshape(B, MAX_AGENTS, NODE_FEAT_DIM)
        edge_index = edge_flat.reshape(B, 2, _NUM_EDGES)
        return node_features, edge_index, mask, self_idx.clamp(0, MAX_AGENTS - 1)

    @override(TorchModelV2)
    def forward(self, input_dict, state_batches, seq_lens):
        obs = input_dict["obs"]
        if isinstance(obs, dict):
            node_features = obs["node_features"]
            edge_index = obs["edge_index"]
            mask = obs["mask"]
            self_node_idx = obs["self_node_idx"]
            if isinstance(node_features, np.ndarray):
                node_features = torch.from_numpy(node_features).float().to(next(self.parameters()).device)
            if isinstance(edge_index, np.ndarray):
                edge_index = torch.from_numpy(edge_index).long().to(node_features.device)
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(node_features.device)
            if isinstance(self_node_idx, np.ndarray):
                self_node_idx = torch.from_numpy(self_node_idx).long().to(node_features.device)
        else:
            node_features, edge_index, mask, self_node_idx = self._unflatten_obs(obs)

        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        B = node_features.shape[0]
        device = node_features.device

        # Embed nodes: (B, N, F) -> (B, N, H)
        x = self.node_embed(node_features)
        H = x.size(-1)

        # Edge index: (2, E) or (B, 2, E); gather() requires int64 indices
        if edge_index.dim() == 2:
            edge_index = edge_index.unsqueeze(0).expand(B, -1, -1)
        edge_index = edge_index.long()
        src_idx = edge_index[:, 0, :]   # (B, E)
        dst_idx = edge_index[:, 1, :]   # (B, E)

        # Message passing (vectorized aggregation per batch item)
        for layer in self.message_layers:
            src_feat = torch.gather(x, 1, src_idx.unsqueeze(-1).expand(-1, -1, H))
            dst_feat = torch.gather(x, 1, dst_idx.unsqueeze(-1).expand(-1, -1, H))
            msg = torch.relu(layer(torch.cat([src_feat, dst_feat], dim=-1)))
            agg = torch.zeros(B, MAX_AGENTS, H, device=device, dtype=x.dtype)
            for b in range(B):
                for n in range(MAX_AGENTS):
                    sel = dst_idx[b] == n
                    if sel.any():
                        agg[b, n] = msg[b, sel].mean(dim=0)
            x = x + agg

        # Mask out disabled nodes (optional: zero their embedding)
        mask_exp = mask.unsqueeze(-1)
        x = x * mask_exp

        # Self embedding: for each batch item, take node at self_node_idx (int64 for indexing)
        self_idx = self_node_idx.view(B).long()
        if self_idx.dim() == 2:
            self_idx = self_idx.squeeze(-1)
        self_emb = x[torch.arange(B, device=device), self_idx.clamp(0, MAX_AGENTS - 1)]  # (B, H)

        logits = self.policy_head(self_emb)
        self._value_out = self.value_head(self_emb).squeeze(-1)
        return logits, state_batches

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out
