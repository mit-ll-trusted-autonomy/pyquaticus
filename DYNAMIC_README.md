# Dynamic PyQuaticus & Graph Observations

This document describes the dynamic environment extensions and how to run/train them.

## What Was Added

### 1. `DynamicPyQuaticusEnv` (`pyquaticus/envs/dynamic_pyquaticus.py`)

- **Variable team sizes (1–3 per team)** at each reset
- **`disabled_agents`** state for inactive agents
- **Mid-game removal**: `tag_removes_agent=True` disables agents when tagged
- **Mid-game spawning**: `reinforcement_interval` and `reinforcement_prob` for reinforcements
- **`is_disabled`** in observations so policies can ignore inactive agents

### 2. `GraphObsWrapper` (`pyquaticus/envs/graph_obs_wrapper.py`)

- Converts vector observations to a graph format:
  - **node_features**: (6, 10) per-agent features (pos, heading, speed, has_flag, is_tagged, etc.)
  - **edge_index**: all-to-all edges between agents
  - **mask**: (6,) indicating active agents
- **Flatten mode**: concatenates to a fixed-size vector for standard RLLib policies

### 3. Training script (`rl_test/train_dynamic.py`)

- Trains Blue team (agents 0–2) vs random Red (agents 3–5)
- Uses graph observations by default
- Supports vector observations with `--no-graph`

---

## Setup

Use the same environment as the base PyQuaticus project:

```bash
# From pyquaticus root
conda activate env-full   # or your conda env with ray, torch, etc.
# OR
pip install -e .[torch,ray]
```

---

## Run the Dynamic Environment

### Quick test (no training)

```bash
cd c:\Users\Joeyt\Documents\GMU\CS_491\pyquaticus
python test/test_dynamic_env.py
```

### Manual run with rendering

```python
from pyquaticus.envs.dynamic_pyquaticus import DynamicPyQuaticusEnv
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std

cfg = config_dict_std.copy()
cfg["max_score"] = 3
cfg["max_time"] = 120
reward_config = {f"agent_{i}": rew.caps_and_grabs for i in range(6)}

env = DynamicPyQuaticusEnv(
    team_size_range=(1, 3),
    tag_removes_agent=False,
    reinforcement_interval=100,
    reinforcement_prob=0.3,
    config_dict=cfg,
    reward_config=reward_config,
    render_mode="human",
)
obs, info = env.reset(seed=42)
print("Active blue:", info["agent_0"]["num_blue_active"])
print("Active red:", info["agent_0"]["num_red_active"])
# Step with no-ops
for _ in range(100):
    actions = {aid: 16 for aid in env.agents}
    obs, rewards, term, trunc, info = env.step(actions)
env.close()
```

---

## Train with RLLib

Training uses **only the custom GNN policy**: graph observations and the GNN model (message passing, self-node embedding for policy/value).

```bash
python rl_test/train_dynamic.py
```

### With rendering

```bash
python rl_test/train_dynamic.py --render
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--render` | False | Render during training |
| `--iters` | 2000 | Training iterations |
| `--save-every` | 250 | Save checkpoint every N iters |
| `--out-dir` | `./ray_dynamic/` | Checkpoint directory |
| `--runners` | 20 | Number of parallel env runners (more = faster if you have CPUs) |
| `--speedup` | 4 | Sim speedup factor (e.g. 8 = env steps 2× faster; minimal impact on learning) |

### Speed (without sacrificing results)

- **Time per 50 iters**: The script prints `time=Xs/iter (est. ~Ys per 50 iters)`. Use that for planning.
- **Faster runs**: Use `--runners 32` (or 40) if you have spare CPUs, and `--speedup 8` to run the sim 2× faster. Same dynamics and learning; only wall-clock time changes.

### Checkpoints

Checkpoints are saved to `./ray_dynamic/iter_500/`, `iter_1000/`, etc. Load with:

```python
from ray.rllib.policy.policy import Policy
policy = Policy.from_checkpoint("./ray_dynamic/iter_1000/policies/blue_policy/")
```

---

## File Reference

| File | Purpose |
|------|---------|
| `pyquaticus/envs/dynamic_pyquaticus.py` | Dynamic env with variable teams and mid-game changes |
| `pyquaticus/envs/graph_obs_wrapper.py` | Graph observation wrapper |
| `pyquaticus/models/gnn_model.py` | GNN TorchModelV2 (message passing, self-node embedding) |
| `rl_test/train_dynamic.py` | RLLib training script |
| `rl_test/deploy_dynamic.py` | Run trained blue policy from checkpoint |
| `test/test_dynamic_env.py` | Basic tests |

---

## How agent embeddings work (GNN)

The GNN turns the graph into one embedding vector per agent, then uses only the **acting agent’s** embedding for its action and value:

1. **Node features** (from `GraphObsWrapper`): Each of the 6 nodes has 10 features (pos, heading, speed, has_flag, is_tagged, on_side, team, is_disabled, cooldown). Node order matches `env.players` (agent_0…agent_5).

2. **Node embedding**: A small MLP maps each node’s 10-D vector to a hidden vector (e.g. 64-D). So each agent is first represented by a single vector from its raw features.

3. **Message passing**: Edges are all-to-all (every agent sees every other). For each edge (source, target), the model computes a message from the concatenation of source and target embeddings, then each node **aggregates** incoming messages (mean) and adds the result to its own embedding (residual). This is repeated for `gnn_layers` (e.g. 2). So each node’s vector now encodes both itself and its neighbors.

4. **Mask**: Embeddings of disabled agents are zeroed so they don’t affect aggregation or the policy.

5. **Self-node selection**: The wrapper tells the model which node is “me” via `self_node_idx` (e.g. agent_0 → 0, agent_1 → 1). The model takes **only that node’s** embedding after message passing.

6. **Policy and value**: That single vector (the “agent embedding” for the acting agent) is passed through the policy head → action logits and the value head → value estimate. So each agent’s action is based on its own contextualized embedding, not on a global pool.

---

## Extending Further

- **GNN policy**: Training uses the GNN by default. See `pyquaticus/models/gnn_model.py`.
- **Larger teams**: Change `team_size_range` and `MAX_AGENTS` in the graph wrapper.
- **Custom rewards**: Edit `reward_config` in `train_dynamic.py` or pass your own reward functions.
