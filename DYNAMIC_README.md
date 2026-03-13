# Dynamic PyQuaticus — Setup, Training & Deployment

## 1. Setup and run

### Setup

From the project root:

```bash
conda activate env-full
# or: pip install -e .[torch,ray]
```

### Training

**Start from scratch (Blue vs random Red):**

```bash
python rl_test/train_dynamic.py
```

**Continue from a checkpoint (e.g. iter_360) with dummy Red (no opponents):**

```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_360 --red-dummy
```

Use your latest checkpoint folder instead of `iter_360` if you have a newer one (e.g. `iter_372`).

**Training with the game window (one window; will briefly freeze each iteration):**

```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_360 --red-dummy --render
```

Checkpoints save to `./ray_dynamic/iter_N/`. Progress is logged to `./ray_dynamic/train.log`.

### Deployment / watch a policy

Run the trained Blue policy and watch it (no training):

```bash
python rl_test/deploy_dynamic.py ./ray_dynamic/iter_360 --red-dummy
```

Use your latest or chosen checkpoint path (e.g. `./ray_dynamic/iter_360`). Add `--no-render` to run without a window.

---

## 2. Defaults

Unless you pass flags, training uses:

| Setting | Default | Meaning |
|--------|---------|--------|
| **Training length** | 2000 iters | Run stops after this many iterations. |
| **Runners** | 8 | Parallel envs (0 when `--render`). |
| **Speedup** | 8 | Sim runs 8× real time. |
| **Episode time** | 600 s (10 min) | Max seconds per game. |
| **Episode score** | 3 | First team to 3 captures wins. |
| **Save every** | 12 iters | Checkpoint written every 12 iterations. |
| **Out directory** | `./ray_dynamic/` | Checkpoints and `train.log` go here. |
| **Train batch size** | 4000 steps | Env steps per PPO update (500 when `--render`). |
| **Red opponent** | random | Use `--red-dummy`, `--red-heuristic`, or `--red-from-checkpoint` (one only). |

---

## 3. Options and toggles

| Flag | Default | Description |
|------|---------|-------------|
| `--resume` | — | Resume from checkpoint (e.g. `./ray_dynamic/iter_360`). |
| `--red-dummy` | False | No Red opponents; Blue plays alone (capture the flag). |
| `--red-heuristic` | False | Red uses built-in heuristic. Use only one of: `--red-dummy`, `--red-heuristic`, `--red-from-checkpoint`. |
| `--red-from-checkpoint` | — | Red uses Blue policy from this checkpoint (self-play vs previous iteration). Path: e.g. `./ray_dynamic/iter_300`. |
| `--red-heuristic-mode` | easy | If `--red-heuristic`: `easy`, `medium`, or `hard`. |
| `--render` | False | Show one game window while training (short freezes each iteration). |
| `--iters` | 2000 | Max training iterations. |
| `--save-every` | 12 | Save checkpoint every N iterations. |
| `--out-dir` | `./ray_dynamic/` | Where checkpoints and `train.log` are written. |
| `--runners` | 8 | Parallel env runners (no effect when `--render`). |
| `--speedup` | 8 | Sim speedup factor. |
| `--max-time` | 600 | Max episode time in seconds. |
| `--max-score` | 3 | Score limit per team to end episode. |
| `--no-log-file` | False | Do not write to `out_dir/train.log`. |

**Save a checkpoint on demand:** create an empty file `SAVE_NOW` in `out_dir`; the next finished iteration will save and then delete it.

---

## 4. Training progression (3 phases)

Recommended order: train Blue to capture (dummy), then vs a simple opponent (heuristic), then vs random Red for diversity. Use `--resume ./ray_dynamic/iter_N` to continue from a checkpoint.

### Phase 1: Dummy — learn to capture

No Red opponents; Blue learns to reach and capture the flag. Use longer episodes so they have time to reach the flag.

```bash
python rl_test/train_dynamic.py --red-dummy --max-time 600 --max-score 3 --save-every 12
```

Run until Blue is reliably capturing (e.g. 100–300 iters). Checkpoints go to `./ray_dynamic/iter_N`. Pick one (e.g. `iter_200` or `iter_300`) for the next phase.

### Phase 2: Heuristic Red — learn vs a simple opponent

Switch to heuristic Red and resume from your best dummy checkpoint. Start easy, then increase difficulty.

```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_300 --red-heuristic --red-heuristic-mode easy --save-every 12
```

When performance looks good, make Red harder and continue:

```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_500 --red-heuristic --red-heuristic-mode medium --save-every 12
```

Use `--red-heuristic-mode hard` when ready.

### Phase 3: Self-play (vs previous iteration)

Train Blue against an older copy of itself. Red uses the Blue policy from a past checkpoint (e.g. a few save intervals behind). Resume from your latest checkpoint and pass the older one as `--red-from-checkpoint`.

```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_700 --red-from-checkpoint ./ray_dynamic/iter_688
```

You can also train from scratch with Red fixed to a checkpoint: `--red-from-checkpoint ./ray_dynamic/iter_500` (no `--resume`).

**Alternative — Random Red:** Omit Red flags to use random Red for diversity: `python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_700`

### Quick reference

| Phase      | Goal              | Command |
|-----------|-------------------|--------|
| **Dummy** | Learn to capture  | `--red-dummy --max-time 600 --max-score 3` |
| **Heuristic** | Learn vs opponent | `--resume ./ray_dynamic/iter_N --red-heuristic --red-heuristic-mode easy` (then `medium` / `hard`) |
| **Self-play** | Vs previous iteration | `--resume ./ray_dynamic/iter_N --red-from-checkpoint ./ray_dynamic/iter_M` (M < N) |
| **Random** | Generalize        | `--resume ./ray_dynamic/iter_N` (no Red flags) |

---

## 5. File reference

| File | Purpose |
|------|---------|
| `rl_test/train_dynamic.py` | Training script (GNN policy, graph obs). |
| `rl_test/deploy_dynamic.py` | Run a saved policy (watch only). |
| `pyquaticus/envs/dynamic_pyquaticus.py` | Dynamic env (variable teams, dummy mode). |
| `pyquaticus/envs/graph_obs_wrapper.py` | Graph observation wrapper. |
| `pyquaticus/models/gnn_model.py` | GNN model. |
