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
python rl_test/deploy_dynamic.py --checkpoint ./ray_dynamic/iter_360 --red-dummy --render
```

Adjust `--checkpoint` to your latest or chosen checkpoint.

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
| **Red opponent** | random | Use `--red-dummy` or `--red-heuristic` to change. |

---

## 3. Options and toggles

| Flag | Default | Description |
|------|---------|-------------|
| `--resume` | — | Resume from checkpoint (e.g. `./ray_dynamic/iter_360`). |
| `--red-dummy` | False | No Red opponents; Blue plays alone (capture the flag). |
| `--red-heuristic` | False | Red uses built-in heuristic (cannot combine with `--red-dummy`). |
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

## 4. Expected training: continue vs dummy from checkpoint 360

To keep training your Blue policy against dummy (no Red) from checkpoint 360 or your latest checkpoint:

1. **Use your latest checkpoint**  
   For example: `./ray_dynamic/iter_360` (or `iter_372`, etc.).

2. **Run training with resume + dummy:**

   ```bash
   python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_360 --red-dummy
   ```

   This restores Blue from the checkpoint and runs with no Red agents. Training continues from the next iteration (e.g. 361).

3. **Optional:** add `--render` to watch one game (window will briefly freeze each iteration), or `--max-time 900` for longer episodes.

4. **Optional:** save more often, e.g. `--save-every 6`.

Example with all of the above:

```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_360 --red-dummy --max-time 900 --save-every 6
```

When you want to train against moving Red again, resume the same way but switch to heuristic instead of dummy:

```bash
python rl_test/train_dynamic.py --resume ./ray_dynamic/iter_400 --red-heuristic --red-heuristic-mode easy
```

---

## 5. File reference

| File | Purpose |
|------|---------|
| `rl_test/train_dynamic.py` | Training script (GNN policy, graph obs). |
| `rl_test/deploy_dynamic.py` | Run a saved policy (watch only). |
| `pyquaticus/envs/dynamic_pyquaticus.py` | Dynamic env (variable teams, dummy mode). |
| `pyquaticus/envs/graph_obs_wrapper.py` | Graph observation wrapper. |
| `pyquaticus/models/gnn_model.py` | GNN model. |
