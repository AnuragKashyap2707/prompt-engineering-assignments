# Adaptive Tutorial Agent (PPO + LinUCB)
Take-Home Final: **Reinforcement Learning for Agentic AI Systems**

This repo contains a notebook implementation of an adaptive tutorial agent that learns
to orchestrate *difficulty selection* (PPO) and *tool selection* (LinUCB). It logs
per-episode metrics, generates plots, and runs statistical tests against a heuristic
baseline with ablations (PPO-only, Bandit-only).

---

## Folder structure (created by the notebook)
```
artifacts/                 # plots and stats.json
experiments/
  runs/
    baseline_metrics.csv
    rl_metrics.csv
    ppo_only_metrics.csv
    bandit_only_metrics.csv
    eval_metrics.csv
```
> If these folders don’t exist yet, the notebook will create them on first run.

---

## Quickstart

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Launch Jupyter and open the notebook
```bash
jupyter notebook
```
Open the provided notebook (e.g., `Untitled4.ipynb`) and follow the run order below.

---

## **Exact run order (cells to execute in sequence)**

> If your notebook has headings, the cell titles below match them.

1. **Cell 3 — “Simple offline tutorial env.”**  
   Loads the simulator and **reward function** (time/hint penalties).  
   *Run this first so all later code uses the correct reward.*

2. **Cell 7 — “PPO (net/adv/ppo loop)”**  
   Defines the Actor-Critic model, GAE, and PPO update loop.

3. **Cell 1.1 — “Stronger PPO for long training”**  
   Sets PPO hyperparameters (γ=0.99, clip=0.2, λ=0.95, entropy=0.005, epochs=6, lr=1.5e-4).

4. **Cell 8 — “Controller Agent”**  
   Wires PPO (difficulty) + LinUCB (tool selection).

5. **Cell 8.2 — “Specialized agents + explicit comms protocol”**  
   Ensures `ControllerAgent.run_step` returns a **4‑tuple** (`decision, out, err, teach_msg`) that
   the trainer expects. **Run this before training** to avoid unpack errors.

6. **Cell 11 — “Run training & ablations”**  
   Runs:
   - Baseline (1000 episodes)
   - RL (PPO + Bandit) (1000 episodes)
   - PPO-only (1000 episodes)
   - Bandit-only (1000 episodes)

   Saves CSVs into `experiments/runs/`.

7. **Cell 12 — “Plots”**  
   Saves learning curves to `artifacts/` (reward/success with EMA).

8. **Cell 13 — “Bootstrap stats”**  
   Computes one‑sided bootstrap test for **RL > Baseline** and writes `artifacts/stats.json`.

9. **Cell 14 — “Evaluation run”**  
   Runs held‑out evaluation (e.g., 150 eps) and writes `experiments/runs/eval_metrics.csv`.

> **Tip:** If you have already generated `baseline_metrics.csv`, you can modify Cell 11 to skip
recomputing baseline and only run RL to save time.

---

## What “good” looks like (for grading)
- `rl_metrics.csv` shows higher mean reward than `baseline_metrics.csv` (Δ > 0).
- `artifacts/stats.json` contains a **small p-value** (e.g., `< 0.05`) for the one‑sided test RL > baseline.
- Plots in `artifacts/` show rising EMA for reward/success.
- Ablations (PPO-only, Bandit-only) underperform the combined RL agent.

---

## Reproducibility & knobs
- **Episodes/Horizon:** look for variables like `RL_EPISODES`, `BASELINE_EPISODES`, and `HORIZON` near the training cell.
- **PPO exploration:** adjust `entropy_coef` in Cell 1.1 (e.g., 0.005).
- **Reward shaping:** edit the reward line in Cell 3 (e.g., `- 0.05 * t` for time cost).

---

## Troubleshooting
- **ValueError: not enough values to unpack (expected 4, got 3):**  
  Re-run **Cell 8.2** *before* training so `run_step` returns 4 values.
- **ModuleNotFoundError (e.g., tqdm):**  
  Run `pip install -r requirements.txt` again, or `pip install tqdm`.
- **Slow training:** reduce episodes or horizon; keep baseline CSV and re-run only RL.

---

## License
For academic use in the course. Adapt as needed with attribution.
