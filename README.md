# OligoRL: Firm-Level Reinforcement Learning in a Simulated Oligopoly with Product and Labor Competition

## Overview
**OligoRL: Firm-Level Reinforcement Learning in a Simulated Oligopoly with Product and Labor Competition** is a seeded, reproducible environment for training and evaluating reinforcement learning (RL) agents. One player firm and four NPC firms form an oligopoly that competes on two fronts. In the **product market**, firms set prices and compete for customers. In the **labor market**, they set wages and compete for a fixed pool of workers. Macroeconomic feedback loops and capacity constraints tie the two together.

OligoRL includes an interactive dashboard (play the firm yourself, or watch agents compete on a leaderboard), scripted baseline agents, and a PPO agent trained with a pre-registered evaluation protocol.

The environment tracks microeconomic variables (revenue, operating costs, wages, capital expenditure) alongside macroeconomic indicators (real GDP, price index, Gini coefficient). A rule-based reward engine scores each step and explains its score in plain text. The PPO experiment below documents the exploits an RL agent found in earlier versions of the rules, and how they were addressed.

## Architecture design

The system is cleanly separated into layered modules:

### 1. Configuration (`config.py`)
Every constant and design decision in one place, with the reasoning written inline: market and production parameters, the labor market (section 7b), the training-solvency score (7c), experiment NPC profiles (7d), the PPO reward (7e) and the v2 rules (7f).

### 2. Reward Engine (`rewards.py`)
Pure functions over plain numbers:
* **Profit Score:** inverse hyperbolic sine (IHS) of net income, normalized by a revenue moving average, so growing volume is not punished.
* **Solvency Score:** sigmoid of wage runway (cash ÷ wage bill, capped), multiplied by an activity factor (units sold ÷ baseline output), so a firm that stops producing is not scored as "safe".
* **GDP Score:** relative, scale-invariant change in real GDP.
* **Gini Penalty:** penalty on wealth inequality above a threshold.
* **Net-worth reward:** the PPO training reward (see the PPO section).

### 3. Core State Machine (`simulator.py`)
A single-threaded `EconSimulator` that runs each timestep in a fixed order:
1. Validate the player's action (price, wage, hiring, capex, lobbying); NPCs set prices and wages.
2. Labor market: firing, quits from underpaying firms, hiring from the unemployed pool, poaching (NPC headcount follows demand under the default v2 rules).
3. Calculate the wealth-driven demand multiplier.
4. Distribute market share and revenue, limited by each firm's production capacity.
5. Pay wages, inputs, tax, hiring and capex; workers spend part of their income. Every dollar is accounted for (checked by the test suite), but total wealth is not constant: spending, taxes and costs leave the economy.
6. Compute real GDP using the previous step's price index, then update the index.
7. End the episode on bankruptcy or after the last step.

### 4. Reward Integration (`reward_integrator.py`)
Computes weighted composite reward mapping RL policies to intended outcomes based on Difficulty Presets. Critically, it generates a real-time, rule-based Semantic Feedback string (causal reasoning text) outlining exactly *why* a numerical reward was assigned (e.g., pricing relative to competitors, capacity utilization, hiring efficiency).

### 5. Network Layer (`api.py`)
A thin FastAPI and Pydantic wrapper with schema validation. Each browser session (`X-Session-Id` header) gets its own simulator, guarded by a lock; it also serves the dashboard.

### 6. Agents and Leaderboard (`agents.py`)
Scripted agents (Rule-based, Hold, Market + 1%, Premium pricer, Minimum wage, Maximum wage, Poacher, Random) and the trained PPO agent. Leaderboards rank agents by final net worth (cash + capital, −$10k if bankrupt) over shared seeds, with 95% confidence intervals.

### 7. Dashboard (`static/`)
A browser dashboard served at `/`:
* **Play tab:** run the firm yourself (price, wage, hiring, capex, lobbying), with live charts, the labor market and the reward commentary.
* **Watch tab:** the agent leaderboard, net-worth curves and step-by-step replays of any agent, including PPO.

### 8. PPO Experiment (`ppo/`)
Observation and action spaces, the gymnasium environment, training, numpy export, and the evaluation and analysis scripts. Results live in `results/ppo/`, and the exported model in `models/`.

### 9. Tests
`test_*.py` check the simulator, labor market, rewards, API and dashboard endpoints. `test_golden.py` checks that both rule sets (legacy and v2) reproduce stored trajectories exactly.

## Difficulty Presets

Each preset sets the composite reward weights (positive weights sum to 1; the Gini penalty is always −0.10) and the episode length:
* **Easy (30 steps):** solvency 0.78, profit 0.11, GDP 0.11. Mostly rewards survival.
* **Medium (30 steps):** profit 0.67, solvency 0.22, GDP 0.11. Mostly rewards profit.
* **Hard (60 steps):** profit 0.44, GDP 0.33, solvency 0.22. Longer episodes with more weight on economy-wide GDP growth.

The PPO agent does not train on these weights; it uses the net-worth reward described below.

## API Documentation

### POST `/reset`
Re-initializes the session's simulation. The same seed always reproduces the same trajectory.
**Payload:** `{"difficulty": "medium", "seed": 42}` 

### POST `/step`
Advances the simulation by one timestep and returns the observation, the composite reward with its feedback text, and a per-step breakdown.
**Payload:** `{"set_price": 10.5, "set_wage": 520, "hire_workers": 2, "capex": 500, "lobby_tax_change": 0.0}` (`set_wage` is optional; the wage moves at most 10% per step toward it)

### GET `/state`
Retrieves the current observation without advancing time.

### Other endpoints
* `GET /` serves the dashboard; `GET /health` is a health check.
* `GET /meta` returns action bounds, defaults and difficulty presets.
* `GET /agents` lists the built-in agents (scripted agents and the trained PPO agent).
* `POST /compare` runs a leaderboard: every agent plays the same seeds, ranked by final net worth with a 95% confidence interval.
* `POST /simulate` returns a full replay of one agent's episode.

## Evaluation & Verification

The test files check the simulator's invariants and behavior, including:
* Money accounting: each step's change in total wealth equals revenue minus every cost and outflow.
* Supply caps: a firm never sells more than its production capacity.
* Labor market: the number of workers is constant, and every worker move is recorded.
* Golden tests: the legacy and v2 rules reproduce stored trajectories exactly.
* Grader alignment: on the scenarios tested (legacy rules), the composite reward ranks strategies in the same order as final net worth.

1 check in `test_simulator.py` fails and predates this work (see Limitations below).

## Setup Instructions

### Run locally

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the tests (the API tests also need `pip install httpx`):
   `python test_api.py` (and the other `test_*.py` files)
3. Launch the API and dashboard:
   `uvicorn api:app --host 0.0.0.0 --port 7860`, then open http://localhost:7860

`requirements.txt` is everything the API, dashboard and trained PPO agent need
(the PPO model runs on numpy alone). Training needs `requirements-train.txt`
as well (torch, Stable-Baselines3, gymnasium, matplotlib).

**Verification:** the app was verified on Python 3.10 (WSL Ubuntu) with only
`requirements.txt` installed and no torch loaded. The API server starts, the
dashboard, leaderboard and replay endpoints respond, and the leaderboard
reproduces PPO's $39,473.44 ± $1,345.99 exactly (as on the Windows development
machine). The test suite was run on the development machine. `test_api.py` and
`test_dashboard_api.py` additionally need `httpx` for FastAPI's test client.
No Docker build was performed, and no Hugging Face Space deployment was done.

## Reinforcement learning: PPO agent

A PPO agent (Stable-Baselines3, in-process simulator, Medium difficulty) is
trained to run the firm and is evaluated against every scripted agent. The full
protocol was pre-registered before any test seed was used:
`results/ppo/preregistration.md`. Deviations, with dates:
`results/ppo/deviations.md`. Final numbers: `results/ppo/final/`.

**Final configuration:** simulator rules **v2** (`npc_growth=True`, smooth
quits) with **spillover 0** (`SPILLOVER_FRACTION = 0.0`). These are the
simulator defaults, used for the PPO experiment, the leaderboard and the
dashboard. Lobbying is disabled for every agent in the PPO experiment.

### Reward
The training reward is the leaderboard metric made dense:
`r_t = (net_worth_t − net_worth_{t−1}) / $10,000 + hinge_t`, with γ = 1, so an
episode's return is its final net-worth change (bankruptcy costs an extra
$10,000) plus a small solvency hinge. The hinge is zero whenever cash covers one
step of wages, and its worst-case total is half the bankruptcy penalty, so it
can never outweigh going bankrupt. Return and final net worth correlate at 1.000
on test. The older composite reward is still computed and shown in the
dashboard as "Composite reward (legacy)".

### Observation and action spaces (`ppo/spaces.py`)
- **Observation:** 14 public features, normalized with fixed constants inside the
  environment: cash and net worth vs starting cash, wage ÷ market wage, price ÷
  rivals' average, progress, workforce and unemployment shares, last demand and
  sales ÷ capacity, market wage and rivals' price vs defaults, tax rate, workers
  lost, payroll runway. No NPC internals.
- **Actions:** price = rivals' average × 2^a (range 0.5×–2×, log scale);
  wage = market × [0.9, 1.2]; hire/fire = a bounded share of headcount, with a
  dead zone around zero and at least one worker outside it. All are squashed with
  tanh inside the environment, and the zero action means "rival price, market
  wage, no change". Lobbying and capex are not PPO actions, and lobbying is
  disabled for every agent in the experiment.

### Protocol
- Rules: **v2** (NPCs size their workforce to last step's demand, smooth quit
  curve) with **spillover off**. These are now the simulator defaults; the
  round-1 rules remain available with `npc_growth=False`, and `test_golden.py`
  checks that both reproduce exactly.
- Training NPCs are randomized per episode (`train` profile). A `heldout`
  profile lies outside the training ranges on both sides (price wars, overpaying
  and frozen-headcount NPCs).
- Disjoint seed sets: training ≥ 100,000; validation 50,000–50,199 (checkpoint
  selection); held-out validation 60,000–60,199; test 10,000–10,199 and held-out
  test 20,000–20,199, each used once.
- 5 training runs × 700k steps (the budget was set from a validation-only pilot).
  Each run keeps its best checkpoint by validation net worth.
- The benchmark (Rule-based) was frozen on validation. All comparisons are
  paired on the same seeds and NPC draws.
- The model is exported to `models/ppo_medium.npz` and inference uses numpy only
  (mean action). The largest torch-vs-numpy difference was ≤ 5.3 × 10⁻⁷ on
  1,000 observations.

### Results (test seeds)
Two runs are reported, each for a different purpose:
- **Headline = the median run (run 1)**, by default-test mean. All headline numbers
  below are from it.
- **Leaderboard = the best run by validation (run 4).** It is used in the dashboard
  leaderboard and Watch tab, labelled "PPO (trained)". It was chosen on
  validation, not test, and is not the headline.

| Headline run 1 | Default test (200 seeds) | Held-out test (200 seeds) |
|---|---|---|
| PPO | **$38,624** ± 517, 0% bankrupt | $44,554 ± 6,033, **12.5% bankrupt** |
| Rule-based (benchmark) | $33,783 | $43,724 |
| PPO − Rule-based (95% CI) | **+$4,841 [+4,363, +5,319]**, wins 92% | +$830 [−1,146, +2,807], **not significant** |

- **Default test (in distribution):** all 5 runs beat Random, Minimum wage,
  Maximum wage, Poacher, Hold and Market + 1% (paired CI lower bound > 0), and all
  5 beat the Rule-based benchmark (paired CI lower bound > 0). Default-test means
  range from $37,709 to $40,309 (std $1,043).
- **Held-out test: mixed.** All 5 runs beat the six naive baselines (H1), but the
  held-out criteria (H2) pass in only **3 of 5 runs**. The headline run 1 fails
  H2b (its lead over the benchmark shrinks by more than 10 points: 1.14× → 1.02×)
  and H2c (12.5% bankrupt, limit 5%). Run 3 fails H2c (10.5% bankrupt). Runs 2, 4
  and 5 pass with 0% bankruptcy. PPO's held-out edge over Rule-based is **not
  significant** for the headline run.
- **These results do not show that PPO generalizes** beyond the training
  distribution. It beats a strong hand-written agent in distribution; out of
  distribution, the outcome depends on the training run.

### Findings
1. **Spillover exploit (rounds 1 and 2).** With 40% of turned-away demand
   spilling to firms that have spare capacity, PPO learned to price about 22%
   above rivals and hold idle workers to harvest that overflow, reaching $107k
   against $68.5k for the best scripted agent (validation seeds). With spillover off, the same
   policy fell *below* that agent. NPC growth (v2) did not fix it. The cause is
   structural: demand is about 2.7× the economy's maximum output and does not
   respond to the price level. Spillover is therefore off by default.
2. **PPO still wins without spillover.** Under the legacy rules with spillover
   off, it beat the best scripted agent by +$7.0k ± 0.7k on every validation
   seed.
3. **The wage band edge.** Under the old quit rule, paying 3% below market cost
   nothing. The v2 smooth quit curve removed that, and profit now peaks at
   about 1.03× market.
4. **The endgame.** The policy requests hires in the last 3 steps in 99–100% of
   episodes. It costs $76–130 of net worth (about 0.3%). This was reported, and
   no endgame rule was hard-coded.

### Limitations
- **Price-war failure (out of distribution).** When three or more NPCs price at
  $5.50–6.50, below the training range of $6.50+, runs 1 and 3 follow rivals'
  prices down to about $6. At that price a worker earns only about its $300 wage,
  and the policy keeps hiring from a large unemployed pool, so it bleeds cash and
  goes bankrupt (around step 22). Rule-based survives every one of those seeds
  because it fires idle workers. This is the same "hire when workers are
  available" habit that caused the round-1 held-out collapse.
- With spillover off, a rival's capacity shortage no longer sends its customers
  anywhere, so firms are not linked through demand.
- The held-out profile is easier on average for adaptive agents: Rule-based
  earns 1.29× its default score there, because held-out NPCs shrink and free up
  workers and customers. It is harsh in its price-war episodes.
- NPC bankruptcies are rare in the training profile (about 1–2% of episodes vs
  15–18% held-out).
- Only one difficulty (Medium) is covered, and test results come from one seed
  range per profile.
- **Known failing test (pre-existing):** "Overpricing: survived >= 10 steps" in
  `test_simulator.py` also fails on the commit before this work (`6ce5578`).
  At a $50 price the firm sells almost nothing but still pays wages, so it
  goes bankrupt around step 5; the test's expectation is unrealistic, not the
  simulator. Three other checks that also failed before this work expected an
  older soft-cap formula for the demand share. They now test the formula the
  code actually uses (`simulator.compute_demand_share`).

### Reproduce
```bash
pip install -r requirements.txt -r requirements-train.txt   # torch, SB3, gymnasium, matplotlib
python -m ppo.benchmark                 # freeze benchmarks on validation
python -m ppo.train --run-seed 1 --val-heldout              # one training run (repeat for 2-5)
python -m ppo.evaluate                  # one-time test evaluation, export, figures
python test_golden.py                   # legacy + v2 rules unchanged
```
