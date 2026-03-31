# OpenEnv Simulation Framework

## Overview
OpenEnv is a highly deterministic, strictly verified macroeconomic environment designed for Reinforcement Learning (RL) agent training and Large Language Model (LLM) economic reasoning evaluation. It implements a multi-agent competitive market simulation with dynamic macroeconomic feedback loops, discrete capacity constraints, and a strongly typed, deterministic reward engine.

The environment tracks granular microeconomic variables (revenue, operational costs, capital expenditure) alongside macroeconomic indicators (real GDP, inflation/price index, Gini coefficient) and wraps them inside a strict mathematical grader designed to prevent reward hacking and volume exploitation.

## Architecture design

The system is cleanly separated into layered modules:

### 1. Mathematical Foundation (`config.py`)
Centralized variables defining all structural relationships, baseline demand thresholds, operational constants, and bounded clamping arrays designed to preserve monotonic gradients.

### 2. Reward Engine (`rewards.py`)
Pure mathematical functions capturing standard economic metrics:
* **Profit Score:** Inverse Hyperbolic Sine (IHS) normalized tracking of absolute profit to prevent margin-punishment during volume expansion.
* **Solvency Score:** Logistic sigmoid mapping of cash runway preventing early starvation.
* **GDP Score:** Relative, scale-invariant delta measurement.
* **Gini Penalty:** O(n log n) sorted-array metric punishing extreme structural inequality.

### 3. Core State Machine (`simulator.py`)
A single-threaded, highly structured `EconSimulator` strictly sequencing the following timeline per timestep:
1. Parse player and NPC actions.
2. Advance hiring and capital expenditures.
3. Calculate wealth-driven demand multipliers.
4. Distribute market share and capture revenue against discrete production constraints.
5. Reallocate expenses to worker and capital pools (conserving aggregate wealth).
6. Advance real GDP using lagged price indexing to prevent immediate inflation cancellation.
7. Terminate upon bankruptcy.

### 4. Reward Integration (`reward_integrator.py`)
Computes weighted composite reward mapping RL policies to intended outcomes based on Difficulty Presets. Critically, it generates a real-time, rule-based Semantic Feedback string (causal reasoning text) outlining exactly *why* a numerical reward was assigned (e.g., pricing relative to competitors, capacity utilization, hiring efficiency).

### 5. Network Layer (`api.py`)
A thin FastAPI and Pydantic wrapper exposing the environment to distributed deployment clusters securely, featuring inherent schema validation and thread safety constraints over the global simulation instance.

## Difficulty Presets

The simulator maps weights across components dynamically based on initialization presets:
* **Easy:** Heavily weights `solvency`. Intended to instruct baseline survival algorithms.
* **Medium:** Heavily weights `profit`. Requires agents to balance strategic underpricing for market share growth against variable labor/input costs.
* **Hard:** Heavily weights `gdp`. Forces agents to cooperatively manage aggregate economic wealth to spark macroeconomic feedback multipliers.

## API Documentation

### POST `/reset`
Re-initializes the simulation and applies random seeds securely for verifiable trajectories.
**Payload:** `{"difficulty": "medium", "seed": 42}` 

### POST `/step`
Advances the underlying state by one timestep returning dense rewards and semantic feedback.
**Payload:** `{"set_price": 10.5, "hire_workers": 2, "capex": 500, "lobby_tax_change": 0.0}`

### GET `/state`
Retrieves current observation parameters without advancing the environment time.

## Evaluation & Verification

The suite guarantees strict mathematical alignment using an automated testing framework comprising 230 explicit assertions verifying invariant rules:
* Money conservation checks (Aggregate system wealth remains stable across exchanges).
* Supply caps (Market captures cannot exceed installed capacity).
* Grader alignment (Dense reward sums strictly correlate with intended sparse metrics like firm survival durations and market dominance distributions).

## Setup Instructions

### Standard Installation

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the unit test verifications locally:
   `python test_api.py` (and adjacent test files)
3. Launch the API:
   `uvicorn api:app --host 0.0.0.0 --port 8000`

### Docker Deployment

1. Build the lightweight image:
   `docker build -t openenv .`
2. Run the container:
   `docker run -p 8000:8000 openenv`
