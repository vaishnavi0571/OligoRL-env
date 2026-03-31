"""
Phase 5 -- API Layer Tests
=============================
Tests the FastAPI wrapper around the OpenEnv simulator.

Run: python test_api.py
"""

import sys
from fastapi.testclient import TestClient

from api import app
from config import PRICE_FLOOR, MAX_HIRE_PER_STEP

client = TestClient(app)

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}]  {label}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)
    if condition:
        passed += 1
    else:
        failed += 1


def heading(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================
heading("1. Setup & Reset")
# ============================================================

# State before reset should fail
resp = client.get("/state")
check("GET /state before reset -> 400", resp.status_code == 400, f"code={resp.status_code}")

# POST /reset
reset_req = {"difficulty": "easy", "seed": 42}
resp = client.post("/reset", json=reset_req)
check("POST /reset -> 200", resp.status_code == 200, f"code={resp.status_code}")

obs = resp.json()
check("Reset returns ObservationSchema", "firm_cash" in obs and "step" in obs)
check("Initial step == 0", obs["step"] == 0)

# ============================================================
heading("2. Endpoint: GET /state")
# ============================================================

resp = client.get("/state")
check("GET /state -> 200", resp.status_code == 200)

state_data = resp.json()
check("State contains observation", "observation" in state_data)
check("State contains step/episode info",
      "step" in state_data and "episode_length" in state_data)
check("State contains termination flags",
      "terminated" in state_data and "truncated" in state_data)


# ============================================================
heading("3. Endpoint: POST /step")
# ============================================================

# Valid action
action = {
    "set_price": 10.0,
    "hire_workers": 2,
    "capex": 500.0,
    "lobby_tax_change": 0.0
}
resp = client.post("/step", json=action)
check("POST /step (valid) -> 200", resp.status_code == 200)

step_data = resp.json()
check("Step returns observation", "observation" in step_data)
check("Observation advanced", step_data["observation"]["step"] == 1)
check("Step returns reward", "reward" in step_data)
check("Reward contains feedback", "feedback" in step_data["reward"])
check("Reward score exists", "composite" in step_data["reward"])

# ============================================================
heading("4. Action Validation & Bounds")
# ============================================================

# Missing required field
bad_action_1 = {"hire_workers": 1}
resp = client.post("/step", json=bad_action_1)
check("Validation: missing set_price -> 422", resp.status_code == 422)

# Price below floor
bad_action_2 = {"set_price": PRICE_FLOOR - 0.5}
resp = client.post("/step", json=bad_action_2)
check("Validation: set_price below floor -> 422", resp.status_code == 422)

# Hire workers > max
bad_action_3 = {"set_price": 10.0, "hire_workers": MAX_HIRE_PER_STEP + 1}
resp = client.post("/step", json=bad_action_3)
check("Validation: hire_workers > max -> 422", resp.status_code == 422)

# Capex < 0
bad_action_4 = {"set_price": 10.0, "capex": -100.0}
resp = client.post("/step", json=bad_action_4)
check("Validation: capex < 0 -> 422", resp.status_code == 422)


# ============================================================
heading("5. Termination Guard")
# ============================================================

client.post("/reset", json={"difficulty": "easy", "seed": 42})

# Underprice until bankrupt
steps = 0
bankrupt = False
for i in range(30):
    r = client.post("/step", json={"set_price": 1.0}).json()
    steps += 1
    if r.get("terminated"):
        bankrupt = True
        break

check("Agent went bankrupt via underpricing", bankrupt, f"steps={steps}")

# Try to step AFTER bankruptcy
resp = client.post("/step", json={"set_price": 10.0})
check("Step after termination -> 400", resp.status_code == 400,
      f"detail={resp.json().get('detail')}")


# ============================================================
heading("6. Seed Reproducibility")
# ============================================================

# Run 1
client.post("/reset", json={"difficulty": "easy", "seed": 99})
for i in range(5):
    client.post("/step", json={"set_price": 10.0})
r1 = client.get("/state").json()

# Run 2
client.post("/reset", json={"difficulty": "easy", "seed": 99})
for i in range(5):
    client.post("/step", json={"set_price": 10.0})
r2 = client.get("/state").json()

cash1 = r1["observation"]["firm_cash"]
cash2 = r2["observation"]["firm_cash"]

check("Seed guarantees explicit reproducibility", abs(cash1 - cash2) < 0.001,
      f"run1={cash1:.2f}, run2={cash2:.2f}")


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")

if failed > 0:
    print("  WARNING: Fix failures before proceeding to Phase 6.\n")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED -- Phase 5 API Layer verified.\n")
    sys.exit(0)
