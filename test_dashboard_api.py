"""
Dashboard & Agent API Tests
===========================
Run:  python test_dashboard_api.py   (needs fastapi + httpx)
"""

import sys
from fastapi.testclient import TestClient

from api import app
from agents import AGENTS, run_episode

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
heading("1. Static dashboard and metadata")
# ============================================================

resp = client.get("/")
check("GET / serves the dashboard HTML",
      resp.status_code == 200 and "text/html" in resp.headers["content-type"])
for path in ("/static/app.js", "/static/style.css"):
    check(f"GET {path}", client.get(path).status_code == 200)
check("GET /health", client.get("/health").json().get("status") == "ok")
meta = client.get("/meta").json()
check("/meta lists presets and bounds",
      set(meta["presets"]) == {"easy", "medium", "hard"} and "max_wage" in meta["bounds"])


# ============================================================
heading("2. Step details")
# ============================================================

h = {"X-Session-Id": "dash-test"}
client.post("/reset", json={"difficulty": "easy", "seed": 1}, headers=h)
step = client.post("/step", json={"set_price": 10.0, "hire_workers": 2}, headers=h).json()
d = step["details"]
check("Step response includes details", "labor_flows" in d and "market_share" in d)
check("Details: units sold <= capacity", d["units_sold"] <= d["max_output"] + 1e-6)
check("Details: hires show up as pool -> You flow", [-1, 0, 2] in d["labor_flows"],
      f"flows = {d['labor_flows']}")


# ============================================================
heading("3. Agents and leaderboard")
# ============================================================

agents = client.get("/agents").json()
check("/agents lists every built-in agent", [a["name"] for a in agents] == list(AGENTS))

res = client.post("/compare", json={"difficulty": "easy", "seeds": 5}).json()
rows = res["rows"]
worths = [r["net_worth"] for r in rows]
check("Leaderboard has one row per agent", len(rows) == len(AGENTS))
check("Leaderboard ranked by net worth", worths == sorted(worths, reverse=True))
check("Confidence intervals are non-negative", all(r["net_worth_ci"] >= 0 for r in rows))
check("Bankrupt agents carry the penalty",
      all(r["net_worth"] < 0 for r in rows if r["bankrupt_rate"] == 1.0))
sub = client.post("/compare", json={"agents": ["hold", "random"], "seeds": 3}).json()
check("Subset comparison", {r["agent"] for r in sub["rows"]} == {"hold", "random"})
check("Unknown agent -> 422",
      client.post("/compare", json={"agents": ["nope"]}).status_code == 422)
check("Unknown difficulty -> 422",
      client.post("/compare", json={"difficulty": "insane"}).status_code == 422)
check("Too many seeds -> 422",
      client.post("/compare", json={"seeds": 500}).status_code == 422)


# ============================================================
heading("4. Replays")
# ============================================================

rep = client.post("/simulate", json={"agent": "rule_based", "difficulty": "hard", "seed": 3}).json()
frames = rep["frames"]
check("Replay has one frame per step plus the start",
      len(frames) == frames[-1]["observation"]["step"] + 1, f"{len(frames)} frames")
check("First frame has no action or reward",
      frames[0]["action"] is None and frames[0]["reward"] is None)
check("Every later frame has action, reward and details",
      all(f["action"] and f["reward"] and f["details"] for f in frames[1:]))
direct = run_episode(AGENTS["rule_based"](), "hard", 3)
check("Replay is deterministic: matches a direct run of the same seed",
      abs(rep["net_worth"] - direct["net_worth"]) < 1e-6,
      f"replay {rep['net_worth']:.0f} vs direct {direct['net_worth']:.0f}")
rand1 = client.post("/simulate", json={"agent": "random", "seed": 7}).json()["net_worth"]
rand2 = client.post("/simulate", json={"agent": "random", "seed": 7}).json()["net_worth"]
check("Random agent is reproducible per seed", rand1 == rand2)
check("Unknown agent -> 422",
      client.post("/simulate", json={"agent": "nope"}).status_code == 422)


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")
sys.exit(1 if failed else 0)
