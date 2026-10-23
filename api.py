import logging
import os
import threading
from collections import OrderedDict
from typing import List, Optional
import random

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simulator import EconSimulator
from reward_integrator import compute_step_reward
from config import (
    PRICE_FLOOR,
    MAX_HIRE_PER_STEP,
    LOBBY_TAX_CHANGE_MIN,
    LOBBY_TAX_CHANGE_MAX,
    MIN_WAGE,
    MAX_WAGE,
    WAGE_MAX_CHANGE,
    DEFAULT_PRICE,
    WAGE_PER_WORKER,
    INPUT_COST_PER_UNIT,
    LABOR_FORCE_TOTAL,
    NUM_AGENTS,
    SPILLOVER_FRACTION,
    BANKRUPTCY_PENALTY,
    PRESETS,
    DEFAULT_DIFFICULTY,
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("OpenEnvAPI")

app = FastAPI(
    title="OpenEnv API",
    description="Thin FastAPI wrapper around the OpenEnv economic simulator",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# One simulator per visitor, keyed by the X-Session-Id header.
# Requests without the header share the "default" session (used by /docs
# and test_api.py).  Least-recently-used sessions are evicted past the cap.
MAX_SESSIONS = 100
DEFAULT_SESSION = "default"
sim_lock = threading.Lock()
sessions: "OrderedDict[str, EconSimulator]" = OrderedDict()


def get_sim(session_id: Optional[str]) -> EconSimulator:
    """Return the caller's simulator, creating it on first use.  Hold sim_lock."""
    key = session_id or DEFAULT_SESSION
    if key not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.popitem(last=False)
        sessions[key] = EconSimulator()
    sessions.move_to_end(key)
    return sessions[key]

# -----------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: str = Field("easy", description="Presets: 'easy', 'medium', 'hard'")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ActionSchema(BaseModel):
    # Action bounds matching config + realistic limits
    set_price: float = Field(
        ..., 
        ge=PRICE_FLOOR, 
        description="Price for the agent's product"
    )
    hire_workers: int = Field(
        0, 
        ge=-1000,  # Simulator will clamp natively to -worker_count
        le=MAX_HIRE_PER_STEP, 
        description="Workers to hire (positive) or fire (negative)"
    )
    capex: float = Field(
        0.0, 
        ge=0.0, 
        description="Capital expenditure amount (dollars)"
    )
    lobby_tax_change: float = Field(
        0.0, 
        ge=LOBBY_TAX_CHANGE_MIN, 
        le=LOBBY_TAX_CHANGE_MAX, 
        description="Attempt to change the flat tax rate"
    )
    set_wage: Optional[float] = Field(
        None,
        ge=MIN_WAGE,
        le=MAX_WAGE,
        description="Target wage per worker per step (omit to keep current). "
                    "The wage moves at most 10% per step toward the target."
    )


class ObservationSchema(BaseModel):
    step: int
    firm_cash: float
    worker_count: int
    capital_stock: float
    current_price: float
    price_index: float
    market_demand: float
    tax_rate: float
    ema_revenue: float
    worker_wealth: float
    episode_length: int
    agent_wealth: List[float]
    # Labor market (lists are player first, then NPCs)
    wage: float
    market_wage: float
    unemployed: int
    labor_force: int
    workers_lost_last_step: int
    firm_active: List[bool]
    firm_wages: List[float]
    firm_workers: List[int]
    firm_prices: List[float]
    last_units_sold: float
    last_capacity: float
    last_demand: float


class RewardSchema(BaseModel):
    profit: float
    solvency: float
    gdp: float
    gini_pen: float
    composite: float
    feedback: str
    gini_coefficient: float
    runway_steps: float
    activity: float


class StepDetails(BaseModel):
    """Per-step breakdown for the dashboard (player first in firm lists)."""
    revenue: float
    labor_cost: float
    input_cost: float
    hiring_cost: float
    capex_spent: float
    tax_paid: float
    total_expenses: float
    net_cash_flow: float
    units_sold: float
    max_output: float
    demand: float
    spill_units: float
    market_share: float
    competitor_avg_price: float
    market_avg_price: float
    real_gdp: float
    hires_requested: int
    hires_filled: int
    poached_in: int
    poached_out: int
    fired: int
    quits: int
    firm_quits: List[int]
    firm_poached_out: List[int]
    firm_hires: List[int]
    labor_flows: List[List[int]]


class StepResponse(BaseModel):
    observation: ObservationSchema
    reward: RewardSchema
    details: StepDetails
    terminated: bool
    truncated: bool
    step: int
    episode_length: int

class StateResponse(BaseModel):
    observation: ObservationSchema
    step: int
    episode_length: int
    terminated: bool
    truncated: bool


# -----------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------

@app.get("/", include_in_schema=False)
def dashboard():
    """Interactive dashboard (static/index.html)."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health_check():
    """Health check endpoint (e.g. Hugging Face Spaces)"""
    return {"status": "ok", "message": "OpenEnv API is running and healthy"}


@app.get("/meta")
def get_meta():
    """Action bounds, defaults, and difficulty presets for clients."""
    return {
        "num_firms": NUM_AGENTS,
        "firm_names": ["You"] + [f"NPC {i}" for i in range(1, NUM_AGENTS)],
        "labor_force": LABOR_FORCE_TOTAL,
        "spillover_fraction": SPILLOVER_FRACTION,
        "bankruptcy_penalty": BANKRUPTCY_PENALTY,
        "default_difficulty": DEFAULT_DIFFICULTY,
        "defaults": {"price": DEFAULT_PRICE, "wage": WAGE_PER_WORKER},
        "bounds": {
            "price_floor": PRICE_FLOOR,
            "input_cost_per_unit": INPUT_COST_PER_UNIT,
            "min_wage": MIN_WAGE,
            "max_wage": MAX_WAGE,
            "wage_max_change": WAGE_MAX_CHANGE,
            "max_hire_per_step": MAX_HIRE_PER_STEP,
            "lobby_min": LOBBY_TAX_CHANGE_MIN,
            "lobby_max": LOBBY_TAX_CHANGE_MAX,
        },
        "presets": {
            name: {"episode_length": p.episode_length, "weights": p.weights_dict()}
            for name, p in PRESETS.items()
        },
    }


@app.post("/reset", response_model=ObservationSchema)
def reset_env(
    req: ResetRequest,
    x_session_id: Optional[str] = Header(None, max_length=64),
):
    """Re-initialize the simulator environment."""
    with sim_lock:
        sim = get_sim(x_session_id)
        if req.seed is not None:
            # Set stdlib global random seed if needed, though EconSimulator
            # uses its own instanced sim.rng. But setting it explicitly is safe.
            random.seed(req.seed)
            sim._seed = req.seed

        try:
            # Recreate the simulator with the correct difficulty and seed
            sim.__init__(difficulty=req.difficulty, seed=sim._seed)
            obs = sim.reset(seed=sim._seed)
            logger.info(f"[RESET] Environment reset. Difficulty: {req.difficulty}, Seed: {sim._seed}")
            return obs
        except Exception as e:
            logger.error(f"[ERROR] Reset failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResponse)
def get_state(x_session_id: Optional[str] = Header(None, max_length=64)):
    """Get the current state without advancing the simulation."""
    with sim_lock:
        sim = get_sim(x_session_id)
        if sim.player is None:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        try:
            obs = sim._build_observation()
            return StateResponse(
                observation=obs,
                step=sim.step_count,
                episode_length=sim.preset.episode_length,
                terminated=sim.terminated,
                truncated=sim.truncated
            )
        except Exception as e:
            logger.error(f"[ERROR] State retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step_env(
    action: ActionSchema,
    x_session_id: Optional[str] = Header(None, max_length=64),
):
    """Execute one timestep of the simulation."""
    with sim_lock:
        sim = get_sim(x_session_id)
        if sim.player is None:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        if sim.terminated or sim.truncated:
            raise HTTPException(
                status_code=400, 
                detail="Episode finished (terminated or truncated). Call /reset."
            )
            
        try:
            # Convert ActionSchema to dict format expected by the simulator
            action_dict = action.model_dump()
            
            # Step the simulator
            obs, info = sim.step(action_dict)
            
            # Formulate reward
            reward_res = compute_step_reward(info, sim)
            
            logger.info(
                f"[STEP {sim.step_count}/{sim.preset.episode_length}] "
                f"Reward: {reward_res.composite:.3f} | Net Flow: ${info.net_cash_flow:,.0f}"
            )
            
            total_rev = info.total_revenue_all
            details = StepDetails(
                revenue=info.revenue,
                labor_cost=info.labor_cost,
                input_cost=info.input_cost,
                hiring_cost=info.hiring_cost,
                capex_spent=info.capex_spent,
                tax_paid=info.tax_paid,
                total_expenses=info.total_expenses,
                net_cash_flow=info.net_cash_flow,
                units_sold=info.units_sold,
                max_output=info.max_output,
                demand=info.demand,
                spill_units=info.spill_units,
                market_share=info.revenue / total_rev if total_rev > 0 else 0.0,
                competitor_avg_price=info.competitor_avg_price,
                market_avg_price=info.market_avg_price,
                real_gdp=info.real_gdp_current,
                hires_requested=info.hires_requested,
                hires_filled=info.hires_filled,
                poached_in=info.poached_in,
                poached_out=info.poached_out,
                fired=info.fired,
                quits=info.quits,
                firm_quits=info.firm_quits,
                firm_poached_out=info.firm_poached_out,
                firm_hires=info.firm_hires,
                labor_flows=info.labor_flows,
            )

            return StepResponse(
                observation=obs,
                # Convert the dataclass to a dict for Pydantic serialization
                reward=RewardSchema(**reward_res.__dict__),
                details=details,
                terminated=sim.terminated,
                truncated=sim.truncated,
                step=sim.step_count,
                episode_length=sim.preset.episode_length
            )
            
        except AssertionError as ae:
            # Catch simulation invariant violations explicitly
            logger.error(f"[ASSERTION ERROR] {str(ae)}")
            raise HTTPException(status_code=500, detail=f"Simulation invariant violation: {str(ae)}")
        except Exception as e:
            logger.error(f"[ERROR] Step failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
