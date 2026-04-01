import logging
import threading
from typing import List, Optional
import random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from simulator import EconSimulator
from reward_integrator import compute_step_reward
from config import (
    PRICE_FLOOR,
    MAX_HIRE_PER_STEP,
    LOBBY_TAX_CHANGE_MIN,
    LOBBY_TAX_CHANGE_MAX,
)

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("OpenEnvAPI")

app = FastAPI(
    title="OpenEnv API",
    description="Thin FastAPI wrapper around the OpenEnv economic simulator",
    version="1.0.0",
)

# Global simulator and lock for single-node thread safety
sim_lock = threading.Lock()
sim = EconSimulator()

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


class RewardSchema(BaseModel):
    profit: float
    solvency: float
    gdp: float
    gini_pen: float
    composite: float
    feedback: str
    gini_coefficient: float
    runway_steps: float


class StepResponse(BaseModel):
    observation: ObservationSchema
    reward: RewardSchema
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

@app.get("/")
def health_check():
    """Root endpoint for health checks (e.g. Hugging Face Spaces)"""
    return {"status": "ok", "message": "OpenEnv API is running and healthy"}


@app.post("/reset", response_model=ObservationSchema)
def reset_env(req: ResetRequest):
    """Re-initialize the simulator environment."""
    with sim_lock:
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
def get_state():
    """Get the current state without advancing the simulation."""
    with sim_lock:
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
def step_env(action: ActionSchema):
    """Execute one timestep of the simulation."""
    with sim_lock:
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
            
            return StepResponse(
                observation=obs,
                # Convert the dataclass to a dict for Pydantic serialization
                reward=RewardSchema(**reward_res.__dict__),
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
