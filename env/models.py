"""Typed Pydantic models for actions, observations, rewards, and task configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Action: neighborhood_id == -1 means route the chosen truck to the landfill to dump.


class Action(BaseModel):
    """Agent decision for one discrete time step (one truck is commanded)."""

    truck_id: int = Field(ge=0, description="Index of the truck to dispatch.")
    neighborhood_id: int = Field(
        ...,
        description=(
            "Neighborhood index (0 .. N-1) to drive to and collect from, "
            "or -1 to send the truck to the landfill to empty its load."
        ),
    )

    @field_validator("neighborhood_id")
    @classmethod
    def validate_target(cls, v: int) -> int:
        if v < -1:
            raise ValueError("neighborhood_id must be >= -1")
        return v


class TruckObservation(BaseModel):
    truck_id: int
    location_node: int = Field(
        description="Node index: 0..N-1 neighborhood, N is landfill depot.",
    )
    load: float = Field(ge=0)
    capacity: float = Field(gt=0)
    cumulative_fuel_used: float = Field(ge=0)


class NeighborhoodObservation(BaseModel):
    neighborhood_id: int
    garbage_mass: float = Field(ge=0)
    """Ratio of current garbage to soft capacity; caps at 1 for RL stability."""
    fill_ratio: float = Field(ge=0)


class Observation(BaseModel):
    """What the agent sees at the start of a step (after any reset bookkeeping)."""

    step: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    num_neighborhoods: int = Field(gt=0)
    landfill_node_id: int = Field(ge=0)
    neighborhoods: List[NeighborhoodObservation]
    trucks: List[TruckObservation]
    total_garbage_across_hoods: float = Field(ge=0)
    total_truck_load: float = Field(ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    """Decomposed reward for debugging and grader alignment."""

    total: float
    collection_gain: float = Field(description="Positive contribution from mass collected this step.")
    fuel_cost: float = Field(description="Negative magnitude; subtracted from total.")
    missed_overflow_penalty: float = Field(description="Penalty for garbage above soft thresholds.")
    invalid_or_constraint_penalty: float = Field(
        default=0.0,
        description="Penalty for invalid moves or hard-mode zone violations.",
    )


class TaskConfig(BaseModel):
    """Scenario definition; each task file builds one of these."""

    name: str = Field(description="Human-readable task name.")
    num_neighborhoods: int = Field(gt=0)
    num_trucks: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    truck_capacities: List[float] = Field(
        min_length=1,
        description="Per-truck max load mass; length must equal num_trucks.",
    )
    fuel_per_distance_unit: float = Field(gt=0, description="Fuel burned per distance unit when moving.")
    accumulation_mean: float = Field(
        gt=0,
        description="Mean garbage mass arriving per hood per step (Poisson or Gaussian drawn around this).",
    )
    accumulation_stochastic: bool = Field(
        default=True,
        description="If True, use Poisson-like draws; else fixed mean per step.",
    )
    soft_garbage_threshold: float = Field(
        gt=0,
        description="Neighborhood garbage above this level increases overflow penalty.",
    )
    overflow_penalty_weight: float = Field(default=0.5, ge=0)
    collection_reward_weight: float = Field(default=1.0, gt=0)
    fuel_penalty_weight: float = Field(default=0.05, gt=0)
    invalid_action_penalty: float = Field(default=2.0, ge=0)
    distance_noise: float = Field(default=0.0, ge=0, description="Jitter added to pairwise distances.")
    seed: Optional[int] = Field(default=None, description="Base RNG seed for reproducibility.")
    """Optional hard constraints: truck_id -> allowed neighborhood ids (empty list = unrestricted)."""
    truck_allowed_neighborhoods: Optional[Dict[int, List[int]]] = None

    @model_validator(mode="after")
    def capacities_match_trucks(self) -> TaskConfig:
        if len(self.truck_capacities) != self.num_trucks:
            raise ValueError("truck_capacities length must equal num_trucks")
        return self


class EpisodeSummary(BaseModel):
    """Aggregates for grading and reporting."""

    task_name: str
    steps_taken: int
    max_steps: int
    total_mass_collected: float
    total_fuel_used: float
    total_garbage_generated_est: float
    final_pending_garbage: float = Field(ge=0)
    total_overflow_penalty_accumulated: float
    invalid_actions: int
    constraint_violations: int
