from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from env.models import (
    Action,
    EpisodeSummary,
    NeighborhoodObservation,
    Observation,
    Reward,
    TaskConfig,
    TruckObservation,
)
from env.simulator import GarbageCollectionSimulator


class GarbageRoutingEnvironment:
    """
    Municipal garbage collection routing simulation.

    One action per timestep selects a single truck and either a neighborhood to
    service (drive, collect up to remaining capacity) or landfill (-1) to dump.
    """

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self._sim = GarbageCollectionSimulator(config)

    def reset(self, seed: Optional[int] = None) -> Observation:
        self._sim.reset(seed=seed)
        return self.get_observation()

    def get_observation(self) -> Observation:
        return self._build_observation()

    def _build_observation(self) -> Observation:
        s = self._sim.state
        cfg = self.config
        hoods = [
            NeighborhoodObservation(
                neighborhood_id=i,
                garbage_mass=float(s.garbage[i]),
                fill_ratio=float(
                    min(s.garbage[i] / max(cfg.soft_garbage_threshold, 1e-6), 3.0),
                ),
            )
            for i in range(cfg.num_neighborhoods)
        ]
        trucks = [
            TruckObservation(
                truck_id=k,
                location_node=int(s.truck_position[k]),
                load=float(s.truck_load[k]),
                capacity=float(cfg.truck_capacities[k]),
                cumulative_fuel_used=float(s.truck_fuel_used[k]),
            )
            for k in range(cfg.num_trucks)
        ]
        meta: Dict[str, Any] = {
            "task": cfg.name,
            "fuel_per_distance_unit": cfg.fuel_per_distance_unit,
        }
        if cfg.truck_allowed_neighborhoods:
            meta["truck_allowed_neighborhoods"] = cfg.truck_allowed_neighborhoods

        return Observation(
            step=s.step,
            max_steps=cfg.max_steps,
            num_neighborhoods=cfg.num_neighborhoods,
            landfill_node_id=s.landfill_node,
            neighborhoods=hoods,
            trucks=trucks,
            total_garbage_across_hoods=float(s.garbage.sum()),
            total_truck_load=float(s.truck_load.sum()),
            metadata=meta,
        )

    def step(
        self,
        action: Action,
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Returns (observation, reward, terminated, info)."""
        cfg = self.config
        if self._sim._state is None:
            raise RuntimeError("Call reset() before step()")

        info, invalid, violated = self._sim.apply_action(action)
        s_after = self._sim.state

        mass_col = float(info.get("mass_collected_step", 0.0))
        fuel = float(info.get("fuel_step", 0.0))
        distance = float(info.get("distance_step", 0.0))

        over = s_after.garbage - cfg.soft_garbage_threshold
        positive_over = np.maximum(over, 0.0)
        miss_pen = float(positive_over.sum()) * cfg.overflow_penalty_weight * 0.05

        invalid_penalty_mag = 0.0
        if invalid or violated:
            invalid_penalty_mag = cfg.invalid_action_penalty

        collection_gain = cfg.collection_reward_weight * mass_col
        fuel_cost = cfg.fuel_penalty_weight * fuel
        distance_penalty = 0.1 * distance

        total = (
            collection_gain
            - fuel_cost
            - distance_penalty
            - miss_pen
            - invalid_penalty_mag
        )

        reward = Reward(
            total=total,
            collection_gain=collection_gain,
            fuel_cost=fuel_cost,
            missed_overflow_penalty=miss_pen,
            invalid_or_constraint_penalty=invalid_penalty_mag,
        )

        obs = self.get_observation()
        terminated = s_after.step >= cfg.max_steps
        truncated = False

        info_out = {
            **info,
            "invalid_action": invalid,
            "constraint_violation": violated,
            "terminated": terminated,
            "truncated": truncated,
            "reward_components": reward.model_dump(),
        }
        return obs, reward, terminated, info_out

    def state(self) -> Dict[str, Any]:
        """Full internal snapshot for debugging / graders."""
        s = self._sim.state
        cfg = self.config
        return {
            "step": s.step,
            "max_steps": cfg.max_steps,
            "garbage": s.garbage.tolist(),
            "truck_position": s.truck_position.tolist(),
            "truck_load": s.truck_load.tolist(),
            "truck_fuel_used": s.truck_fuel_used.tolist(),
            "landfill_node": s.landfill_node,
            "distance_matrix": s.distance_matrix.tolist(),
            "total_mass_collected": s.total_mass_collected,
            "total_fuel_used": s.total_fuel_used,
            "total_garbage_generated": s.total_garbage_generated,
            "overflow_penalty_accum": s.overflow_penalty_accum,
            "invalid_actions": s.invalid_actions,
            "constraint_violations": s.constraint_violations,
        }

    def episode_summary(self) -> EpisodeSummary:
        s = self._sim.state
        cfg = self.config
        return EpisodeSummary(
            task_name=cfg.name,
            steps_taken=s.step,
            max_steps=cfg.max_steps,
            total_mass_collected=s.total_mass_collected,
            total_fuel_used=s.total_fuel_used,
            total_garbage_generated_est=s.total_garbage_generated,
            final_pending_garbage=float(s.garbage.sum()),
            total_overflow_penalty_accumulated=s.overflow_penalty_accum,
            invalid_actions=s.invalid_actions,
            constraint_violations=s.constraint_violations,
        )
