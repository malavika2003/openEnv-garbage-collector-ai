"""Low-level stochastic dynamics: distances, accumulation, truck physics, fuel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.models import Action, TaskConfig


@dataclass
class SimulationState:
    """Mutable internal state; not exposed to agents directly."""

    step: int = 0
    garbage: np.ndarray = field(default_factory=lambda: np.array([]))
    truck_position: np.ndarray = field(default_factory=lambda: np.array([]))
    truck_load: np.ndarray = field(default_factory=lambda: np.array([]))
    truck_fuel_used: np.ndarray = field(default_factory=lambda: np.array([]))
    distance_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    landfill_node: int = 0
    total_mass_collected: float = 0.0
    total_fuel_used: float = 0.0
    total_garbage_generated: float = 0.0
    overflow_penalty_accum: float = 0.0
    invalid_actions: int = 0
    constraint_violations: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


def _build_distance_matrix(
    n_hoods: int,
    rng: np.random.Generator,
    noise: float,
) -> np.ndarray:
    """Symmetric distances between hoods + landfill node index n_hoods."""
    n_nodes = n_hoods + 1
    coords = rng.uniform(0.0, 10.0, size=(n_nodes, 2))
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    if noise > 0:
        dist += rng.normal(0.0, noise, size=dist.shape)
        dist = np.maximum(dist, 0.01)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


class GarbageCollectionSimulator:
    """Advances world state given task configuration and actions."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self._state: Optional[SimulationState] = None

    @property
    def state(self) -> SimulationState:
        if self._state is None:
            raise RuntimeError("Simulator not reset")
        return self._state

    def reset(self, seed: Optional[int] = None) -> SimulationState:
        cfg = self.config
        base = cfg.seed if cfg.seed is not None else 0
        merged = int(base) + (int(seed) if seed is not None else 0)
        rng = np.random.default_rng(merged)

        n = cfg.num_neighborhoods
        nt = cfg.num_trucks
        dist = _build_distance_matrix(n, rng, cfg.distance_noise)
        landfill = n

        # Start trucks empty at landfill (depot).
        trucks_at_landfill = np.full(nt, landfill, dtype=np.int32)
        load = np.zeros(nt, dtype=np.float64)
        fuel = np.zeros(nt, dtype=np.float64)

        # Initial garbage: light random backlog.
        garbage = rng.uniform(0.0, cfg.soft_garbage_threshold * 0.8, size=n).astype(np.float64)

        self._state = SimulationState(
            step=0,
            garbage=garbage,
            truck_position=trucks_at_landfill,
            truck_load=load,
            truck_fuel_used=fuel,
            distance_matrix=dist,
            landfill_node=landfill,
            rng=rng,
        )
        return self._state

    def _accumulate_garbage(self) -> float:
        s = self.state
        cfg = self.config
        n = cfg.num_neighborhoods
        arrivals = np.zeros(n, dtype=np.float64)
        if cfg.accumulation_stochastic:
            # Poisson counts scaled to mean accumulation_mean
            lam = max(cfg.accumulation_mean, 1e-6)
            draws = s.rng.poisson(lam=lam, size=n).astype(np.float64)
            arrivals = draws
        else:
            arrivals.fill(cfg.accumulation_mean)
        s.garbage += arrivals
        mass_in = float(arrivals.sum())
        s.total_garbage_generated += mass_in
        return mass_in

    def _overflow_penalty(self) -> float:
        s = self.state
        cfg = self.config
        over = np.maximum(s.garbage - cfg.soft_garbage_threshold, 0.0)
        pen = float(over.sum()) * cfg.overflow_penalty_weight
        s.overflow_penalty_accum += pen
        return pen

    def apply_action(self, action: Action) -> Tuple[Dict[str, Any], bool, bool]:
        """
        One environment step:
        1) validate / constrain action
        2) move truck, burn fuel, collect or dump
        3) accumulate garbage
        4) compute overflow penalty term

        Returns (info_dict, invalid_flag, constraint_violation_flag)
        """
        s = self.state
        cfg = self.config
        n = cfg.num_neighborhoods
        landfill = s.landfill_node
        info: Dict[str, Any] = {
            "mass_collected_step": 0.0,
            "fuel_step": 0.0,
            "distance_step": 0.0,
            "moved": False,
        }
        invalid = False
        violated = False

        tid = action.truck_id
        if tid < 0 or tid >= cfg.num_trucks:
            s.invalid_actions += 1
            invalid = True
            self._accumulate_garbage()
            self._overflow_penalty()
            s.step += 1
            return info, invalid, violated

        target = action.neighborhood_id
        if target != -1 and (target < 0 or target >= n):
            s.invalid_actions += 1
            invalid = True
            self._accumulate_garbage()
            self._overflow_penalty()
            s.step += 1
            return info, invalid, violated

        # Hard-mode zoning: if truck may only visit subset of hoods.
        allowed = cfg.truck_allowed_neighborhoods
        if allowed is not None and tid in allowed and target != -1:
            if target not in allowed[tid]:
                s.constraint_violations += 1
                violated = True
                self._accumulate_garbage()
                self._overflow_penalty()
                s.step += 1
                return info, invalid, violated

        here = int(s.truck_position[tid])
        if target == -1:
            dest = landfill
        else:
            dest = int(target)

        dist = float(s.distance_matrix[here, dest])
        fuel_step = dist * cfg.fuel_per_distance_unit
        s.truck_fuel_used[tid] += fuel_step
        s.total_fuel_used += fuel_step
        s.truck_position[tid] = dest
        info["fuel_step"] = fuel_step
        info["distance_step"] = dist
        info["moved"] = True

        mass_collected = 0.0
        if dest != landfill:
            cap = cfg.truck_capacities[tid]
            room = max(cap - s.truck_load[tid], 0.0)
            take = min(room, s.garbage[dest])
            s.garbage[dest] -= take
            s.truck_load[tid] += take
            mass_collected = take
        else:
            # dump all at landfill
            mass_collected = 0.0
            s.truck_load[tid] = 0.0

        s.total_mass_collected += mass_collected
        info["mass_collected_step"] = mass_collected

        self._accumulate_garbage()
        self._overflow_penalty()

        s.step += 1
        return info, invalid, violated
