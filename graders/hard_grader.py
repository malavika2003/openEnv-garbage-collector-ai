"""Hard grading: strict fuel and zone-violation penalties."""

from __future__ import annotations

from typing import Any, Dict

from env.models import EpisodeSummary


def grade(summary: EpisodeSummary) -> Dict[str, Any]:
    mass = summary.total_mass_collected
    pending = summary.final_pending_garbage
    denom = mass + pending + 1e-6
    collection_ratio = float(mass / denom)

    fuel = max(summary.total_fuel_used, 1e-6)
    fuel_efficiency = mass / fuel
    benchmark = 6.0
    fuel_score = float(min(1.0, fuel_efficiency / benchmark))

    time_pressure = float(summary.steps_taken / max(summary.max_steps, 1))
    time_score = float(max(0.0, 1.0 - 0.55 * time_pressure))

    base = (
        0.45 * collection_ratio
        + 0.35 * fuel_score
        + 0.20 * time_score
    ) * 100.0

    penalty = summary.invalid_actions * 5.5 + summary.constraint_violations * 12.0
    penalty += min(summary.total_overflow_penalty_accumulated * 0.12, 25.0)

    score = float(max(0.0, min(100.0, base - penalty)))
    return {
        "score": score,
        "breakdown": {
            "collection_ratio": collection_ratio,
            "fuel_score": fuel_score,
            "time_score": time_score,
            "penalty": penalty,
            "total_mass_collected": mass,
            "final_pending_garbage": pending,
            "total_fuel_used": summary.total_fuel_used,
            "constraint_violations": summary.constraint_violations,
        },
    }
