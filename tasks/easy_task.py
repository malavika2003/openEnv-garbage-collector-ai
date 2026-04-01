"""Easy scenario: 5 neighborhoods, single truck."""

from env.models import TaskConfig


def get_task_config() -> TaskConfig:
    return TaskConfig(
        name="easy",
        num_neighborhoods=5,
        num_trucks=1,
        max_steps=120,
        truck_capacities=[120.0],
        fuel_per_distance_unit=0.35,
        accumulation_mean=2.5,
        accumulation_stochastic=True,
        soft_garbage_threshold=25.0,
        overflow_penalty_weight=0.45,
        collection_reward_weight=1.0,
        fuel_penalty_weight=0.06,
        invalid_action_penalty=1.5,
        distance_noise=0.05,
        seed=11,
        truck_allowed_neighborhoods=None,
    )
