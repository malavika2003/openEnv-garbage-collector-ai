"""Medium scenario: 10 neighborhoods, two trucks."""

from env.models import TaskConfig


def get_task_config() -> TaskConfig:
    return TaskConfig(
        name="medium",
        num_neighborhoods=10,
        num_trucks=2,
        max_steps=220,
        truck_capacities=[90.0, 90.0],
        fuel_per_distance_unit=0.40,
        accumulation_mean=3.0,
        accumulation_stochastic=True,
        soft_garbage_threshold=22.0,
        overflow_penalty_weight=0.55,
        collection_reward_weight=1.0,
        fuel_penalty_weight=0.065,
        invalid_action_penalty=2.0,
        distance_noise=0.08,
        seed=42,
        truck_allowed_neighborhoods=None,
    )
