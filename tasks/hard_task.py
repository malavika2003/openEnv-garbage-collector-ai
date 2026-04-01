"""Hard scenario: 20 neighborhoods, three trucks with non-overlapping zones."""

from env.models import TaskConfig


def get_task_config() -> TaskConfig:
    # Each truck restricted to a slice; must coordinate dumps at shared landfill (-1).
    return TaskConfig(
        name="hard",
        num_neighborhoods=20,
        num_trucks=3,
        max_steps=420,
        truck_capacities=[70.0, 70.0, 70.0],
        fuel_per_distance_unit=0.48,
        accumulation_mean=3.4,
        accumulation_stochastic=True,
        soft_garbage_threshold=20.0,
        overflow_penalty_weight=0.65,
        collection_reward_weight=1.0,
        fuel_penalty_weight=0.07,
        invalid_action_penalty=2.5,
        distance_noise=0.10,
        seed=97,
        truck_allowed_neighborhoods={
            0: list(range(0, 7)),
            1: list(range(7, 14)),
            2: list(range(14, 20)),
        },
    )
