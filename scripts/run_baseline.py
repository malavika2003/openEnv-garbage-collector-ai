#!/usr/bin/env python3
"""Baseline rollouts: random heuristic or OpenAI chat JSON actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.environment import GarbageRoutingEnvironment
from env.models import Action, Observation, TaskConfig
from graders.easy_grader import grade as grade_easy
from graders.hard_grader import grade as grade_hard
from graders.medium_grader import grade as grade_medium
from tasks.easy_task import get_task_config as easy_task
from tasks.hard_task import get_task_config as hard_task
from tasks.medium_task import get_task_config as medium_task

TaskGetter = Callable[[], TaskConfig]
GraderFn = Callable[..., dict]

REGISTRY: Dict[str, Tuple[TaskGetter, GraderFn]] = {
    "easy": (easy_task, grade_easy),
    "medium": (medium_task, grade_medium),
    "hard": (hard_task, grade_hard),
}


def observation_to_prompt_dict(obs: Observation) -> dict:
    """Compact JSON-serializable view for LLM consumption."""
    return {
        "step": obs.step,
        "max_steps": obs.max_steps,
        "num_neighborhoods": obs.num_neighborhoods,
        "landfill_node_id": obs.landfill_node_id,
        "neighborhoods": [n.model_dump() for n in obs.neighborhoods],
        "trucks": [t.model_dump() for t in obs.trucks],
        "totals": {
            "garbage_on_streets": obs.total_garbage_across_hoods,
            "truck_load_sum": obs.total_truck_load,
        },
        "metadata": obs.metadata,
    }


def valid_targets_for_truck(cfg: TaskConfig, truck_id: int) -> list[int]:
    hoods = list(range(cfg.num_neighborhoods))
    allowed = cfg.truck_allowed_neighborhoods
    if allowed is not None and truck_id in allowed:
        return list(allowed[truck_id])
    return hoods


def random_policy(obs: Observation, cfg: TaskConfig, rng: np.random.Generator) -> Action:
    truck_id = int(rng.integers(0, cfg.num_trucks))
    truck = obs.trucks[truck_id]
    targets = valid_targets_for_truck(cfg, truck_id)
    if truck.load >= 0.82 * truck.capacity:
        return Action(truck_id=truck_id, neighborhood_id=-1)
    if not targets:
        return Action(truck_id=truck_id, neighborhood_id=-1)
    nid = int(rng.choice(targets))
    return Action(truck_id=truck_id, neighborhood_id=nid)


def openai_policy(
    obs: Observation,
    cfg: TaskConfig,
    *,
    client: Any,
    model: str,
) -> Action:
    system = (
        "You are a municipal fleet router. Each turn you choose one truck and either "
        "service a neighborhood index from 0..N-1, or send the truck to the landfill with "
        "neighborhood_id = -1 to empty its load. Trucks must respect metadata.truck_allowed_neighborhoods "
        "when present. Reply ONLY JSON: {\"truck_id\": int, \"neighborhood_id\": int}"
    )
    user = json.dumps(
        {
            "observation": observation_to_prompt_dict(obs),
            "rules": {
                "landfill_id_observation_field": "landfill_node_id",
                "dump_action_value": -1,
            },
        },
        indent=2,
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    act = Action.model_validate(data)
    if act.neighborhood_id != -1:
        allowed = valid_targets_for_truck(cfg, act.truck_id)
        if allowed and act.neighborhood_id not in allowed:
            act = Action(truck_id=act.truck_id, neighborhood_id=-1)
    return act


def rollout(
    task_name: str,
    *,
    agent: str,
    seed: int,
    model: str,
    max_steps_override: int | None,
) -> dict:
    getter, grader = REGISTRY[task_name]
    cfg = getter()
    if max_steps_override is not None:
        cfg = cfg.model_copy(update={"max_steps": max_steps_override})
    env = GarbageRoutingEnvironment(cfg)
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    rewards: list[float] = []
    done = False
    client = None
    if agent == "openai":
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --agent openai")
        client = OpenAI()

    while not done:
        if agent == "random":
            action = random_policy(obs, cfg, rng)
        else:
            assert client is not None
            action = openai_policy(obs, cfg, client=client, model=model)
        obs, rew, done, info = env.step(action)
        rewards.append(rew.total)

    summary = env.episode_summary()
    report = grader(summary)
    return {
        "task": task_name,
        "agent": agent,
        "seed": seed,
        "cumulative_reward": float(np.sum(rewards)),
        "grader": report,
        "summary": summary.model_dump(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Garbage Routing OpenEnv baselines")
    parser.add_argument("--task", choices=list(REGISTRY.keys()), default="easy")
    parser.add_argument("--agent", choices=["random", "openai"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    result = rollout(
        args.task,
        agent=args.agent,
        seed=args.seed,
        model=args.model,
        max_steps_override=args.max_steps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
