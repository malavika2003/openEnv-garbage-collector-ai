import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import GarbageRoutingEnvironment
from env.models import Action, TaskConfig
from tasks.easy_task import get_task_config as get_easy_task_config
from tasks.hard_task import get_task_config as get_hard_task_config
from tasks.medium_task import get_task_config as get_medium_task_config

# Load `.env` from the repo root (same directory as this file). Does not override
# variables already set in the process environment.
load_dotenv(Path(__file__).resolve().parent / ".env")

_TASK_FACTORIES: Dict[str, Callable[[], TaskConfig]] = {
    "easy": get_easy_task_config,
    "medium": get_medium_task_config,
    "hard": get_hard_task_config,
}


def resolve_task_config(task: Optional[str] = None) -> TaskConfig:
    name = (task or os.getenv("TASK") or "medium").strip().lower()
    factory = _TASK_FACTORIES.get(name)
    if factory is None:
        allowed = ", ".join(sorted(_TASK_FACTORIES))
        raise ValueError(f"Unknown task {name!r}; choose one of: {allowed}")
    return factory()

_client: Optional[OpenAI] = None

# Cleared each simulation step in run(); avoids multiple trucks targeting the same hood in one step.
assigned_targets: set[int] = set()


def _get_client() -> Optional[OpenAI]:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("API_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    _client = OpenAI(**kwargs)
    return _client


def _parse_json_message(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def heuristic_action(
    obs,
    env: GarbageRoutingEnvironment,
    assigned_targets: set[int],
) -> Action:
    best_score = -1
    best_truck = None
    best_target = None

    dm = env.state()["distance_matrix"]
    allowed_all = obs.metadata.get("truck_allowed_neighborhoods")

    for truck_id, truck in enumerate(obs.trucks):
        # send full trucks to landfill
        if truck.load > truck.capacity * 0.8:
            return Action(truck_id=truck_id, neighborhood_id=-1)

        here = truck.location_node
        allowed_ids: Optional[set[int]] = None
        if allowed_all is not None and truck_id in allowed_all:
            allowed_ids = set(allowed_all[truck_id])

        for i in range(obs.num_neighborhoods):
            if i in assigned_targets:
                continue
            if allowed_ids is not None and i not in allowed_ids:
                continue
            garbage = obs.neighborhoods[i].garbage_mass
            if garbage <= 0:
                continue

            distance = float(dm[here][i])

            score = garbage / (distance + 1)

            if score > best_score:
                best_score = score
                best_truck = truck_id
                best_target = i

    if best_truck is None or best_target is None:
        for truck_id, truck in enumerate(obs.trucks):
            if truck.load > truck.capacity * 0.8:
                return Action(truck_id=truck_id, neighborhood_id=-1)
        truck_id = 0
        truck = obs.trucks[truck_id]
        if truck.load > truck.capacity * 0.8:
            return Action(truck_id=truck_id, neighborhood_id=-1)
        allowed_ids_list: Optional[list[int]] = None
        if allowed_all is not None and truck_id in allowed_all:
            allowed_ids_list = [
                i for i in allowed_all[truck_id] if 0 <= i < obs.num_neighborhoods
            ]
        candidates = (
            allowed_ids_list
            if allowed_ids_list is not None
            else list(range(obs.num_neighborhoods))
        )
        candidates = [c for c in candidates if c not in assigned_targets]
        if not candidates:
            return Action(truck_id=truck_id, neighborhood_id=-1)
        best_target = max(candidates, key=lambda i: obs.neighborhoods[i].garbage_mass)
        assigned_targets.add(best_target)
        return Action(truck_id=truck_id, neighborhood_id=best_target)

    assigned_targets.add(best_target)
    return Action(truck_id=best_truck, neighborhood_id=best_target)


def llm_refine_action(obs, heuristic: Action):
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        return heuristic
    client = _get_client()
    if client is None:
        return heuristic

    prompt = f"""
You control garbage trucks.

Observation summary:
- trucks: {len(obs.trucks)}
- neighborhoods: {obs.num_neighborhoods}

Heuristic suggestion:
truck_id = {heuristic.truck_id}
neighborhood_id = {heuristic.neighborhood_id}

Return JSON only:
{{
 "truck_id": int,
 "neighborhood_id": int
}}
"""

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content
        if not raw:
            return heuristic
        data = _parse_json_message(raw)
        nid = data.get("neighborhood_id", data.get("target_node"))
        if nid is None:
            return heuristic
        return Action(
            truck_id=int(data["truck_id"]),
            neighborhood_id=int(nid),
        )
    except Exception:
        return heuristic


def run(task: Optional[str] = None) -> None:
    cfg = resolve_task_config(task)

    env = GarbageRoutingEnvironment(cfg)

    seed_raw = os.getenv("EPISODE_SEED")
    if seed_raw is not None and str(seed_raw).strip() != "":
        episode_seed = int(seed_raw)
    else:
        episode_seed = random.randint(0, 10000)
    obs = env.reset(seed=episode_seed)

    print(
        json.dumps(
            {
                "event": "[START]",
                "task": cfg.name,
                "max_steps": cfg.max_steps,
                "episode_seed": episode_seed,
            }
        )
    )

    step = 0
    done = False
    total_reward = 0.0

    while not done and step < cfg.max_steps:
        assigned_targets.clear()
        heuristic = heuristic_action(obs, env, assigned_targets)

        action = llm_refine_action(obs, heuristic)

        obs, reward, done, info = env.step(action)

        total_reward += reward.total

        print(
            json.dumps(
                {
                    "event": "[STEP]",
                    "step": step,
                    "truck_id": action.truck_id,
                    "neighborhood_id": action.neighborhood_id,
                    "reward": reward.model_dump(),
                    "done": done,
                }
            )
        )

        step += 1

    print(
        json.dumps(
            {
                "event": "[END]",
                "steps": step,
                "total_reward": total_reward,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Garbage routing inference loop.")
    parser.add_argument(
        "--task",
        choices=sorted(_TASK_FACTORIES),
        default=None,
        help="Task tier (default: TASK env var, else medium).",
    )
    args = parser.parse_args()
    run(task=args.task)
    import time
    while True:
        time.sleep(3600)
