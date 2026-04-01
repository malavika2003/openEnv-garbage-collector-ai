# Garbage Collection Routing OpenEnv

Production-style simulation of **municipal garbage collection**: an agent dispatches trucks to neighborhoods, balances **capacity** and **fuel**, and times **landfill dumps** while garbage **accumulates stochastically**. The repo exposes an OpenEnv-style API (`reset`, `step`, `state`, `get_observation`), three task tiers, graders, and baseline rollouts (**random** or **OpenAI**).

## Features

- **Pydantic** models: `Action`, `Observation`, `Reward`, `TaskConfig`, `EpisodeSummary`.
- **Dynamics:** stochastic per-step garbage arrivals (Poisson-like), symmetric **distance matrix** over neighborhoods + landfill, **fuel** ∝ distance, truck **capacity** and **dump** at landfill (`neighborhood_id == -1`).
- **Reward shaping:** collection bonus, fuel cost, optional **distance** term (`0.1 ×` step travel distance from the simulator), soft **overflow** penalty, invalid / zone-violation penalties.
- **Tasks:** easy (5 hoods, 1 truck), medium (10, 2), hard (20, 3 with **zone constraints** via `truck_allowed_neighborhoods`).
- **Graders:** score in **[0, 100]** from collection ratio, fuel efficiency, time pressure, overflow / invalid / constraint penalties.
- **Scripts:** `run_baseline.py` (single episode JSON to stdout), `run_all_tasks.py` (easy/medium/hard random sweep + scoreboard).

## Repository layout

```text
openEnv-garbage-collector-ai/
├── openenv.yaml              # Environment metadata, task/grader entrypoints, smoke_tests
├── Dockerfile
├── requirements.txt
├── README.md
├── LICENSE
├── env/
│   ├── __init__.py
│   ├── environment.py        # GarbageRoutingEnvironment
│   ├── models.py             # Action, Observation, Reward, TaskConfig, EpisodeSummary
│   └── simulator.py          # RNG, distances, transitions, step info (fuel_step, distance_step)
├── tasks/
│   ├── easy_task.py          # get_task_config()
│   ├── medium_task.py
│   └── hard_task.py
├── graders/
│   ├── easy_grader.py        # grade(summary) -> dict
│   ├── medium_grader.py
│   └── hard_grader.py
└── scripts/
    ├── run_baseline.py
    └── run_all_tasks.py      # subprocess sweep; run from repo root
```

## Requirements

- **Python 3.12** (see `Dockerfile` / `openenv.yaml`).
- Install: `pip install -r requirements.txt`  
  Core: `numpy`, `pydantic`, `openai`, `PyYAML`. The file may also list tooling such as `matplotlib` / `imageio` for local experimentation.

## Quick start (local)

From the repository root (`openEnv-garbage-collector-ai`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_baseline.py --task easy --agent random --seed 0
```

`run_baseline.py` prepends the repo root to `sys.path`, so you **do not** need `PYTHONPATH` for that script when your current working directory is the repo root.

For **one-off imports** (REPL, notebooks, `python -c "..."`), set the root on the path, e.g.:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -c "from env.environment import GarbageRoutingEnvironment; from tasks.easy_task import get_task_config; e=GarbageRoutingEnvironment(get_task_config()); print(e.reset(0).step)"
```

```bash
# macOS / Linux
export PYTHONPATH="$PWD"
```

### Baseline CLI (`scripts/run_baseline.py`)

| Flag | Description |
|------|-------------|
| `--task` | `easy` \| `medium` \| `hard` (default `easy`) |
| `--agent` | `random` \| `openai` (default `random`) |
| `--seed` | RNG integer (default `0`) |
| `--model` | Chat model when using OpenAI (default `gpt-4o-mini`) |
| `--max-steps` | Optional cap overriding the task’s `max_steps` |

Output is a single **JSON** object: `cumulative_reward`, `grader` (`score`, `breakdown`), and `summary` (`EpisodeSummary` fields).

### OpenAI agent

```powershell
$env:OPENAI_API_KEY = "sk-..."
python scripts/run_baseline.py --task medium --agent openai --model gpt-4o-mini --seed 1
```

The model must return **only** JSON: `{"truck_id": <int>, "neighborhood_id": <int>}`. Use `neighborhood_id = -1` to send the truck to the landfill. If the model picks a neighborhood forbidden by `metadata.truck_allowed_neighborhoods` (hard task), the baseline **replaces** the action with a dump (`-1`).

### Run all tasks (`scripts/run_all_tasks.py`)

Runs `run_baseline.py` for **easy**, **medium**, and **hard** with `--agent random` and `--seed 0`, parses JSON scores, and prints a short scoreboard. Execute from the **repo root** so `python scripts/run_baseline.py` resolves correctly:

```powershell
python scripts/run_all_tasks.py
```

## Environment API

### Action

- `truck_id`: which truck to command this timestep.
- `neighborhood_id`: `0 .. N-1` to drive there and collect (up to free capacity), or **`-1`** to go to the landfill and empty the truck.

### `GarbageRoutingEnvironment(config: TaskConfig)`

| Method | Returns |
|--------|---------|
| `reset(seed=None)` | `Observation` |
| `get_observation()` | `Observation` (same structure as after `reset` / `step`) |
| `step(action)` | `(Observation, Reward, terminated: bool, info: dict)` — `terminated` when `step >= max_steps` |
| `state()` | Full numeric dict (garbage vector, positions, distance matrix, counters) |
| `episode_summary()` | `EpisodeSummary` for graders |

### Step `info` dict (excerpt)

Includes simulator keys such as `mass_collected_step`, `fuel_step`, **`distance_step`** (edge length for the move, `0` if no move), `moved`, plus `invalid_action`, `constraint_violation`, `terminated`, `truncated`, `reward_components`.

### Reward intuition

- **Positive:** mass collected this step (weighted by `collection_reward_weight`).
- **Negative:** fuel (`fuel_penalty_weight × fuel_step`), **distance** (`0.1 × distance_step`), overflow mass above soft threshold, invalid / constraint penalties from `TaskConfig`.

## Tasks

| Task   | Neighborhoods | Trucks | Notes |
|--------|----------------|--------|--------|
| easy   | 5              | 1      | Single-truck routing |
| medium | 10             | 2      | Two trucks, shared landfill |
| hard   | 20             | 3      | Zoning: see `tasks/hard_task.py` |

## Graders

Each `graders/<difficulty>_grader.py` exposes `grade(summary: EpisodeSummary) -> dict` with `score` in **[0, 100]** and `breakdown` (collection ratio, fuel score, time score, penalties, key totals).

## Docker
### Build

```bash
docker build -t garbage-ai .
```

### Run a single task (baseline JSON output)

Easy:

```bash
docker run --rm garbage-ai python scripts/run_baseline.py --task easy
```

Medium:

```bash
docker run --rm garbage-ai python scripts/run_baseline.py --task medium
```

Hard:

```bash
docker run --rm garbage-ai python scripts/run_baseline.py --task hard
```

### Get the scoreboard (easy + medium + hard)

1. Open `Dockerfile`.
2. Switch the container `CMD` to run `scripts/run_all_tasks.py`:
   - Comment out the current `CMD` that runs `scripts/run_baseline.py`
   - Uncomment the `CMD` that runs `scripts/run_all_tasks.py`
3. Rebuild the image:

   ```bash
   docker build -t garbage-ai .
   ```

4. Run:

   ```bash
   docker run --rm garbage-ai
   ```

Alternative (no `Dockerfile` editing): you can also run the scoreboard directly with a command override:

```bash
docker run --rm garbage-ai python scripts/run_all_tasks.py
```

## OpenEnv validation

1. `pip install -r requirements.txt`
2. From repo root, run checks listed under `validation.smoke_tests` in **`openenv.yaml`**, for example:
   - `python -c "from env.environment import GarbageRoutingEnvironment; from tasks.easy_task import get_task_config; e=GarbageRoutingEnvironment(get_task_config()); e.reset(0); print('ok')"`
   - `python scripts/run_baseline.py --task easy --agent random --seed 0`

Success: JSON output with `"grader": {"score": ...}`.

## Example baseline output (random, easy)

Numbers vary with RNG and environment version; shape is:

```json
{
  "task": "easy",
  "agent": "random",
  "seed": 0,
  "cumulative_reward": 1409.67,
  "grader": {
    "score": 76.37,
    "breakdown": {
      "collection_ratio": 0.967,
      "fuel_score": 1.0,
      "time_score": 0.65,
      "penalty": 15.0,
      "total_mass_collected": 1450.23,
      "final_pending_garbage": 49.0,
      "total_fuel_used": 228.32
    }
  }
}
```

Re-run `python scripts/run_baseline.py --task easy --agent random --seed 0` on your machine for current metrics.

## Results I got


### Easy

**Command:** `python scripts/run_baseline.py --task easy --agent random --seed 0` or
**Command:** `docker run --rm garbage-ai python scripts/run_baseline.py --task easy`

```text
{
  "task": "easy",
  "agent": "random",
  "seed": 0,
  "cumulative_reward": 1409.6710583503896,
  "grader": {
    "score": 76.36583176025194,
    "breakdown": {
      "collection_ratio": 0.9673166352050386,
      "fuel_score": 1.0,
      "time_score": 0.65,
      "penalty": 15.0,
      "total_mass_collected": 1450.2336705451687,
      "final_pending_garbage": 49.0,
      "total_fuel_used": 228.32026751310474
    }
  },
  "summary": {
    "task_name": "easy",
    "steps_taken": 120,
    "max_steps": 120,
    "total_mass_collected": 1450.2336705451687,
    "total_fuel_used": 228.32026751310474,
    "total_garbage_generated_est": 1452.0,
    "final_pending_garbage": 49.0,
    "total_overflow_penalty_accumulated": 537.2679228798572,
    "invalid_actions": 0,
    "constraint_violations": 0
  }
}

```

### Medium

**Command:** `python scripts/run_baseline.py --task medium --agent random --seed 0` or
**Command:** `docker run --rm garbage-ai python scripts/run_baseline.py --task medium`


```text
{
  "task": "medium",
  "agent": "random",
  "seed": 0,
  "cumulative_reward": 3403.5781787112787,
  "grader": {
    "score": 66.34480481537486,
    "breakdown": {
      "collection_ratio": 0.884266766986976,
      "fuel_score": 1.0,
      "time_score": 0.55,
      "penalty": 20.0,
      "total_mass_collected": 5851.413340848027,
      "final_pending_garbage": 765.835614353229,
      "total_fuel_used": 393.3601362616054
    }
  },
  "summary": {
    "task_name": "medium",
    "steps_taken": 220,
    "max_steps": 220,
    "total_mass_collected": 5851.413340848027,
    "total_fuel_used": 393.3601362616054,
    "total_garbage_generated_est": 6545.0,
    "final_pending_garbage": 765.835614353229,
    "total_overflow_penalty_accumulated": 48445.335065594845,
    "invalid_actions": 0,
    "constraint_violations": 0
  }
}

```

### Hard

**Command:** `python scripts/run_baseline.py --task hard --agent random --seed 0` or
**Command:** `docker run --rm garbage-ai python scripts/run_baseline.py --task hard`


```text
{
  "task": "hard",
  "agent": "random",
  "seed": 0,
  "cumulative_reward": -83371.9725203798,
  "grader": {
    "score": 41.322194110884226,
    "breakdown": {
      "collection_ratio": 0.4960487580196496,
      "fuel_score": 1.0,
      "time_score": 0.44999999999999996,
      "penalty": 25.0,
      "total_mass_collected": 14135.994070414356,
      "final_pending_garbage": 14361.192630249348,
      "total_fuel_used": 1199.8245750309268,
      "constraint_violations": 0
    }
  },
  "summary": {
    "task_name": "hard",
    "steps_taken": 420,
    "max_steps": 420,
    "total_mass_collected": 14135.994070414356,
    "total_fuel_used": 1199.8245750309268,
    "total_garbage_generated_est": 28350.0,
    "final_pending_garbage": 14361.192630249348,
    "total_overflow_penalty_accumulated": 1948479.5774108416,
    "invalid_actions": 0,
    "constraint_violations": 0
  }
}

```

### Scoreboard (`run_all_tasks.py`)

**Command:** `python scripts/run_all_tasks.py` or
**Command:** `docker run --rm garbage-ai`


```text

Running task: easy
Score: 76.36583176025194

Running task: medium
Score: 66.34480481537486

Running task: hard
Score: 41.322194110884226

FINAL SCOREBOARD
EASY : 76.36583176025194
MEDIUM : 66.34480481537486
HARD : 41.322194110884226

```

## License

See `LICENSE` in this repository.
