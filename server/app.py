from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from env.environment import GarbageRoutingEnvironment
from env.models import Action
from tasks.easy_task import get_task_config as get_easy_task_config
from tasks.medium_task import get_task_config as get_medium_task_config
from tasks.hard_task import get_task_config as get_hard_task_config

app = FastAPI()

env = None

TASKS = {
    "easy": get_easy_task_config,
    "medium": get_medium_task_config,
    "hard": get_hard_task_config,
}


class StepAction(BaseModel):
    truck_id: int
    neighborhood_id: int


@app.post("/reset")
def reset(task: str = "medium"):
    global env

    cfg = TASKS[task]()
    env = GarbageRoutingEnvironment(cfg)

    obs = env.reset(seed=0)

    return {
        "status": "ok",
        "task": task,
        "observation": obs.model_dump(),
    }


@app.post("/step")
def step(action: StepAction):
    global env

    obs, reward, done, info = env.step(
        Action(
            truck_id=action.truck_id,
            neighborhood_id=action.neighborhood_id,
        )
    )

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
    }


@app.get("/state")
def state():
    global env

    if env is None:
        return {"error": "Environment not initialized"}

    return env.state()
    
@app.get("/")
def home():
    return {"message": "Garbage Routing OpenEnv API is running"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
