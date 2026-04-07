# inference.py
# HostelGrid++ — OpenEnv baseline inference script
# Uses OpenAI client with [START][STEP][END] structured logging

import os
import random
import time
from openai import OpenAI
from env.openenv_api import HostelGridOpenEnv, Action

# ── API Configuration ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

client = OpenAI(
    api_key  = HF_TOKEN if HF_TOKEN else os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    base_url = API_BASE_URL,
)

TASKS = ["task_easy", "task_medium", "task_hard"]


def get_action_from_llm(observation: dict, task_id: str, step: int) -> int:
    """Ask LLM to choose an action given current observation."""
    prompt = f"""You are an energy management AI for a hostel.
Current situation (Hour {step}/24):
- Power usage: {observation['power_usage']:.2f} kW
- Avg temperature: {observation['avg_temperature']:.1f}°C
- Occupancy: {observation['avg_occupancy']:.1%}
- Complaints: {observation['complaint_level']}
- Time of day: {observation['time_of_day']}:00
- Carbon rate: {observation['carbon_rate']:.2f} gCO2/kWh
- Current cost: Rs {observation['current_cost']:.2f}
- Task: {task_id}

Choose ONE action (respond with just the number 0-5):
0: increase_ac (comfort up, cost up)
1: decrease_ac (save energy, complaints may rise)
2: lights_off_empty (save energy silently)
3: lights_on (restore lights)
4: defer_heavy_load (shift appliances to off-peak)
5: do_nothing

Respond with only a single digit 0-5."""

    try:
        response = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 5,
            temperature = 0.0,
        )
        text   = response.choices[0].message.content.strip()
        action = int(text[0])
        if action not in range(6):
            action = random.randint(0, 5)
    except Exception:
        action = random.randint(0, 5)

    return action


def run_task(task_id: str) -> float:
    """Run one full episode for a task and return score 0.0-1.0."""

    env = HostelGridOpenEnv(task_id=task_id, num_rooms=20)
    obs = env.reset()

    # ── [START] log ───────────────────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    done  = False
    step  = 0
    score = 0.0

    while not done:
        step += 1

        # Get action from LLM
        action_id = get_action_from_llm(obs.model_dump(), task_id, step)
        action    = Action(action_id=action_id)

        # Step environment
        next_obs, reward, done, info = env.step(action)

        # ── [STEP] log ────────────────────────────────────────
        print(f"[STEP] step={step} reward={reward.value:.4f}", flush=True)

        obs = next_obs

    score = env.score()

    # ── [END] log ─────────────────────────────────────────────
    print(f"[END] task={task_id} score={score:.4f} steps={step}", flush=True)

    return score


def main():
    """Run all 3 tasks and report final scores."""

    scores = {}
    for task_id in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"Running {task_id}...", flush=True)
        print(f"{'='*50}", flush=True)
        score = run_task(task_id)
        scores[task_id] = score
        print(f"\n✅ {task_id} Score: {score:.4f}", flush=True)
        time.sleep(1)

    avg = sum(scores.values()) / len(scores)

    print(f"\n{'='*50}", flush=True)
    print(f"📊 Final Scores:", flush=True)
    for t, s in scores.items():
        print(f"   {t}: {s:.4f}", flush=True)
    print(f"   Average: {avg:.4f}", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()