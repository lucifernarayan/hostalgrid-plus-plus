# inference.py

import random
from env.hostelgrid_env import HostelGridEnv
from simulation.hostel import Hostel
from simulation.appliances import ApplianceManager
from simulation.grid import Grid

def random_agent(observation, action_count=6):
    """Simple random agent — baseline before we add real RL"""
    return random.randint(0, action_count - 1)

def run_episode():
    print("\n" + "="*50)
    print("🏨 HostelGrid++ Simulation Starting...")
    print("="*50)

    env = HostelGridEnv(num_rooms=20, episode_hours=24)
    hostel = Hostel(num_rooms=20)
    appliances = ApplianceManager()
    grid = Grid()

    appliances.turn_on_all()

    obs = env.reset()
    total_reward = 0
    total_cost = 0
    total_complaints = 0
    done = False

    while not done:
        # Agent picks action
        action = random_agent(obs)

        # Environment steps
        obs, reward, done, info = env.step(action)

        # Grid cost this hour
        hour_cost = grid.get_cost(info["power"], info["hour"])
        total_cost += hour_cost
        total_reward += reward
        total_complaints += info["complaints"]

        print(f"  Hour {info['hour']:02d} | "
              f"Action: {action} | "
              f"Power: {info['power']:.2f} kW | "
              f"Complaints: {info['complaints']} | "
              f"Cost: Rs {hour_cost:.2f} | "
              f"Reward: {reward:.4f}")

    print("\n" + "="*50)
    print("📊 Episode Summary")
    print(f"   Total Reward     : {total_reward:.4f}")
    print(f"   Total Cost       : Rs {total_cost:.2f}")
    print(f"   Total Complaints : {total_complaints}")
    print("="*50)

if __name__ == "__main__":
    run_episode()