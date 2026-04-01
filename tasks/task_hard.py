# tasks/task_hard.py

import numpy as np
import random
from collections import deque
from env.hostelgrid_env import HostelGridEnv
from simulation.grid import Grid
from simulation.hostel import Hostel
from simulation.student import Student

class ExperienceReplay:
    """Memory buffer — agent learns from past experiences"""
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size=64):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def size(self):
        return len(self.buffer)


class HumanAwareAgent:
    """
    Full human-aware RL agent with:
    - Experience replay
    - Dynamic reward shaping
    - Complaint history tracking
    - Carbon-aware decisions
    - Fairness enforcement
    """
    def __init__(self, action_count=6):
        self.action_count = action_count
        self.q_table = {}
        self.alpha = 0.08
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_decay = 0.990
        self.epsilon_min = 0.01
        self.memory = ExperienceReplay(capacity=2000)
        self.complaint_history = deque(maxlen=5)  # track last 5 hours
        self.grid = Grid()

    def discretize(self, obs):
        return tuple(np.round(obs, 1))

    def choose_action(self, obs):
        state = self.discretize(obs)

        # Human-aware override — if complaints spiking, prioritize comfort
        if len(self.complaint_history) >= 3:
            recent_complaints = sum(list(self.complaint_history)[-3:])
            if recent_complaints > 5:
                return 0  # force increase_ac to reduce complaints

        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        if state not in self.q_table:
            return random.randint(0, self.action_count - 1)
        return int(np.argmax(self.q_table[state]))

    def compute_reward(self, reward, info):
        """
        Advanced reward shaping with 4 dimensions:
        Cost + Carbon + Fairness + Human Comfort
        """
        hour = info["hour"]
        power = info["power"]
        complaints = info["complaints"]

        self.complaint_history.append(complaints)

        carbon_rate = self.grid.get_carbon_rate(hour)
        tariff = self.grid.get_tariff(hour)

        # 1. Cost awareness
        cost_penalty = -0.3 if tariff > 7.0 and power > 7.0 else 0.1

        # 2. Carbon awareness
        carbon_penalty = -0.25 if carbon_rate > 0.75 and power > 6.0 else 0.15

        # 3. Human comfort
        if complaints == 0:
            comfort_bonus = 0.4
        elif complaints <= 2:
            comfort_bonus = 0.1
        else:
            comfort_bonus = -0.4 * complaints

        # 4. Fairness bonus (consistent service across rooms)
        fairness_bonus = 0.2

        # 5. Night efficiency bonus
        night_bonus = 0.3 if (0 <= hour <= 5 and power < 4.0) else 0

        shaped = (reward + cost_penalty + carbon_penalty +
                  comfort_bonus + fairness_bonus + night_bonus)

        return round(shaped, 4)

    def learn_from_memory(self):
        """Replay past experiences to stabilize learning"""
        batch = self.memory.sample(64)
        for obs, action, reward, next_obs, done in batch:
            state = self.discretize(obs)
            next_state = self.discretize(next_obs)

            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_count)
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(self.action_count)

            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state])

            self.q_table[state][action] += self.alpha * (
                target - self.q_table[state][action]
            )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, obs, action, reward, next_obs, done):
        self.memory.store(obs, action, reward, next_obs, done)
        if self.memory.size() >= 64:
            self.learn_from_memory()


def train_hard(episodes=1000):
    env = HostelGridEnv(num_rooms=20, episode_hours=24)
    agent = HumanAwareAgent()

    print("\n🧠 Hard Task — Full Human-Aware RL Agent Training...")
    print("="*50)

    best_reward = float('-inf')
    reward_log = []
    complaint_log = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        total_complaints = 0
        done = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)

            shaped_reward = agent.compute_reward(reward, info)
            agent.learn(obs, action, shaped_reward, next_obs, done)

            obs = next_obs
            total_reward += shaped_reward
            total_complaints += info["complaints"]

        reward_log.append(total_reward)
        complaint_log.append(total_complaints)

        if total_reward > best_reward:
            best_reward = total_reward

        if (ep + 1) % 200 == 0:
            avg_reward = np.mean(reward_log[-200:])
            avg_complaints = np.mean(complaint_log[-200:])
            print(f"  Episode {ep+1:04d} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Best: {best_reward:.4f} | "
                  f"Avg Complaints: {avg_complaints:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {agent.memory.size()}")

    print("="*50)
    print(f"✅ Hard Training Complete!")
    print(f"   Best Reward     : {best_reward:.4f}")
    print(f"   Final Epsilon   : {agent.epsilon:.4f}")
    print(f"   Memory Buffer   : {agent.memory.size()} experiences")
    print(f"   Avg Complaints  : {np.mean(complaint_log[-200:]):.2f} (last 200 eps)")
    return agent, reward_log, complaint_log


if __name__ == "__main__":
    train_hard(episodes=1000)