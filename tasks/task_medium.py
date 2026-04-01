# tasks/task_medium.py

import numpy as np
import random
from env.hostelgrid_env import HostelGridEnv
from simulation.grid import Grid

class MultiObjectiveAgent:
    def __init__(self, action_count=6):
        self.action_count = action_count
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.993
        self.epsilon_min = 0.01

        # Multi-objective weights (this is what makes it MEDIUM difficulty)
        self.w_cost     = 0.35
        self.w_comfort  = 0.30
        self.w_carbon   = 0.20
        self.w_fairness = 0.15

    def discretize(self, obs):
        return tuple(np.round(obs, 1))

    def choose_action(self, obs):
        state = self.discretize(obs)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        if state not in self.q_table:
            return random.randint(0, self.action_count - 1)
        return int(np.argmax(self.q_table[state]))

    def compute_shaped_reward(self, reward, info, hour):
        """
        Reward shaping — agent gets extra signal based on:
        - Time of day (peak vs off-peak awareness)
        - Complaint level (human comfort priority)
        - Carbon rate (green energy awareness)
        """
        grid = Grid()
        carbon_rate = grid.get_carbon_rate(hour)
        tariff = grid.get_tariff(hour)

        # Penalize heavily during peak hours if power is high
        peak_penalty = 0
        if 9 <= hour <= 12 or 18 <= hour <= 22:
            if info["power"] > 8.0:
                peak_penalty = -0.5

        # Bonus for keeping complaints at zero
        comfort_bonus = 0.3 if info["complaints"] == 0 else -0.2

        # Carbon bonus during high carbon hours if power is low
        carbon_bonus = 0.2 if (carbon_rate > 0.7 and info["power"] < 5.0) else 0

        shaped = reward + peak_penalty + comfort_bonus + carbon_bonus
        return round(shaped, 4)

    def learn(self, obs, action, reward, next_obs):
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_count)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_count)

        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state][action]
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_medium(episodes=500):
    env = HostelGridEnv(num_rooms=20, episode_hours=24)
    agent = MultiObjectiveAgent()

    print("\n🧠 Medium Task — Multi-Objective Agent Training...")
    print("="*50)

    best_reward = float('-inf')
    reward_log = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        total_complaints = 0
        total_cost = 0
        done = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)

            # Apply reward shaping
            shaped_reward = agent.compute_shaped_reward(
                reward, info, info["hour"]
            )

            agent.learn(obs, action, shaped_reward, next_obs)
            obs = next_obs
            total_reward += shaped_reward
            total_complaints += info["complaints"]
            total_cost += info["cost"]

        reward_log.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward

        if (ep + 1) % 100 == 0:
            avg = np.mean(reward_log[-100:])
            print(f"  Episode {ep+1:03d} | "
                  f"Avg Reward: {avg:.4f} | "
                  f"Best: {best_reward:.4f} | "
                  f"Complaints: {total_complaints} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("="*50)
    print(f"✅ Medium Training Complete! Best Reward: {best_reward:.4f}")
    return agent, reward_log


if __name__ == "__main__":
    train_medium(episodes=500)