# tasks/task_easy.py

import numpy as np
import random
from env.hostelgrid_env import HostelGridEnv

class QLearningAgent:
    def __init__(self, state_bins=10, action_count=6):
        self.action_count = action_count
        self.state_bins = state_bins
        self.q_table = {}
        self.alpha = 0.1       # learning rate
        self.gamma = 0.95      # discount factor
        self.epsilon = 1.0     # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def discretize(self, obs):
        """Convert continuous observation to discrete state key"""
        return tuple(np.round(obs, 1))

    def choose_action(self, obs):
        state = self.discretize(obs)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)  # explore
        if state not in self.q_table:
            return random.randint(0, self.action_count - 1)
        return int(np.argmax(self.q_table[state]))           # exploit

    def learn(self, obs, action, reward, next_obs):
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_count)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_count)

        # Q-learning update rule
        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state][action]
        )

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train(episodes=200):
    env = HostelGridEnv(num_rooms=20, episode_hours=24)
    agent = QLearningAgent()

    print("\n🧠 Training Q-Learning Agent...")
    print("="*50)

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1:03d} | "
                  f"Total Reward: {total_reward:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("="*50)
    print("✅ Training Complete!")
    return agent


if __name__ == "__main__":
    train(episodes=200)