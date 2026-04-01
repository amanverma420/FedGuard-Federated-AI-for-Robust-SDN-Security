"""
dqn/replay_buffer.py - Experience replay buffer for DQN agent.
Stores (state, action, reward, next_state, done) transitions.
Random sampling breaks temporal correlations for stable training.
"""
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
