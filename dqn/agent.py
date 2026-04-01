"""
dqn/agent.py - Deep Q-Network agent for autonomous SDN mitigation.

Architecture: DQN with target network + experience replay + epsilon-greedy exploration.
  Input:  state_dim (50)
  Hidden: 256 -> 128
  Output: num_actions (8) Q-values

Two networks:
  - Online network:  updated every step
  - Target network:  updated every TARGET_UPDATE episodes (stabilizes training)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dqn.replay_buffer import ReplayBuffer
from utils.logger import get_logger
import config

logger = get_logger("DQN-Agent")


class QNetwork(nn.Module):
    def __init__(self, state_dim=config.STATE_DIM, action_dim=config.NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    DQN agent that learns optimal mitigation policies.
    Selects actions: block_ip, rate_limit, reroute, honeypot, alert, quarantine, drop, null_route
    """

    def __init__(self):
        self.device       = config.DEVICE
        self.online_net   = QNetwork().to(self.device)
        self.target_net   = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.Adam(self.online_net.parameters(), lr=config.DQN_LR)
        self.replay       = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
        self.epsilon      = config.DQN_EPSILON_START
        self.gamma        = config.DQN_GAMMA
        self.batch_size   = config.DQN_BATCH_SIZE
        self.episode      = 0
        self.losses       = []
        self.rewards      = []

        logger.info(
            f"DQN Agent | state={config.STATE_DIM} | actions={config.NUM_ACTIONS} | "
            f"params={sum(p.numel() for p in self.online_net.parameters()):,}"
        )

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(config.NUM_ACTIONS)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        """Sample a batch and update online network via Q-learning."""
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t      = torch.tensor(states,      dtype=torch.float32).to(self.device)
        actions_t     = torch.tensor(actions,     dtype=torch.long   ).to(self.device)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t       = torch.tensor(dones,       dtype=torch.float32).to(self.device)

        # Current Q values for chosen actions
        q_current = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values: r + gamma * max_a Q_target(s', a) * (1 - done)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = nn.SmoothL1Loss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(config.DQN_EPSILON_END,
                           self.epsilon * config.DQN_EPSILON_DECAY)

    def train(self, env) -> list:
        """
        Full DQN training loop.
        Returns episode reward history.
        """
        logger.info(f"DQN training | episodes={config.DQN_EPISODES} | epsilon={self.epsilon:.2f}")
        episode_rewards = []

        for ep in range(config.DQN_EPISODES):
            state = env.reset()
            total_reward = 0.0
            ep_losses    = []

            for step in range(config.DQN_MAX_STEPS):
                action     = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.store(state, action, reward, next_state, done)
                loss = self.learn()
                if loss > 0:
                    ep_losses.append(loss)
                state        = next_state
                total_reward += reward
                if done:
                    break

            self.decay_epsilon()
            if (ep + 1) % config.DQN_TARGET_UPDATE == 0:
                self.update_target()

            episode_rewards.append(total_reward)
            self.rewards.append(total_reward)

            if (ep + 1) % 20 == 0:
                avg_r   = np.mean(episode_rewards[-20:])
                avg_l   = np.mean(ep_losses) if ep_losses else 0.0
                logger.info(
                    f"  DQN Ep {ep+1}/{config.DQN_EPISODES} | "
                    f"AvgReward={avg_r:.2f} | Loss={avg_l:.4f} | eps={self.epsilon:.3f}"
                )

        logger.info(f"DQN training complete. Final avg reward: {np.mean(episode_rewards[-20:]):.2f}")
        return episode_rewards

    def evaluate_mitigation(self, env, n_episodes=50) -> dict:
        """
        Evaluate agent's mitigation performance.
        Returns metrics: accuracy, avg_latency, mitigation_rate.
        """
        self.online_net.eval()
        old_epsilon   = self.epsilon
        self.epsilon  = 0.0  # Greedy during evaluation

        correct, total = 0, 0
        latencies, rewards = [], []

        for _ in range(n_episodes):
            state = env.reset()
            for _ in range(config.DQN_MAX_STEPS):
                action = self.select_action(state)
                state, reward, done, info = env.step(action)
                latencies.append(info["latency_ms"])
                rewards.append(reward)
                total   += 1
                correct += int(info["mitigated"])
                if done:
                    break

        self.epsilon = old_epsilon
        self.online_net.train()

        return {
            "mitigation_accuracy": correct / max(total, 1),
            "avg_latency_ms":      float(np.mean(latencies)),
            "avg_reward":          float(np.mean(rewards)),
            "total_actions":       total,
        }
