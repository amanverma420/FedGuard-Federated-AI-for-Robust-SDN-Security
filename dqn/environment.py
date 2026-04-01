"""
dqn/environment.py - SDN Mitigation Environment for DQN.

Models the SDN network as a reinforcement learning environment.
State:  network traffic features + attack probability vector + env metrics
Action: one of 8 mitigation actions (block, rate-limit, reroute, etc.)
Reward: +10 correctly mitigated attack, +2 correctly passed normal,
        -5 false positive (blocked normal), -8 missed attack, -1 per step
"""
import numpy as np
import config
from utils.logger import get_logger

logger = get_logger("DQN-Env")

# Reward structure
REWARD_CORRECT_BLOCK   = +10.0   # Attack detected and mitigated
REWARD_CORRECT_PASS    = +2.0    # Normal traffic correctly passed
REWARD_FALSE_POSITIVE  = -5.0    # Normal traffic blocked (service disruption)
REWARD_MISSED_ATTACK   = -8.0    # Attack not caught (security breach)
REWARD_STEP_PENALTY    = -0.1    # Small penalty per step (encourage speed)

# Action effectiveness: probability that action successfully mitigates each attack type
# Rows = actions, Cols = attack classes (0=Normal,1=DoS,2=Probe,3=R2L,4=U2R)
ACTION_EFFECTIVENESS = np.array([
    # Normal  DoS    Probe  R2L    U2R
    [0.0,    0.95,  0.40,  0.30,  0.20],   # 0: block_ip
    [0.0,    0.90,  0.30,  0.20,  0.10],   # 1: rate_limit
    [0.0,    0.85,  0.60,  0.40,  0.30],   # 2: reroute_traffic
    [0.0,    0.70,  0.80,  0.70,  0.60],   # 3: honeypot_redirect
    [0.0,    0.10,  0.10,  0.10,  0.10],   # 4: alert_only
    [0.0,    0.80,  0.70,  0.80,  0.70],   # 5: quarantine_flow
    [0.0,    0.98,  0.50,  0.40,  0.30],   # 6: drop_packet
    [0.0,    0.92,  0.45,  0.35,  0.25],   # 7: null_route
], dtype=np.float32)

# Latency cost per action (milliseconds)
ACTION_LATENCY = np.array([50, 30, 120, 200, 5, 80, 20, 40], dtype=np.float32)


class SDNMitigationEnv:
    """
    Simulated SDN environment for DQN training.

    Episode flow:
      1. Reset: sample a traffic event (features + true label)
      2. Agent observes state (features + detection model probabilities)
      3. Agent selects mitigation action
      4. Environment computes reward based on action effectiveness
      5. Episode ends after MAX_STEPS or when attack is resolved
    """

    def __init__(self, X_data: np.ndarray, y_data: np.ndarray, detector=None):
        self.X_data   = X_data
        self.y_data   = y_data
        self.detector = detector
        self.n        = len(X_data)

        self.state_dim  = config.STATE_DIM
        self.action_dim = config.NUM_ACTIONS
        self.max_steps  = config.DQN_MAX_STEPS

        self.current_idx    = 0
        self.step_count     = 0
        self.episode_reward = 0.0
        self.latencies      = []

        logger.info(
            f"SDN Env | samples={self.n} | "
            f"state_dim={self.state_dim} | actions={self.action_dim}"
        )

    def _get_state(self, idx: int) -> np.ndarray:
        """
        Build state vector:
          [41 traffic features] + [5 attack probabilities from detector] + [4 env metrics]
        """
        features = self.X_data[idx].copy()  # (41,)

        # Get detector probability estimates
        if self.detector is not None:
            import torch
            with torch.no_grad():
                x_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                probs = self.detector.predict_proba(x_t).cpu().numpy()[0]  # (5,)
        else:
            # Random probs if no detector available yet
            probs = np.random.dirichlet(np.ones(config.NUM_CLASSES)).astype(np.float32)

        # Environment metrics: step_ratio, attack_rate, avg_latency, threat_level
        step_ratio   = self.step_count / self.max_steps
        attack_rate  = float(np.mean(self.y_data[:min(100, self.n)] != 0))
        avg_latency  = float(np.mean(self.latencies[-10:])) / 200.0 if self.latencies else 0.0
        threat_level = float(probs[1:].sum())  # sum of all attack probs

        env_metrics = np.array([step_ratio, attack_rate, avg_latency, threat_level], dtype=np.float32)

        state = np.concatenate([features, probs, env_metrics])  # (41+5+4=50)
        # Pad or truncate to exact state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        return state[:self.state_dim].astype(np.float32)

    def reset(self) -> np.ndarray:
        """Start new episode with a random traffic sample."""
        self.current_idx    = np.random.randint(0, self.n)
        self.step_count     = 0
        self.episode_reward = 0.0
        return self._get_state(self.current_idx)

    def step(self, action: int) -> tuple:
        """
        Execute action, compute reward, advance to next state.
        Returns: (next_state, reward, done, info)
        """
        true_label = int(self.y_data[self.current_idx])
        is_attack  = (true_label != 0)

        # Compute reward
        if is_attack:
            effectiveness = float(ACTION_EFFECTIVENESS[action, true_label])
            if np.random.random() < effectiveness:
                reward = REWARD_CORRECT_BLOCK
                mitigated = True
            else:
                reward = REWARD_MISSED_ATTACK
                mitigated = False
        else:
            # Normal traffic — actions other than alert_only cause disruption
            if action == 4:  # alert_only — no impact on normal traffic
                reward = REWARD_CORRECT_PASS
            else:
                # Probability of false positive depends on action aggressiveness
                fp_prob = ACTION_EFFECTIVENESS[action, 0] if action < len(ACTION_EFFECTIVENESS) else 0.1
                if np.random.random() < 0.2:  # 20% chance false positive
                    reward = REWARD_FALSE_POSITIVE
                else:
                    reward = REWARD_CORRECT_PASS
            mitigated = not is_attack

        # Step penalty + latency component
        latency = float(ACTION_LATENCY[action])
        self.latencies.append(latency)
        reward += REWARD_STEP_PENALTY - (latency / 1000.0)

        self.episode_reward += reward
        self.step_count     += 1

        # Move to next traffic sample
        self.current_idx = (self.current_idx + 1) % self.n
        next_state = self._get_state(self.current_idx)

        # Episode ends after max_steps or if attack resolved
        done = (self.step_count >= self.max_steps) or (is_attack and mitigated)

        info = {
            "true_label":  true_label,
            "action_name": config.MITIGATION_ACTIONS[action],
            "latency_ms":  latency,
            "mitigated":   mitigated,
        }
        return next_state, reward, done, info
