"""
simulation/attack_simulator.py - Simulates attack traffic events in the SDN.
"""
import numpy as np
from utils.logger import get_logger
import config

logger = get_logger("AttackSim")


class AttackSimulator:
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        self.n      = len(X_data)
        self.ptr    = 0
        logger.info(f"AttackSimulator loaded {self.n} traffic samples.")

    def next_event(self):
        idx = self.ptr % self.n
        self.ptr += 1
        label = int(self.y_data[idx])
        return {
            "features":   self.X_data[idx],
            "label":      label,
            "label_name": config.CLASS_NAMES[label] if label < len(config.CLASS_NAMES) else "Unknown",
            "src_ip":     f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
            "switch":     f"sw{np.random.randint(0, config.NUM_SWITCHES):02d}",
            "timestamp":  self.ptr * 0.001,
        }

    def simulate_batch(self, n_events=1000):
        events = [self.next_event() for _ in range(n_events)]
        labels = [e["label"] for e in events]
        attack_rate = sum(l != 0 for l in labels) / len(labels)
        logger.info(f"Simulated {n_events} events | attack_rate={attack_rate:.2%}")
        return events
