"""
simulation/sdn_simulator.py - Simulated SDN topology.
Represents a network of switches and hosts managed by SDN controllers.
Used to generate realistic traffic flow metadata for testing.
"""
import numpy as np
import random
from utils.logger import get_logger
import config

logger = get_logger("SDN-Sim")


class SDNTopology:
    """
    Simple fat-tree-inspired SDN topology simulation.
    Generates flow table entries and traffic metadata.
    """

    def __init__(self, n_switches=config.NUM_SWITCHES, n_hosts=config.NUM_HOSTS):
        self.n_switches = n_switches
        self.n_hosts    = n_hosts
        self.switches   = [f"sw{i:02d}" for i in range(n_switches)]
        self.hosts      = [f"10.0.{i//10}.{i%10+1}" for i in range(n_hosts)]
        self.flow_table = {}  # switch -> list of flow rules
        self._build_topology()
        logger.info(f"SDN Topology: {n_switches} switches, {n_hosts} hosts")

    def _build_topology(self):
        """Build simple mesh topology with flow rules."""
        for sw in self.switches:
            self.flow_table[sw] = []
            for i in range(random.randint(5, 20)):
                src = random.choice(self.hosts)
                dst = random.choice(self.hosts)
                self.flow_table[sw].append({
                    "src": src, "dst": dst,
                    "priority": random.randint(1, 100),
                    "action": "forward",
                    "packet_count": random.randint(0, 10000),
                })

    def get_controller_zones(self, n_controllers):
        """Partition switches among controllers (one zone per controller)."""
        zones = {}
        sw_per_ctrl = max(1, self.n_switches // n_controllers)
        for i in range(n_controllers):
            start = i * sw_per_ctrl
            end   = start + sw_per_ctrl if i < n_controllers - 1 else self.n_switches
            zones[i] = self.switches[start:end]
        return zones

    def install_mitigation(self, switch: str, action: str, target_ip: str):
        """Simulate installing a mitigation flow rule."""
        rule = {"src": target_ip, "action": action, "priority": 1000, "packet_count": 0}
        if switch in self.flow_table:
            self.flow_table[switch].insert(0, rule)
        logger.debug(f"Installed {action} rule on {switch} for {target_ip}")
        return True

    def get_stats(self):
        total_flows = sum(len(flows) for flows in self.flow_table.values())
        return {"switches": self.n_switches, "hosts": self.n_hosts, "total_flows": total_flows}
