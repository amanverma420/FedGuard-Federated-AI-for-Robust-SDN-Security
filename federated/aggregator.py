import numpy as np
from utils.logger import get_logger

logger = get_logger("Aggregator")

def fedavg(client_updates):
    total_samples = sum(u["n_samples"] for u in client_updates)
    n_layers = len(client_updates[0]["weights"])
    aggregated = []
    for layer_idx in range(n_layers):
        weighted_sum = sum(
            u["weights"][layer_idx] * (u["n_samples"] / total_samples)
            for u in client_updates
        )
        aggregated.append(weighted_sum)
    return aggregated


def krum(client_updates, num_byzantine=1):
    n = len(client_updates)
    f = num_byzantine
    if n <= 2 * f + 2:
        logger.warning(f"Krum: n={n} insufficient for f={f}. Using FedAvg.")
        return fedavg(client_updates)

    def flatten(u):
        return np.concatenate([w.flatten() for w in u["weights"]])

    flat = [flatten(u) for u in client_updates]
    scores = []
    for i in range(n):
        dists = sorted(float(np.sum((flat[i] - flat[j]) ** 2)) for j in range(n) if i != j)
        scores.append(sum(dists[:n - f - 2]))
    idx = int(np.argmin(scores))
    return client_updates[idx]["weights"]


def trimmed_mean(client_updates, trim_fraction=0.1):
    n = len(client_updates)
    n_trim = max(1, int(n * trim_fraction))
    n_layers = len(client_updates[0]["weights"])
    aggregated = []
    for layer_idx in range(n_layers):
        stacked = np.stack([u["weights"][layer_idx] for u in client_updates], axis=0)
        sorted_s = np.sort(stacked, axis=0)
        trimmed = sorted_s[n_trim: n - n_trim]
        if len(trimmed) == 0:
            trimmed = sorted_s
        aggregated.append(trimmed.mean(axis=0))
    return aggregated


def aggregate(client_updates, strategy=None):
    import config
    strategy = strategy or config.BYZANTINE_DEFENSE
    logger.info(f"Aggregating {len(client_updates)} updates | strategy='{strategy}'")
    if strategy == "krum":
        return krum(client_updates, num_byzantine=config.NUM_BYZANTINE)
    elif strategy == "trimmed_mean":
        return trimmed_mean(client_updates)
    else:
        return fedavg(client_updates)