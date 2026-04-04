"""
data/synthetic_generator.py
Generates realistic synthetic network traffic mimicking NSL-KDD's 41-feature schema.
Labels: 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
"""
import numpy as np
from utils.logger import get_logger

logger = get_logger("SyntheticGen")

PROFILES = {
    0: {  # Normal
        "duration": (5.0, 10.0), "src_bytes": (2000, 5000), "dst_bytes": (1500, 4000),
        "land": (0.0, 0.01), "wrong_fragment": (0.0, 0.1), "urgent": (0.0, 0.05),
        "hot": (1.0, 2.0), "num_failed_logins": (0.0, 0.1), "logged_in": (0.8, 0.2),
        "num_compromised": (0.0, 0.3), "root_shell": (0.0, 0.05), "su_attempted": (0.0, 0.02),
        "num_root": (0.0, 0.1), "num_file_creations": (0.1, 0.5), "num_shells": (0.0, 0.1),
        "num_access_files": (0.0, 0.2), "num_outbound_cmds": (0.0, 0.0),
        "is_host_login": (0.0, 0.0), "is_guest_login": (0.0, 0.05),
        "count": (20.0, 30.0), "srv_count": (18.0, 28.0),
        "serror_rate": (0.0, 0.05), "srv_serror_rate": (0.0, 0.05),
        "rerror_rate": (0.05, 0.1), "srv_rerror_rate": (0.05, 0.1),
        "same_srv_rate": (0.9, 0.1), "diff_srv_rate": (0.05, 0.1),
        "srv_diff_host_rate": (0.1, 0.15), "dst_host_count": (200, 80),
        "dst_host_srv_count": (180, 70), "dst_host_same_srv_rate": (0.85, 0.1),
        "dst_host_diff_srv_rate": (0.05, 0.08), "dst_host_same_src_port_rate": (0.1, 0.15),
        "dst_host_srv_diff_host_rate": (0.05, 0.08), "dst_host_serror_rate": (0.0, 0.05),
        "dst_host_srv_serror_rate": (0.0, 0.05), "dst_host_rerror_rate": (0.05, 0.08),
        "dst_host_srv_rerror_rate": (0.05, 0.08),
        "protocol_type": (1.0, 0.8), "service": (10.0, 8.0), "flag": (5.0, 2.0),
    },
    1: {  # DoS
        "duration": (0.1, 0.5), "src_bytes": (500, 200), "dst_bytes": (0, 100),
        "land": (0.1, 0.2), "wrong_fragment": (0.3, 0.5), "urgent": (0.0, 0.1),
        "hot": (0.0, 0.1), "num_failed_logins": (0.0, 0.1), "logged_in": (0.0, 0.1),
        "num_compromised": (0.0, 0.0), "root_shell": (0.0, 0.0), "su_attempted": (0.0, 0.0),
        "num_root": (0.0, 0.0), "num_file_creations": (0.0, 0.0), "num_shells": (0.0, 0.0),
        "num_access_files": (0.0, 0.0), "num_outbound_cmds": (0.0, 0.0),
        "is_host_login": (0.0, 0.0), "is_guest_login": (0.0, 0.0),
        "count": (400, 200), "srv_count": (400, 200),
        "serror_rate": (0.9, 0.1), "srv_serror_rate": (0.9, 0.1),
        "rerror_rate": (0.0, 0.05), "srv_rerror_rate": (0.0, 0.05),
        "same_srv_rate": (1.0, 0.05), "diff_srv_rate": (0.0, 0.02),
        "srv_diff_host_rate": (0.0, 0.02), "dst_host_count": (255, 5),
        "dst_host_srv_count": (255, 5), "dst_host_same_srv_rate": (1.0, 0.02),
        "dst_host_diff_srv_rate": (0.0, 0.02), "dst_host_same_src_port_rate": (1.0, 0.02),
        "dst_host_srv_diff_host_rate": (0.0, 0.02), "dst_host_serror_rate": (0.9, 0.1),
        "dst_host_srv_serror_rate": (0.9, 0.1), "dst_host_rerror_rate": (0.0, 0.02),
        "dst_host_srv_rerror_rate": (0.0, 0.02),
        "protocol_type": (2.0, 0.5), "service": (2.0, 3.0), "flag": (1.0, 1.0),
    },
    2: {  # Probe
        "duration": (0.5, 1.0), "src_bytes": (100, 200), "dst_bytes": (50, 100),
        "land": (0.0, 0.0), "wrong_fragment": (0.0, 0.0), "urgent": (0.0, 0.0),
        "hot": (0.0, 0.5), "num_failed_logins": (0.0, 0.0), "logged_in": (0.0, 0.1),
        "num_compromised": (0.0, 0.0), "root_shell": (0.0, 0.0), "su_attempted": (0.0, 0.0),
        "num_root": (0.0, 0.0), "num_file_creations": (0.0, 0.0), "num_shells": (0.0, 0.0),
        "num_access_files": (0.0, 0.0), "num_outbound_cmds": (0.0, 0.0),
        "is_host_login": (0.0, 0.0), "is_guest_login": (0.0, 0.0),
        "count": (10, 20), "srv_count": (5, 10),
        "serror_rate": (0.3, 0.3), "srv_serror_rate": (0.3, 0.3),
        "rerror_rate": (0.3, 0.3), "srv_rerror_rate": (0.3, 0.3),
        "same_srv_rate": (0.1, 0.2), "diff_srv_rate": (0.7, 0.2),
        "srv_diff_host_rate": (0.5, 0.3), "dst_host_count": (50, 80),
        "dst_host_srv_count": (20, 40), "dst_host_same_srv_rate": (0.1, 0.15),
        "dst_host_diff_srv_rate": (0.7, 0.2), "dst_host_same_src_port_rate": (0.1, 0.15),
        "dst_host_srv_diff_host_rate": (0.5, 0.3), "dst_host_serror_rate": (0.3, 0.3),
        "dst_host_srv_serror_rate": (0.3, 0.3), "dst_host_rerror_rate": (0.3, 0.3),
        "dst_host_srv_rerror_rate": (0.3, 0.3),
        "protocol_type": (0.5, 0.8), "service": (20.0, 15.0), "flag": (3.0, 2.0),
    },
    3: {  # R2L
        "duration": (50.0, 80.0), "src_bytes": (1000, 2000), "dst_bytes": (800, 1500),
        "land": (0.0, 0.0), "wrong_fragment": (0.0, 0.1), "urgent": (0.0, 0.0),
        "hot": (3.0, 4.0), "num_failed_logins": (3.0, 3.0), "logged_in": (0.3, 0.4),
        "num_compromised": (0.5, 1.0), "root_shell": (0.0, 0.1), "su_attempted": (0.1, 0.3),
        "num_root": (0.0, 0.2), "num_file_creations": (0.0, 0.3), "num_shells": (0.0, 0.2),
        "num_access_files": (0.5, 1.0), "num_outbound_cmds": (0.0, 0.0),
        "is_host_login": (0.0, 0.0), "is_guest_login": (0.5, 0.4),
        "count": (5, 8), "srv_count": (5, 8),
        "serror_rate": (0.1, 0.2), "srv_serror_rate": (0.1, 0.2),
        "rerror_rate": (0.2, 0.3), "srv_rerror_rate": (0.2, 0.3),
        "same_srv_rate": (0.7, 0.2), "diff_srv_rate": (0.1, 0.15),
        "srv_diff_host_rate": (0.05, 0.1), "dst_host_count": (30, 50),
        "dst_host_srv_count": (25, 45), "dst_host_same_srv_rate": (0.65, 0.2),
        "dst_host_diff_srv_rate": (0.1, 0.15), "dst_host_same_src_port_rate": (0.05, 0.1),
        "dst_host_srv_diff_host_rate": (0.05, 0.08), "dst_host_serror_rate": (0.1, 0.2),
        "dst_host_srv_serror_rate": (0.1, 0.2), "dst_host_rerror_rate": (0.2, 0.3),
        "dst_host_srv_rerror_rate": (0.2, 0.3),
        "protocol_type": (1.5, 0.5), "service": (15.0, 10.0), "flag": (6.0, 2.0),
    },
    4: {  # U2R
        "duration": (100.0, 120.0), "src_bytes": (3000, 5000), "dst_bytes": (2000, 4000),
        "land": (0.0, 0.0), "wrong_fragment": (0.0, 0.1), "urgent": (0.2, 0.4),
        "hot": (10.0, 8.0), "num_failed_logins": (1.0, 2.0), "logged_in": (0.9, 0.1),
        "num_compromised": (5.0, 8.0), "root_shell": (0.8, 0.3), "su_attempted": (0.7, 0.4),
        "num_root": (3.0, 5.0), "num_file_creations": (2.0, 3.0), "num_shells": (2.0, 2.0),
        "num_access_files": (3.0, 4.0), "num_outbound_cmds": (0.0, 0.0),
        "is_host_login": (0.0, 0.0), "is_guest_login": (0.0, 0.0),
        "count": (3, 5), "srv_count": (3, 5),
        "serror_rate": (0.05, 0.1), "srv_serror_rate": (0.05, 0.1),
        "rerror_rate": (0.05, 0.1), "srv_rerror_rate": (0.05, 0.1),
        "same_srv_rate": (0.8, 0.15), "diff_srv_rate": (0.05, 0.08),
        "srv_diff_host_rate": (0.02, 0.05), "dst_host_count": (10, 15),
        "dst_host_srv_count": (10, 15), "dst_host_same_srv_rate": (0.75, 0.15),
        "dst_host_diff_srv_rate": (0.05, 0.08), "dst_host_same_src_port_rate": (0.02, 0.05),
        "dst_host_srv_diff_host_rate": (0.02, 0.04), "dst_host_serror_rate": (0.05, 0.1),
        "dst_host_srv_serror_rate": (0.05, 0.1), "dst_host_rerror_rate": (0.05, 0.1),
        "dst_host_srv_rerror_rate": (0.05, 0.1),
        "protocol_type": (1.0, 0.5), "service": (8.0, 5.0), "flag": (5.0, 1.0),
    },
}

FEATURE_NAMES = list(PROFILES[0].keys())


def generate_class_samples(class_id: int, n_samples: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed + class_id)
    profile = PROFILES[class_id]
    samples = []
    for feat in FEATURE_NAMES:
        mu, sigma = profile[feat]
        col = rng.normal(mu, sigma + 1e-8, n_samples)
        if "rate" in feat or feat in ("land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"):
            col = np.clip(col, 0.0, 1.0)
        col = np.abs(col)
        samples.append(col)
    return np.column_stack(samples)


def generate_dataset(n_total: int = 50000, class_weights: list = None, seed: int = 42) -> tuple:
    if class_weights is None:
        class_weights = [0.60, 0.20, 0.10, 0.06, 0.04]

    assert abs(sum(class_weights) - 1.0) < 1e-6
    assert len(class_weights) == 5

    X_parts, y_parts = [], []
    for cls, w in enumerate(class_weights):
        n = int(n_total * w)
        X_parts.append(generate_class_samples(cls, n, seed=seed))
        y_parts.append(np.full(n, cls, dtype=np.int64))

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def split_for_clients(X, y, n_clients: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    from numpy.random import dirichlet
    class_indices = [np.where(y == c)[0] for c in range(5)]
    client_indices = [[] for _ in range(n_clients)]
    for cls_idx in class_indices:
        rng.shuffle(cls_idx)
        proportions = dirichlet([0.5] * n_clients)
        splits = (proportions * len(cls_idx)).astype(int)
        splits[-1] = len(cls_idx) - splits[:-1].sum()
        pos = 0
        for c, s in enumerate(splits):
            client_indices[c].extend(cls_idx[pos:pos + s].tolist())
            pos += s

    client_data = []
    for c_idx in client_indices:
        c_idx = np.array(c_idx)
        rng.shuffle(c_idx)
        client_data.append((X[c_idx], y[c_idx]))
    return client_data