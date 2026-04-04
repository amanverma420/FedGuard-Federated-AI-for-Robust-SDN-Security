import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logger import get_logger
import config

logger = get_logger("DataLoader")


def load_dataset():
    """
    Master loader. Falls back to synthetic data if real datasets not found.
    Returns: X_train, X_test, y_train, y_test
    """
    X, y = None, None

    if config.DATASET.lower() != "synthetic":
        try:
            logger.info(f"Attempting to load {config.DATASET} from {config.DATASET_PATH}")
            if config.DATASET.lower() == "nsl_kdd":
                X, y = _load_nsl_kdd(config.DATASET_PATH)
        except Exception as e:
            logger.warning(f"{e}. Falling back to synthetic data.")
            X, y = None, None

    if X is None:
        logger.info("Generating synthetic SDN traffic dataset (50,000 samples)...")
        from data.synthetic_generator import generate_dataset
        X, y = generate_dataset(n_total=50000, seed=config.SEED)
        logger.info(f"Synthetic dataset ready: {X.shape}")

    if X.shape[1] < config.NUM_FEATURES:
        pad = np.zeros((X.shape[0], config.NUM_FEATURES - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > config.NUM_FEATURES:
        X = X[:, :config.NUM_FEATURES]

    unique, counts = np.unique(y, return_counts=True)
    logger.info("Class distribution:")
    for u, c in zip(unique, counts):
        name = config.CLASS_NAMES[u] if u < len(config.CLASS_NAMES) else str(u)
        logger.info(f"  {name}: {c} ({100*c/len(y):.1f}%)")

    return train_test_split(X, y, test_size=config.TEST_SPLIT, random_state=config.SEED, stratify=y)


def _load_nsl_kdd(path: str):
    train_file = os.path.join(path, "KDDTrain+.csv")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"NSL-KDD not found at {train_file}")

    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]

    df = pd.read_csv(train_file, names=col_names)
    for col in ["protocol_type", "service", "flag"]:
        df[col] = pd.Categorical(df[col]).codes

    def map_label(lbl):
        lbl = str(lbl).lower()
        if lbl == "normal":
            return 0
        if lbl in ("neptune", "smurf", "pod", "teardrop", "land", "back", "udpstorm", "apache2"):
            return 1
        if lbl in ("satan", "ipsweep", "portsweep", "nmap", "mscan", "saint"):
            return 2
        if lbl in ("warezclient", "imap", "ftp_write", "guess_passwd", "multihop", "phf", "spy", "warezmaster"):
            return 3
        return 4

    df["label"] = df["label"].apply(map_label)
    X = df[col_names[:41]].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    return X, y