"""
data/data_loader.py
Loads CIC-IDS2018, NSL-KDD, UNSW-NB15, or synthetic data.
Falls back to synthetic if real datasets not found at DATASET_PATH.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logger import get_logger
import config

logger = get_logger("DataLoader")


def load_nsl_kdd(path: str):
    """
    Load NSL-KDD dataset from CSV.
    Expected columns: 41 features + label + difficulty
    """
    train_file = os.path.join(path, "KDDTrain+.csv")
    test_file  = os.path.join(path, "KDDTest+.csv")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"NSL-KDD not found at {train_file}")

    col_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
    ]

    df_train = pd.read_csv(train_file, names=col_names)
    df_test  = pd.read_csv(test_file,  names=col_names)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Map categorical features
    df["protocol_type"] = pd.Categorical(df["protocol_type"]).codes
    df["service"]       = pd.Categorical(df["service"]).codes
    df["flag"]          = pd.Categorical(df["flag"]).codes

    # Map labels to 5 classes
    def map_label(lbl):
        lbl = str(lbl).lower()
        if lbl == "normal": return 0
        if lbl in ("neptune","smurf","pod","teardrop","land","back",
                   "udpstorm","apache2","processtable","mailbomb"): return 1  # DoS
        if lbl in ("satan","ipsweep","portsweep","nmap","mscan","saint"): return 2  # Probe
        if lbl in ("warezclient","imap","ftp_write","guess_passwd",
                   "multihop","phf","spy","warezmaster","snmpguess",
                   "snmpgetattack","httptunnel","sendmail","named"): return 3  # R2L
        return 4  # U2R (anything else)

    df["label"] = df["label"].apply(map_label)

    feature_cols = col_names[:41]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    return X, y


def load_unsw_nb15(path: str):
    """Load UNSW-NB15 dataset. Maps to 5-class schema."""
    files = [os.path.join(path, f"UNSW-NB15_{i}.csv") for i in range(1, 5)]
    existing = [f for f in files if os.path.exists(f)]
    if not existing:
        raise FileNotFoundError(f"UNSW-NB15 not found at {path}")

    dfs = [pd.read_csv(f, low_memory=False) for f in existing]
    df = pd.concat(dfs, ignore_index=True)

    # Encode categoricals
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.Categorical(df[col]).codes

    # Use first 41 numeric columns as features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" in numeric_cols: numeric_cols.remove("label")
    feature_cols = numeric_cols[:41]

    X = df[feature_cols].values.astype(np.float32)
    label_col = "label" if "label" in df.columns else df.columns[-1]
    y = (df[label_col].values > 0).astype(np.int64)  # Binary: normal vs attack
    return X, y


def load_cicids2018(path: str):
    """Load CIC-IDS2018 CSV files. Each file is one day of traffic."""
    csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"CIC-IDS2018 files not found at {path}")

    dfs = []
    for csv in sorted(csvs)[:5]:  # Load up to 5 days
        try:
            df = pd.read_csv(os.path.join(path, csv), low_memory=False)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not read {csv}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    for col in df.select_dtypes(include="object").columns:
        if col.strip().lower() == "label":
            df[col] = df[col].apply(lambda x: 0 if str(x).strip().lower() == "benign" else 1)
        else:
            df[col] = pd.Categorical(df[col]).codes

    label_col = [c for c in df.columns if c.strip().lower() == "label"][0]
    feature_cols = [c for c in df.columns if c != label_col][:41]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)
    return X, y


def load_dataset():
    """
    Master loader. Tries real dataset first, falls back to synthetic.
    Returns: X_train, X_test, y_train, y_test
    """
    dataset = config.DATASET.lower()
    path = config.DATASET_PATH

    X, y = None, None

    if dataset != "synthetic":
        try:
            logger.info(f"Attempting to load {dataset} from {path}")
            if dataset == "nsl_kdd":
                X, y = load_nsl_kdd(path)
            elif dataset == "unsw_nb15":
                X, y = load_unsw_nb15(path)
            elif dataset == "cicids2018":
                X, y = load_cicids2018(path)
            logger.info(f"Loaded {dataset}: {X.shape[0]} samples, {X.shape[1]} features")
        except FileNotFoundError as e:
            logger.warning(f"{e}. Falling back to synthetic data.")
            X, y = None, None

    if X is None:
        logger.info("Generating synthetic SDN traffic dataset (50,000 samples)...")
        from data.synthetic_generator import generate_dataset
        X, y = generate_dataset(n_total=50000, seed=config.SEED)
        logger.info(f"Synthetic dataset ready: {X.shape}")

    # Ensure exactly NUM_FEATURES features
    if X.shape[1] < config.NUM_FEATURES:
        pad = np.zeros((X.shape[0], config.NUM_FEATURES - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    elif X.shape[1] > config.NUM_FEATURES:
        X = X[:, :config.NUM_FEATURES]

    # Log class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info("Class distribution:")
    for u, c in zip(unique, counts):
        name = config.CLASS_NAMES[u] if u < len(config.CLASS_NAMES) else str(u)
        logger.info(f"  {name}: {c} ({100*c/len(y):.1f}%)")

    return train_test_split(X, y, test_size=config.TEST_SPLIT,
                            random_state=config.SEED, stratify=y)
