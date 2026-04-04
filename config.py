"""
config.py - Global configuration for FedGuard
"""
import torch

# ─── Dataset ───────────────────────────────────────────────
DATASET       = "synthetic"
DATASET_PATH  = "./datasets/"
NUM_FEATURES  = 41
NUM_CLASSES   = 5
CLASS_NAMES   = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# ─── Federated Learning ────────────────────────────────────
NUM_CLIENTS      = 5
FL_ROUNDS        = 30
LOCAL_EPOCHS     = 5
LOCAL_BATCH_SIZE = 128
LOCAL_LR         = 0.01
FL_FRACTION      = 1.0

BYZANTINE_DEFENSE = "fedavg"
NUM_BYZANTINE     = 0

# ─── Detection Model ───────────────────────────────────────
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT_RATE = 0.3

# ─── GAN ───────────────────────────────────────────────────
GAN_EPOCHS     = 30
GAN_BATCH_SIZE = 128
GAN_LATENT_DIM = 64
GAN_LR_G       = 0.0002
GAN_LR_D       = 0.0002
ADVERSARIAL_RATIO = 0.3

# ─── DQN ───────────────────────────────────────────────────
DQN_EPISODES      = 200
DQN_MAX_STEPS     = 50
DQN_BATCH_SIZE    = 64
DQN_LR            = 0.001
DQN_GAMMA         = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_END   = 0.01
DQN_EPSILON_DECAY = 0.995
DQN_TARGET_UPDATE = 10
REPLAY_BUFFER_SIZE = 10000

MITIGATION_ACTIONS = [
    "block_ip", "rate_limit", "reroute_traffic", "honeypot_redirect",
    "alert_only", "quarantine_flow", "drop_packet", "null_route",
]
NUM_ACTIONS = len(MITIGATION_ACTIONS)
STATE_DIM   = NUM_FEATURES + NUM_CLASSES + 4

# ─── Simulation ────────────────────────────────────────────
NUM_SWITCHES      = 10
NUM_HOSTS         = 20
TRAFFIC_RATE      = 1000
ATTACK_PROBABILITY = 0.3

# ─── Evaluation ────────────────────────────────────────────
TEST_SPLIT                  = 0.2
EVAL_ADVERSARIAL_STRENGTH   = 0.1
TARGET_DETECTION_ACCURACY   = 0.97
TARGET_ADVERSARIAL_ACCURACY = 0.89
TARGET_MITIGATION_LATENCY_MS = 340.0

# ─── Privacy ───────────────────────────────────────────────
ENCRYPT_GRADIENTS  = False
GRADIENT_NOISE_STD = 0.001

# ─── Misc ──────────────────────────────────────────────────
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LOG_LEVEL   = "INFO"
RESULTS_DIR = "./results/"