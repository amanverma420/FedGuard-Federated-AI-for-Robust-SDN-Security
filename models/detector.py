"""
models/detector.py
Deep Neural Network classifier for SDN intrusion detection.

Architecture:
  Input(41) → FC(256) → BN → ReLU → Dropout
            → FC(128) → BN → ReLU → Dropout
            → FC(64)  → BN → ReLU → Dropout
            → FC(5)   → Softmax

Design choices:
  - BatchNorm: stabilizes training with heterogeneous federated data
  - Dropout: reduces overfitting on small client datasets
  - He init: optimal for ReLU activations
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
from utils.logger import get_logger
import config

logger = get_logger("Detector")


class IntrusionDetector(nn.Module):
    """
    Multi-class DNN intrusion detector.
    """

    def __init__(
        self,
        input_dim: int = config.NUM_FEATURES,
        hidden_dims: list = config.HIDDEN_DIMS,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

        # He initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        return self.predict_proba(x).argmax(dim=1)

    def get_gradients(self) -> list:
        """Extract current gradient arrays for FL sharing."""
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().cpu().numpy().copy())
            else:
                grads.append(np.zeros(param.shape, dtype=np.float32))
        return grads

    def get_weights(self) -> list:
        """Extract model weights for FL aggregation."""
        return [p.detach().cpu().numpy().copy() for p in self.parameters()]

    def set_weights(self, weights: list):
        """Set model weights from a list of numpy arrays."""
        with torch.no_grad():
            for param, w in zip(self.parameters(), weights):
                param.copy_(torch.tensor(w, dtype=torch.float32))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_one_epoch(
    model: IntrusionDetector,
    X: np.ndarray,
    y: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int = config.LOCAL_BATCH_SIZE,
    device: str = config.DEVICE,
) -> float:
    """
    Train model for one epoch on local data.
    Returns average cross-entropy loss.
    """
    model.train()
    model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(X)


def evaluate_model(
    model: IntrusionDetector,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
    device: str = config.DEVICE,
) -> dict:
    """
    Evaluate model accuracy and per-class metrics.
    Returns dict with accuracy, per-class precision/recall/f1.
    """
    model.eval()
    model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            preds = model.predict(X_batch).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    accuracy = (all_preds == y).mean()

    # Per-class metrics
    from sklearn.metrics import classification_report, f1_score
    report = classification_report(y, all_preds,
                                   target_names=config.CLASS_NAMES[:config.NUM_CLASSES],
                                   output_dict=True, zero_division=0)
    macro_f1 = f1_score(y, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "report": report,
        "predictions": all_preds,
    }
