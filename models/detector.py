import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import get_logger
import config

logger = get_logger("Detector")

class IntrusionDetector(nn.Module):
    def __init__(
        self,
        input_dim: int = config.NUM_FEATURES,
        hidden_dims: list = None,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = config.HIDDEN_DIMS
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(dim=1)

    def get_weights(self) -> list:
        return [p.detach().cpu().numpy().copy() for p in self.parameters()]

    def set_weights(self, weights: list):
        with torch.no_grad():
            for param, w in zip(self.parameters(), weights):
                param.copy_(torch.tensor(w, dtype=torch.float32))

    def get_gradients(self) -> list:
        return [
            p.grad.detach().cpu().numpy().copy() if p.grad is not None else np.zeros(p.shape, dtype=np.float32)
            for p in self.parameters()
        ]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_one_epoch(model, X, y, optimizer, batch_size=128, device=None):
    if device is None:
        device = config.DEVICE
    model.train()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(X)


def evaluate_model(model, X, y, batch_size=512, device=None):
    if device is None:
        device = config.DEVICE
    model.eval()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for (xb,) in loader:
            all_preds.extend(model.predict(xb).cpu().numpy())
    all_preds = np.array(all_preds)
    accuracy = float((all_preds == y).mean())
    from sklearn.metrics import f1_score, classification_report
    macro_f1 = float(f1_score(y, all_preds, average="macro", zero_division=0))
    report = classification_report(y, all_preds, target_names=config.CLASS_NAMES[:config.NUM_CLASSES], output_dict=True, zero_division=0)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "report": report, "predictions": all_preds}