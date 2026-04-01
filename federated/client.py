"""
federated/client.py - FL Client with FedProx (final version)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.detector import IntrusionDetector, evaluate_model
from utils.crypto import GradientEncryptor
from utils.logger import get_logger
import config

logger = get_logger("FL-Client")


def train_fedprox(model, global_weights, X, y, optimizer, mu=0.01,
                  batch_size=128, device="cpu"):
    model.train()
    model.to(device)
    global_tensors = [torch.tensor(w, dtype=torch.float32).to(device) for w in global_weights]
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        ce_loss = criterion(model(xb), yb)
        prox = sum(torch.sum((lp - gp)**2) for lp, gp in zip(model.parameters(), global_tensors))
        loss = ce_loss + (mu / 2.0) * prox
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += ce_loss.item() * len(xb)
    return total_loss / len(X)


class FederatedClient:
    def __init__(self, client_id, X_local, y_local):
        self.client_id      = client_id
        self.X_local        = X_local
        self.y_local        = y_local
        self.n_samples      = len(X_local)
        self.global_weights = None
        self.model          = IntrusionDetector()
        self.optimizer      = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.encryptor      = GradientEncryptor()
        logger.info(f"Client {client_id} | samples={self.n_samples} | params={self.model.count_parameters():,}")

    def receive_global_weights(self, global_weights):
        self.global_weights = copy.deepcopy(global_weights)
        self.model.set_weights(global_weights)

    def local_train(self):
        losses = []
        for _ in range(config.LOCAL_EPOCHS):
            loss = train_fedprox(self.model, self.global_weights, self.X_local, self.y_local,
                                 self.optimizer, mu=0.01, batch_size=config.LOCAL_BATCH_SIZE, device=config.DEVICE)
            losses.append(loss)
        avg_loss = float(np.mean(losses))
        logger.debug(f"Client {self.client_id} | loss={avg_loss:.4f}")
        raw_grads = self.model.get_gradients()
        if config.ENCRYPT_GRADIENTS:
            noisy = self.encryptor.add_differential_privacy_noise(raw_grads, std=config.GRADIENT_NOISE_STD)
            enc = self.encryptor.encrypt_gradients(noisy)
            _ = self.encryptor.decrypt_gradients(enc)
        return {"client_id": self.client_id, "gradients": raw_grads,
                "weights": self.model.get_weights(), "n_samples": self.n_samples, "train_loss": avg_loss}

    def evaluate_local(self):
        return evaluate_model(self.model, self.X_local, self.y_local)
