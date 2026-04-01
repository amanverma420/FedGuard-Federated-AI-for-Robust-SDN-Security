"""
federated/server.py - Global FL Server
Fixed: evaluates global model correctly using torch no_grad inference.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.detector import IntrusionDetector
from federated.aggregator import aggregate
from utils.logger import get_logger
import config

logger = get_logger("FL-Server")


def _evaluate(model, X, y, batch_size=512, device="cpu"):
    """Direct evaluation — bypasses any caching issues."""
    model.eval()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb)
            preds.append(out.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    acc = float((preds == y).mean())

    from sklearn.metrics import f1_score, classification_report
    f1  = float(f1_score(y, preds, average="macro", zero_division=0))
    rep = classification_report(
        y, preds,
        target_names=config.CLASS_NAMES[:config.NUM_CLASSES],
        output_dict=True, zero_division=0
    )
    # Show per-class prediction counts for debugging
    unique, counts = np.unique(preds, return_counts=True)
    pred_dist = {int(u): int(c) for u, c in zip(unique, counts)}
    logger.debug(f"Global model pred distribution: {pred_dist}")

    return {"accuracy": acc, "macro_f1": f1, "report": rep, "predictions": preds}


class FederatedServer:
    def __init__(self, X_test, y_test):
        self.global_model  = IntrusionDetector()
        self.X_test        = X_test
        self.y_test        = y_test
        self.round_history = []
        self.current_round = 0

        # Verify initial prediction distribution
        metrics = _evaluate(self.global_model, X_test, y_test)
        logger.info(
            f"FL Server initialized | params={self.global_model.count_parameters():,} | "
            f"test={len(X_test)} | init_acc={metrics['accuracy']:.4f}"
        )

    def get_global_weights(self):
        return self.global_model.get_weights()

    def aggregate_updates(self, client_updates):
        new_weights = aggregate(client_updates, strategy=config.BYZANTINE_DEFENSE)
        self.global_model.set_weights(new_weights)

    def run_fl_round(self, clients):
        self.current_round += 1
        logger.info(f"{'='*50}")
        logger.info(f"FL Round {self.current_round}/{config.FL_ROUNDS}")

        # Broadcast
        global_weights = self.get_global_weights()
        for client in clients:
            client.receive_global_weights(copy.deepcopy(global_weights))

        # Local training
        client_updates = []
        for client in clients:
            update = client.local_train()
            client_updates.append(update)
            logger.debug(
                f"  Client {client.client_id} | loss={update['train_loss']:.4f} | "
                f"n={update['n_samples']}"
            )

        # Aggregate
        self.aggregate_updates(client_updates)

        # Evaluate fresh
        metrics = _evaluate(self.global_model, self.X_test, self.y_test)
        avg_loss = float(np.mean([u["train_loss"] for u in client_updates]))

        stats = {
            "round":      self.current_round,
            "train_loss": avg_loss,
            "accuracy":   metrics["accuracy"],
            "macro_f1":   metrics["macro_f1"],
        }
        self.round_history.append(stats)

        logger.info(
            f"Round {self.current_round} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['macro_f1']:.4f} | "
            f"Loss={avg_loss:.4f}"
        )
        return stats

    def run_all_rounds(self, clients):
        logger.info(
            f"Starting FL: {config.FL_ROUNDS} rounds x {len(clients)} clients | "
            f"strategy={config.BYZANTINE_DEFENSE}"
        )
        for _ in range(config.FL_ROUNDS):
            self.run_fl_round(clients)

        final = self.round_history[-1]
        status = "PASSED" if final["accuracy"] >= config.TARGET_DETECTION_ACCURACY else "BELOW TARGET"
        logger.info(
            f"\nFL Complete | Final Acc={final['accuracy']:.4f} | "
            f"F1={final['macro_f1']:.4f} | {status}"
        )
        return self.round_history
