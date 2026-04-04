import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.detector import IntrusionDetector
from federated.aggregator import aggregate
from utils.logger import get_logger
import config

logger = get_logger("FL-Server")


def _evaluate(model, X, y, batch_size=512, device=None):
    if device is None:
        device = config.DEVICE
    model.eval()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    acc = float((preds == y).mean())
    from sklearn.metrics import f1_score
    f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "macro_f1": f1, "predictions": preds}


class FederatedServer:
    def __init__(self, X_test, y_test):
        self.global_model = IntrusionDetector()
        self.X_test = X_test
        self.y_test = y_test
        self.round_history = []
        self.current_round = 0
        metrics = _evaluate(self.global_model, X_test, y_test)
        logger.info(f"FL Server initialized | params={self.global_model.count_parameters():,} | init_acc={metrics['accuracy']:.4f}")

    def get_global_weights(self):
        return self.global_model.get_weights()

    def aggregate_updates(self, client_updates):
        new_weights = aggregate(client_updates, strategy=config.BYZANTINE_DEFENSE)
        self.global_model.set_weights(new_weights)

    def run_fl_round(self, clients):
        self.current_round += 1
        logger.info(f"FL Round {self.current_round}/{config.FL_ROUNDS}")

        global_weights = self.get_global_weights()
        for client in clients:
            client.receive_global_weights(copy.deepcopy(global_weights))

        client_updates = []
        for client in clients:
            update = client.local_train()
            client_updates.append(update)

        self.aggregate_updates(client_updates)
        metrics = _evaluate(self.global_model, self.X_test, self.y_test)
        avg_loss = float(np.mean([u["train_loss"] for u in client_updates]))

        stats = {
            "round": self.current_round,
            "train_loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        self.round_history.append(stats)
        logger.info(f"Round {self.current_round} | Acc={metrics['accuracy']:.4f} | F1={metrics['macro_f1']:.4f} | Loss={avg_loss:.4f}")
        return stats

    def run_all_rounds(self, clients):
        logger.info(f"Starting FL: {config.FL_ROUNDS} rounds x {len(clients)} clients | strategy={config.BYZANTINE_DEFENSE}")
        for _ in range(config.FL_ROUNDS):
            self.run_fl_round(clients)
        final = self.round_history[-1]
        logger.info(f"FL Complete | Final Acc={final['accuracy']:.4f} | F1={final['macro_f1']:.4f}")
        return self.round_history