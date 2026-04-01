"""
test_fl_full.py - Full FL test with FedProx + SGD
Run: python test_fl_full.py
"""
import config
config.NUM_CLIENTS       = 5
config.FL_ROUNDS         = 20
config.LOCAL_EPOCHS      = 5
config.LOCAL_BATCH_SIZE  = 128
config.BYZANTINE_DEFENSE = "fedavg"
config.NUM_BYZANTINE     = 0
config.ENCRYPT_GRADIENTS = False   # skip for speed

from data.synthetic_generator import generate_dataset, split_for_clients
from data.preprocessor import SDNPreprocessor
from federated.client import FederatedClient
from federated.server import FederatedServer
from sklearn.model_selection import train_test_split
import numpy as np

print("=" * 60)
print("  FedGuard - FL Test (FedProx + SGD + FedAvg)")
print("  5 Controllers | 20 Rounds | non-IID data")
print("=" * 60)

print("\n[1/4] Generating 50,000 samples...")
X, y = generate_dataset(n_total=50000)

print("[2/4] Preprocessing...")
pre = SDNPreprocessor()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = pre.fit_transform(X_train)
X_test  = pre.transform(X_test)
print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

# Show test class distribution
unique, counts = np.unique(y_test, return_counts=True)
print(f"      Test class dist: { {int(u): int(c) for u,c in zip(unique,counts)} }")

print("[3/4] Distributing across 5 controllers (non-IID)...")
client_data = split_for_clients(X_train, y_train, n_clients=config.NUM_CLIENTS)
for i, (Xc, yc) in enumerate(client_data):
    u2, c2 = np.unique(yc, return_counts=True)
    print(f"      Controller {i}: {len(Xc)} samples | { {int(u):int(c) for u,c in zip(u2,c2)} }")

print("\n[4/4] Running Federated Learning (FedProx)...")
server  = FederatedServer(X_test, y_test)
clients = [FederatedClient(i, cd[0], cd[1]) for i, cd in enumerate(client_data)]
history = server.run_all_rounds(clients)

print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  Round | Accuracy | Macro F1 | Train Loss")
print(f"  ------|----------|----------|------------")
for h in history:
    marker = " <-- TARGET MET" if h['accuracy'] >= 0.97 else ""
    print(f"  {h['round']:5d} | {h['accuracy']:.4f}   | {h['macro_f1']:.4f}   | {h['train_loss']:.4f}{marker}")

final = history[-1]
print(f"\n  Final Accuracy : {final['accuracy']:.4f}  (target >= 0.97)")
print(f"  Final Macro F1 : {final['macro_f1']:.4f}")
print(f"  Status: {'TARGET MET' if final['accuracy'] >= 0.97 else 'Below target'}")
print("=" * 60)
