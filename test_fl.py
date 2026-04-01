"""
test_fl.py - Quick test for Federated Learning
Run: python test_fl.py
"""
from data.synthetic_generator import generate_dataset, split_for_clients
from data.preprocessor import SDNPreprocessor
from federated.client import FederatedClient
from federated.server import FederatedServer
from sklearn.model_selection import train_test_split

X, y = generate_dataset(n_total=10000)
pre = SDNPreprocessor()
X = pre.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
client_data = split_for_clients(X_train, y_train, n_clients=3)

server = FederatedServer(X_test, y_test)
clients = [FederatedClient(i, cd[0], cd[1]) for i, cd in enumerate(client_data)]

print("Running 2 FL rounds with 3 clients...")
for r in range(2):
    stats = server.run_fl_round(clients)
    print(f"Round {r+1} -> Acc: {round(stats['accuracy'], 4)} | F1: {round(stats['macro_f1'], 4)}")

print("\nFederated Learning test PASSED!")
