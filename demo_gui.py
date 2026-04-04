"""
demo_gui.py - Demonstration of GUI functionality
Shows how the GUI interacts with the FedGuard project
"""
import config
import numpy as np
from data.data_loader import load_dataset
from data.preprocessor import SDNPreprocessor
from models.detector import IntrusionDetector, train_one_epoch, evaluate_model
import torch

def demo_data_loading():
    """Demo data loading and preprocessing"""
    print("📊 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_dataset()
    pre = SDNPreprocessor()
    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)

    print(f"✅ Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

def demo_model_training(X_train, y_train, X_test, y_test):
    """Demo quick model training"""
    print("🤖 Training intrusion detector...")
    model = IntrusionDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Quick training for demo
    for epoch in range(3):
        loss = train_one_epoch(model, X_train, y_train, optimizer, batch_size=128)
        metrics = evaluate_model(model, X_test, y_test)
        print(".3f")

    print("✅ Model trained!")
    return model

def demo_evaluation(model, X_test, y_test):
    """Demo model evaluation"""
    print("📈 Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(".4f")
    print(".4f")
    print(".4f")
    return metrics

def main():
    print("🎯 FedGuard GUI Demo")
    print("=" * 50)

    # Load data
    X_train, X_test, y_train, y_test = demo_data_loading()

    # Train model
    model = demo_model_training(X_train, y_train, X_test, y_test)

    # Evaluate
    metrics = demo_evaluation(model, X_test, y_test)

    print("\n🎉 Demo complete! The GUI provides interactive access to all these features.")
    print("Run 'python launch_gui.py' to start the interactive GUI!")

if __name__ == "__main__":
    main()