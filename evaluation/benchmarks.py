"""
evaluation/benchmarks.py - Full benchmark runner for FedGuard.
"""
import numpy as np
from utils.logger import get_logger
import config

logger = get_logger("Benchmarks")


def run_detection_benchmark(model, X_test, y_test):
    from evaluation.metrics import detection_metrics
    import torch
    logger.info("Running detection benchmark...")
    with torch.no_grad():
        x_t   = torch.tensor(X_test, dtype=torch.float32)
        preds = model.predict(x_t).numpy()
    metrics = detection_metrics(y_test, preds)
    logger.info(f"Detection Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Detection Rate:      {metrics['detection_rate']:.4f}")
    logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    return metrics


def run_adversarial_benchmark(model, X_test, y_test, epsilon=0.1):
    from adversarial.gan import fgsm_attack
    from sklearn.metrics import accuracy_score
    import torch
    logger.info(f"Running adversarial benchmark (FGSM eps={epsilon})...")
    X_adv = fgsm_attack(model, X_test[:2000], y_test[:2000], epsilon=epsilon)
    with torch.no_grad():
        x_t   = torch.tensor(X_adv, dtype=torch.float32)
        preds = model.predict(x_t).numpy()
    acc = float(accuracy_score(y_test[:2000], preds))
    logger.info(f"Adversarial Accuracy (eps={epsilon}): {acc:.4f}")
    return acc


def run_mitigation_benchmark(agent, env, n_episodes=100):
    logger.info("Running mitigation benchmark...")
    metrics = agent.evaluate_mitigation(env, n_episodes=n_episodes)
    logger.info(f"Mitigation Accuracy: {metrics['mitigation_accuracy']:.4f}")
    logger.info(f"Avg Latency:         {metrics['avg_latency_ms']:.1f} ms")
    return metrics
