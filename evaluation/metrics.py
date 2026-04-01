"""
evaluation/metrics.py - Comprehensive evaluation metrics for FedGuard.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from utils.logger import get_logger
import config

logger = get_logger("Metrics")


def detection_metrics(y_true, y_pred, y_proba=None):
    """Full detection performance metrics."""
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    rep  = classification_report(y_true, y_pred,
           target_names=config.CLASS_NAMES[:config.NUM_CLASSES], zero_division=0)

    # False positive rate (normal classified as attack)
    normal_mask = (y_true == 0)
    fpr = float(np.mean(y_pred[normal_mask] != 0)) if normal_mask.any() else 0.0

    # Detection rate (attacks caught)
    attack_mask = (y_true != 0)
    dr = float(np.mean(y_pred[attack_mask] != 0)) if attack_mask.any() else 0.0

    metrics = {
        "accuracy":        float(acc),
        "macro_f1":        float(f1),
        "macro_precision": float(prec),
        "macro_recall":    float(rec),
        "false_positive_rate": fpr,
        "detection_rate":  dr,
        "confusion_matrix": cm.tolist(),
        "report":          rep,
    }
    return metrics


def adversarial_robustness_metrics(model, X_test, y_test, epsilons=None):
    """
    Evaluate model robustness under FGSM adversarial perturbations
    at multiple epsilon strengths.
    """
    from adversarial.gan import fgsm_attack
    if epsilons is None:
        epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]

    results = {}
    for eps in epsilons:
        if eps == 0.0:
            X_eval = X_test
        else:
            X_eval = fgsm_attack(model, X_test, y_test, epsilon=eps)

        import torch
        with torch.no_grad():
            x_t   = torch.tensor(X_eval, dtype=torch.float32)
            preds = model.predict(x_t).numpy()

        acc = float(accuracy_score(y_test, preds))
        f1  = float(f1_score(y_test, preds, average="macro", zero_division=0))
        results[eps] = {"accuracy": acc, "macro_f1": f1}
        logger.info(f"  FGSM eps={eps:.2f} | Acc={acc:.4f} | F1={f1:.4f}")

    return results


def mitigation_latency_stats(latencies: list) -> dict:
    """Compute latency statistics for DQN mitigation actions."""
    arr = np.array(latencies, dtype=np.float32)
    return {
        "mean_ms":   float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "min_ms":    float(np.min(arr)),
        "max_ms":    float(np.max(arr)),
    }


def print_summary(detection_acc, adversarial_acc, mitigation_latency_ms):
    """Print final benchmark summary against project targets."""
    print("\n" + "=" * 65)
    print("  FEDGUARD — FINAL BENCHMARK RESULTS")
    print("=" * 65)
    targets = [
        ("Detection Accuracy",     detection_acc,        config.TARGET_DETECTION_ACCURACY,    "≥"),
        ("Adversarial Accuracy",   adversarial_acc,      config.TARGET_ADVERSARIAL_ACCURACY,  "≥"),
        ("Mitigation Latency (ms)",mitigation_latency_ms,config.TARGET_MITIGATION_LATENCY_MS, "≤"),
    ]
    all_pass = True
    for name, value, target, op in targets:
        if op == "≥":
            passed = value >= target
        else:
            passed = value <= target
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:<28} {value:>8.4f}  (target {op} {target})  {status}")
    print("=" * 65)
    print(f"  Overall: {'✅ ALL TARGETS MET' if all_pass else '⚠  Some targets not met'}")
    print("=" * 65)
