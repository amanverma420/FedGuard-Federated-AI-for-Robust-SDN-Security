"""
adversarial/augmentor.py - Adversarial training augmentation pipeline.

Combines GAN-generated samples + FGSM perturbations to create a
hardened training dataset that is robust to adversarial evasion.
"""
import numpy as np
from utils.logger import get_logger
import config

logger = get_logger("Augmentor")


class AdversarialAugmentor:
    """
    Two-stage adversarial augmentation:
      Stage 1: GAN generates evasion-optimized synthetic attack samples
      Stage 2: FGSM perturbs existing samples toward decision boundary

    Combined augmented data is mixed with clean data for robust training.
    """

    def __init__(self, model=None):
        self.model = model
        self.gan   = None

    def fit_gan(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train GAN on attack-class samples."""
        from adversarial.gan import AttackGAN
        attack_mask = y_train != 0
        X_attack = X_train[attack_mask]
        if len(X_attack) < config.GAN_BATCH_SIZE:
            logger.warning(f"Too few attack samples ({len(X_attack)}) for GAN. Skipping.")
            return
        logger.info(f"Fitting GAN on {len(X_attack)} attack samples...")
        self.gan = AttackGAN()
        self.gan.train(X_attack, epochs=config.GAN_EPOCHS, batch_size=config.GAN_BATCH_SIZE)

    def augment(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        """
        Augment training data with adversarial samples.
        Returns (X_augmented, y_augmented).
        """
        X_parts = [X_train]
        y_parts = [y_train]
        n_aug   = int(len(X_train) * config.ADVERSARIAL_RATIO)

        # Stage 1: GAN-generated adversarial samples
        if self.gan is not None and self.gan.trained:
            X_gan = self.gan.generate(n_aug)
            # Label GAN samples as DoS (class 1) — they mimic attack traffic
            y_gan = np.ones(len(X_gan), dtype=np.int64)
            X_parts.append(X_gan)
            y_parts.append(y_gan)
            logger.info(f"Added {len(X_gan)} GAN adversarial samples.")

        # Stage 2: FGSM perturbations on existing samples
        if self.model is not None:
            from adversarial.gan import fgsm_attack
            idx = np.random.choice(len(X_train), size=min(n_aug, len(X_train)), replace=False)
            try:
                X_fgsm = fgsm_attack(self.model, X_train[idx], y_train[idx],
                                     epsilon=config.EVAL_ADVERSARIAL_STRENGTH)
                X_parts.append(X_fgsm)
                y_parts.append(y_train[idx])
                logger.info(f"Added {len(X_fgsm)} FGSM-perturbed samples.")
            except Exception as e:
                logger.warning(f"FGSM augmentation skipped: {e}")

        X_aug = np.vstack(X_parts).astype(np.float32)
        y_aug = np.concatenate(y_parts).astype(np.int64)

        # Shuffle
        idx = np.random.permutation(len(X_aug))
        logger.info(f"Augmented dataset: {len(X_train)} → {len(X_aug)} samples (+{len(X_aug)-len(X_train)})")
        return X_aug[idx], y_aug[idx]
