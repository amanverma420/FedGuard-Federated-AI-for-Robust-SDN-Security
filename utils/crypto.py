"""
utils/crypto.py - Gradient encryption for privacy-preserving FL
Uses Fernet symmetric encryption to protect gradient updates in transit.
"""
import io
import numpy as np
from cryptography.fernet import Fernet
from utils.logger import get_logger

logger = get_logger("Crypto")

class GradientEncryptor:
    """
    Encrypts and decrypts model gradient tensors using symmetric encryption.
    In a real deployment, each client would have a unique key shared
    with the aggregation server over a secure channel.
    """

    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        logger.info("GradientEncryptor initialized with fresh symmetric key.")

    def encrypt_gradients(self, gradients: list) -> bytes:
        """
        Serialize and encrypt a list of numpy gradient arrays.
        Returns encrypted bytes.
        """
        buffer = io.BytesIO()
        np.save(buffer, np.array(gradients, dtype=object), allow_pickle=True)
        raw = buffer.getvalue()
        encrypted = self.cipher.encrypt(raw)
        return encrypted

    def decrypt_gradients(self, encrypted: bytes) -> list:
        """
        Decrypt and deserialize gradient arrays.
        Returns list of numpy arrays.
        """
        raw = self.cipher.decrypt(encrypted)
        buffer = io.BytesIO(raw)
        gradients = np.load(buffer, allow_pickle=True)
        return list(gradients)

    def add_differential_privacy_noise(self, gradients: list, std: float = 0.001) -> list:
        """
        Add Gaussian noise to gradients for differential privacy.
        This prevents the server from reconstructing private training data.
        """
        noisy = []
        for g in gradients:
            noise = np.random.normal(0, std, size=g.shape).astype(g.dtype)
            noisy.append(g + noise)
        return noisy
