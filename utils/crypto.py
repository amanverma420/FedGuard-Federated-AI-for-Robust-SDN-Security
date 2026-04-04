import io
import numpy as np
from cryptography.fernet import Fernet
from utils.logger import get_logger

logger = get_logger("Crypto")


class GradientEncryptor:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        logger.info("GradientEncryptor initialized with fresh symmetric key.")

    def encrypt_gradients(self, gradients: list) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, np.array(gradients, dtype=object), allow_pickle=True)
        raw = buffer.getvalue()
        return self.cipher.encrypt(raw)

    def decrypt_gradients(self, encrypted: bytes) -> list:
        raw = self.cipher.decrypt(encrypted)
        buffer = io.BytesIO(raw)
        return list(np.load(buffer, allow_pickle=True))

    def add_differential_privacy_noise(self, gradients: list, std: float = 0.001) -> list:
        return [g + np.random.normal(0, std, size=g.shape).astype(g.dtype) for g in gradients]