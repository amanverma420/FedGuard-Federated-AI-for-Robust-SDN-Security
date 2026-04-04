import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.logger import get_logger
import config

logger = get_logger("GAN")


class Generator(nn.Module):
    def __init__(self, latent_dim=config.GAN_LATENT_DIM, output_dim=config.NUM_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim), nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=config.NUM_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class AttackGAN:
    def __init__(self, device=None):
        self.device = device or config.DEVICE
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=config.GAN_LR_G, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=config.GAN_LR_D, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.g_losses = []
        self.d_losses = []
        self.trained = False
        logger.info(f"AttackGAN initialized | G params={sum(p.numel() for p in self.G.parameters()):,}")

    def train(self, X_attack: np.ndarray, epochs=None, batch_size=None):
        if epochs is None:
            epochs = config.GAN_EPOCHS
        if batch_size is None:
            batch_size = config.GAN_BATCH_SIZE
        logger.info(f"GAN training | samples={len(X_attack)} | epochs={epochs}")
        X_t = torch.tensor(X_attack, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            g_ep, d_ep, n_batches = 0.0, 0.0, 0
            for (real_batch,) in loader:
                bs = real_batch.size(0)
                real_labels = torch.ones(bs, 1).to(self.device)
                fake_labels = torch.zeros(bs, 1).to(self.device)

                self.opt_D.zero_grad()
                d_real = self.D(real_batch)
                loss_real = self.criterion(d_real, real_labels)
                z = torch.randn(bs, config.GAN_LATENT_DIM).to(self.device)
                fake_batch = self.G(z).detach()
                d_fake = self.D(fake_batch)
                loss_fake = self.criterion(d_fake, fake_labels)
                d_loss = (loss_real + loss_fake) / 2
                d_loss.backward()
                self.opt_D.step()

                self.opt_G.zero_grad()
                z = torch.randn(bs, config.GAN_LATENT_DIM).to(self.device)
                g_loss = self.criterion(self.D(self.G(z)), real_labels)
                g_loss.backward()
                self.opt_G.step()

                g_ep += g_loss.item()
                d_ep += d_loss.item()
                n_batches += 1

            avg_g = g_ep / max(n_batches, 1)
            avg_d = d_ep / max(n_batches, 1)
            self.g_losses.append(avg_g)
            self.d_losses.append(avg_d)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  GAN Epoch {epoch+1}/{epochs} | G_loss={avg_g:.4f} | D_loss={avg_d:.4f}")

        self.trained = True
        logger.info("GAN training complete.")

    def generate(self, n_samples: int) -> np.ndarray:
        assert self.trained, "Call train() first."
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, config.GAN_LATENT_DIM).to(self.device)
            samples = self.G(z).cpu().numpy()
        return samples.astype(np.float32)


def fgsm_attack(model, X: np.ndarray, y: np.ndarray, epsilon: float = 0.1, device=None) -> np.ndarray:
    if device is None:
        device = config.DEVICE
    model.eval()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(model(X_t), y_t)
    loss.backward()
    X_adv = (X_t + epsilon * X_t.grad.sign()).detach().cpu().numpy()
    return X_adv.astype(np.float32)