import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    """1D Convolutional Denoising Autoencoder for EEG artifact removal.

    Architecture:
        Encoder: Conv1d layers -> latent representation
        Decoder: ConvTranspose1d layers -> reconstructed clean signal

    Input:  (batch, n_channels, n_samples)  - noisy EEG
    Output: (batch, n_channels, n_samples)  - denoised EEG
    """

    def __init__(self, n_channels=64, n_samples=250, latent_dim=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(16, latent_dim, kernel_size=1),
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, n_channels, kernel_size=1),
        )

    def forward(self, x):
        """Forward pass: noisy -> denoised."""
        z = self.encoder(x)
        z = self.bottleneck(z)
        reconstructed = self.decoder(z)
        # Ensure output matches input length exactly
        reconstructed = reconstructed[:, :, :x.shape[-1]]
        return reconstructed

    def encode(self, x):
        """Get latent representation."""
        z = self.encoder(x)
        return self.bottleneck(z)


class EEGNetDenoiser(nn.Module):
    """EEGNet-based denoiser using depthwise & separable convolutions.

    More parameter-efficient alternative to the full DAE.
    """

    def __init__(self, n_channels=64, n_samples=250, F1=8, D=2, F2=16):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3),
        )

        # Decoder to reconstruct back to full resolution
        self.decode = nn.Sequential(
            nn.ConvTranspose1d(F2, 32, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, n_channels, kernel_size=1),
        )

    def forward(self, x):
        """x: (batch, n_channels, n_samples)"""
        batch = x.shape[0]
        x2d = x.unsqueeze(1)  # (B, 1, C, T)

        h = self.block1(x2d)
        h = self.depthwise(h)
        h = self.separable(h)

        # Flatten spatial dims and decode
        h = h.squeeze(2)  # (B, F2, T')
        out = self.decode(h)
        return out[:, :, :x.shape[-1]]
