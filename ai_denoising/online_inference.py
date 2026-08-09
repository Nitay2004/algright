import math
import numpy as np
import torch
from .model import DenoisingAutoencoder
from .trainer import normalize_trials, denormalize_trials


class OnlineDenoiser:
    """Online inference wrapper for the trained DAE.

    Processes single epochs in real-time for BCI pipeline integration.
    """

    def __init__(self, model=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        else:
            self.model = None

    def load_model(self, path):
        """Load a trained model from disk."""
        self.model = DenoisingAutoencoder()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def denoise_epoch(self, epoch_data):
        """Denoise a single epoch.

        Normalizes input to unit-std before inference (matching training
        distribution) and restores original scale in the output.

        Args:
            epoch_data: ndarray (n_channels, n_samples)

        Returns:
            ndarray (n_channels, n_samples) - denoised, same scale as input
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")

        # Per-epoch z-score normalization
        mean = epoch_data.mean(axis=-1, keepdims=True)
        std  = epoch_data.std(axis=-1, keepdims=True)
        std  = np.where(std < 1e-10, 1.0, std)
        epoch_norm = (epoch_data - mean) / std

        x = torch.FloatTensor(epoch_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.model(x)
        out_norm = y.squeeze(0).cpu().numpy()

        # Restore original scale
        return out_norm * std + mean

    def denoise_batch(self, batch_data):
        """Denoise a batch of epochs.

        Normalizes each trial independently before inference and restores
        original scale in the output.

        Args:
            batch_data: ndarray (n_epochs, n_channels, n_samples)

        Returns:
            ndarray (n_epochs, n_channels, n_samples) - denoised, original scale
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        batch_norm, means, stds = normalize_trials(batch_data)

        x = torch.FloatTensor(batch_norm).to(self.device)
        with torch.no_grad():
            y = self.model(x)
        out_norm = y.cpu().numpy()

        # Restore per-trial scale
        return denormalize_trials(out_norm, means, stds)

    def online_adapt_step(self, noisy_epoch, clean_target, lr=1e-4):
        """Single-step online adaptation (self-learning).

        Performs one gradient update to adapt the model to the current
        signal distribution. Input and target are normalized to match
        the training distribution before the gradient step.

        Args:
            noisy_epoch: ndarray (n_channels, n_samples)
            clean_target: ndarray (n_channels, n_samples) - ground truth

        Returns:
            (loss value, parameter shift): loss is the MSE of the update
            step; parameter shift is the root-mean-square change in the
            model weights induced by this update.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        # Normalize input — must match training-time normalization
        mean = noisy_epoch.mean(axis=-1, keepdims=True)
        std  = noisy_epoch.std(axis=-1, keepdims=True)
        std  = np.where(std < 1e-10, 1.0, std)
        noisy_norm  = (noisy_epoch  - mean) / std
        target_norm = (clean_target - mean) / std   # same stats as noisy

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        x      = torch.FloatTensor(noisy_norm).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(target_norm).unsqueeze(0).to(self.device)

        before = [p.detach().clone() for p in self.model.parameters()]

        optimizer.zero_grad()
        y = self.model(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        # Root-mean-square weight change across all parameters
        total_numel = 0
        total_sq = 0.0
        for p, b in zip(self.model.parameters(), before):
            delta = (p.detach() - b)
            total_sq += float(delta.pow(2).sum().item())
            total_numel += b.numel()
        param_shift = math.sqrt(total_sq / total_numel) if total_numel > 0 else 0.0

        self.model.eval()
        return loss.item(), param_shift
