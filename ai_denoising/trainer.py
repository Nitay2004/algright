import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .model import DenoisingAutoencoder


class DenoisingDataset(Dataset):
    """Dataset for training the DAE: (noisy_input, clean_target) pairs."""

    def __init__(self, noisy_signals, clean_signals):
        """
        Args:
            noisy_signals: ndarray (n_trials, n_channels, n_samples)  — corrupted input
            clean_signals: ndarray (n_trials, n_channels, n_samples)  — reconstruction target
        """
        self.noisy = torch.FloatTensor(noisy_signals)
        self.clean = torch.FloatTensor(clean_signals)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def normalize_trials(data):
    """Z-score normalize each trial independently.

    Args:
        data: ndarray (n_trials, n_channels, n_samples)

    Returns:
        normalized: same shape, zero-mean unit-std per trial
        means:  (n_trials, n_channels, 1)
        stds:   (n_trials, n_channels, 1)
    """
    means = data.mean(axis=-1, keepdims=True)
    stds  = data.std(axis=-1, keepdims=True)
    stds  = np.where(stds < 1e-10, 1.0, stds)   # avoid div-by-zero
    return (data - means) / stds, means, stds


def denormalize_trials(data_norm, means, stds):
    """Reverse the per-trial z-score normalization."""
    return data_norm * stds + means


def create_training_data(clean_data, noise_level_db=-5.0, sfreq=250):
    """Generate noisy training pairs from clean EEG data.

    Args:
        clean_data: ndarray (n_trials, n_channels, n_samples)
        noise_level_db: desired SNR of the noisy signal
        sfreq: sampling frequency

    Returns:
        noisy_data, clean_data (same shape as input)
    """
    n_trials, n_channels, n_samples = clean_data.shape
    noisy_data = np.zeros_like(clean_data)

    for i in range(n_trials):
        clean = clean_data[i]
        # Generate colored noise (1/f + white)
        noise = np.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            # White noise
            white = np.random.randn(n_samples)
            # 1/f (pink) noise
            freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
            freqs[0] = 1.0
            pink_spectrum = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
            pink_spectrum /= np.sqrt(freqs)
            pink = np.fft.irfft(pink_spectrum, n=n_samples)
            pink = pink[:n_samples]

            combined_noise = 0.5 * white + 0.5 * pink

            # Scale to desired SNR
            signal_power = np.mean(clean[ch] ** 2)
            noise_power  = np.mean(combined_noise ** 2)
            snr_linear   = 10 ** (noise_level_db / 10.0)
            scale = np.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
            noise[ch] = combined_noise * scale

        noisy_data[i] = clean + noise

    return noisy_data, clean_data


def train_dae(clean_data, config=None, model=None, noise_level_db=-5.0, noisy_data=None):
    """Train the Denoising Autoencoder.

    The model is trained on z-score normalized (noisy → clean) pairs so that
    input and target are always on the same unit-std scale.  The returned
    model therefore also expects normalized input at inference time — use
    OnlineDenoiser.denoise_epoch / denoise_batch which handle the
    normalization/denormalization automatically.

    Args:
        clean_data: ndarray (n_trials, n_channels, n_samples) - clean EEG
        config: PipelineConfig instance
        model: pre-existing model to continue training (optional)
        noise_level_db: noise level for synthetic noise injection
        noisy_data: ndarray - if provided, use this instead of synthesizing noise

    Returns:
        trained model, training losses
    """
    if config is None:
        from config import PipelineConfig
        config = PipelineConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DAE-Train] Using device: {device}")

    # --- Step 1: synthesize noisy signals from clean if not provided ---
    if noisy_data is None:
        noisy_data, target_data = create_training_data(clean_data, noise_level_db)
    else:
        target_data = clean_data

    # --- Step 2: normalize BOTH noisy and clean to unit std (per trial) ---
    # This ensures the model always operates in a consistent amplitude range
    # regardless of the raw EEG scale (µV, normalized, etc.)
    noisy_norm, noisy_means, noisy_stds = normalize_trials(noisy_data)
    # Use the same stats (from the noisy signal) for the clean target so the
    # model learns to output the "clean version in the same normalized space"
    clean_norm = (target_data - noisy_means) / noisy_stds

    # --- Step 3: build dataset (noisy first, clean second — correct order!) ---
    dataset = DenoisingDataset(noisy_norm, clean_norm)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    n_channels = clean_data.shape[1]
    if model is None:
        model = DenoisingAutoencoder(
            n_channels=n_channels,
            n_samples=clean_data.shape[2],
            latent_dim=config.latent_dim
        ).to(device)
    else:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
    )
    criterion = torch.nn.MSELoss()

    model.train()
    losses = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_noisy, batch_clean in dataloader:
            batch_noisy = batch_noisy.to(device)
            batch_clean = batch_clean.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch_noisy)
            loss = criterion(reconstructed, batch_clean)
            loss.backward()
            # Gradient clipping — prevents exploding gradients on small datasets
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_noisy.shape[0]

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [DAE-Train] Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.6f}")

    model.eval()
    print(f"[DAE-Train] Training complete. Final loss: {losses[-1]:.6f}")
    return model, losses
