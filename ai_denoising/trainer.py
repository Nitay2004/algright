import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import get_device
from .model import DenoisingAutoencoder


class ClassificationHead(nn.Module):
    """Global-average-pooled classifier on top of the DAE bottleneck latent."""

    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(latent_dim, n_classes)

    def forward(self, z):
        z = self.pool(z).squeeze(-1)
        return self.fc(z)


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


class MaskedDenoisingDataset(Dataset):
    """Self-supervised masked-reconstruction dataset (noise2self style).

    No clean ground truth is required: random contiguous segments of the
    (normalized) real signal are zeroed out and the model must reconstruct
    them from surrounding context. A model that can rebuild occluded signal
    learns the underlying EEG structure, so running it on the full signal
    acts as artifact suppression without needing paired clean data.
    """

    def __init__(self, signals, mask_ratio=0.3, mask_len=16):
        """
        Args:
            signals: ndarray (n_trials, n_channels, n_samples) - z-score normalized
            mask_ratio: fraction of each trial's samples to occlude
            mask_len: length of each occluded segment (samples)
        """
        self.signal = torch.FloatTensor(signals)
        self.mask_ratio = mask_ratio
        self.mask_len = mask_len

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, idx):
        x = self.signal[idx].clone()
        n_ch, n_s = x.shape
        n_masks = max(1, int(np.ceil(n_s * self.mask_ratio / self.mask_len)))
        for _ in range(n_masks):
            # Occlude the SAME time window on ALL channels so the model must
            # exploit cross-channel spatial structure (like real EEG artifacts),
            # not single-channel temporal inpainting alone.
            start = np.random.randint(0, n_s - self.mask_len + 1)
            x[:, start:start + self.mask_len] = 0.0
        return x, self.signal[idx]


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


def train_dae(clean_data, config=None, model=None, noise_level_db=-5.0, noisy_data=None,
              val_split=0.15, masked=False, mask_ratio=0.3):
    """Train the Denoising Autoencoder.

    Two training schemes are supported:

    1. Supervised (synthetic or paired data, ``masked=False``):
       The model is trained on z-score normalized (noisy → clean) pairs so
       that input and target are always on the same unit-std scale. The
       returned model therefore also expects normalized input at inference
       time — use OnlineDenoiser.denoise_epoch / denoise_batch which handle
       the normalization/denormalization automatically.

    2. Self-supervised masked reconstruction (real data, ``masked=True``):
       When no clean ground truth exists (real EEG), random segments of the
       normalized real signal are occluded and the model must rebuild them
       from context. This teaches the real signal structure and provides a
       principled denoising prior without paired clean data.

    Args:
        clean_data: ndarray (n_trials, n_channels, n_samples) - clean EEG
        config: PipelineConfig instance
        model: pre-existing model to continue training (optional)
        noise_level_db: noise level for synthetic noise injection
        noisy_data: ndarray - if provided, use this instead of synthesizing noise
        val_split: fraction of training trials held out for validation
        masked: use masked-reconstruction self-supervision (real data)
        mask_ratio: fraction of samples occluded per trial (masked mode only)

    Returns:
        trained model, training losses, validation losses
    """
    if config is None:
        from config import PipelineConfig
        config = PipelineConfig()

    device = get_device()
    print(f"[DAE-Train] Using device: {device}")

    # --- Step 1: build train/validation indices (shared by both schemes) ---
    n_total = len(clean_data)
    n_val = int(n_total * val_split)
    n_val = min(n_val, n_total - 1)
    if n_val > 0:
        indices = np.random.RandomState(42).permutation(n_total)
        train_idx, val_idx = indices[n_val:], indices[:n_val]
    else:
        train_idx, val_idx = np.arange(n_total), np.array([], dtype=int)

    if masked:
        # --- Self-supervised: reconstruct the (z-scored) real signal from
        # --- its occluded segments. No clean target needed.
        target_data = clean_data
        norm, _, _ = normalize_trials(target_data)
        train_dataset = MaskedDenoisingDataset(
            norm[train_idx], mask_ratio=mask_ratio)
        val_dataset = (MaskedDenoisingDataset(norm[val_idx], mask_ratio=mask_ratio)
                       if len(val_idx) > 0 else None)
    else:
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

        # --- Step 3: build train/validation split ---
        train_dataset = DenoisingDataset(noisy_norm[train_idx], clean_norm[train_idx])
        val_dataset = (DenoisingDataset(noisy_norm[val_idx], clean_norm[val_idx])
                       if len(val_idx) > 0 else None)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

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
    val_losses = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_noisy, batch_clean in train_dataloader:
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
        avg_loss = epoch_loss / len(train_dataset)
        losses.append(avg_loss)

        # Validation loss (same normalized target space)
        if val_dataset is not None and len(val_dataset) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_noisy, batch_clean in DataLoader(
                        val_dataset, batch_size=config.batch_size):
                    batch_noisy = batch_noisy.to(device)
                    batch_clean = batch_clean.to(device)
                    reconstructed = model(batch_noisy)
                    val_loss += criterion(reconstructed, batch_clean).item() * batch_noisy.shape[0]
            model.train()
            val_losses.append(val_loss / len(val_dataset))
        else:
            val_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [DAE-Train] Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.6f} | ValLoss: {val_losses[-1]:.6f}")

    model.eval()
    print(f"[DAE-Train] Training complete. Final loss: {losses[-1]:.6f} | Final val loss: {val_losses[-1]:.6f}")
    return model, losses, val_losses


def finetune_dae_classifier(model, X_input, X_target, y_labels, config=None,
                            n_epochs=None, lambda_rec=None):
    """Task-aware finetuning: reconstruction + classification multi-task loss.

    After the reconstruction (teacher-distillation) pretraining, the DAE is
    fine-tuned so the bottleneck latent is also class-discriminative:

        L = lambda_rec * MSE(recon, target)
          + (1 - lambda_rec) * CrossEntropy(head(pool(z)), y)

    The classification objective keeps task-relevant information inside the
    latent, preventing pure reconstruction from over-smoothing the features
    the downstream CSP/SVM needs. Normalization matches ``train_dae``'s
    supervised scheme (both input and target normalized by the input stats).

    Args:
        model: pretrained DenoisingAutoencoder
        X_input: ndarray (n_trials, n_channels, n_samples) - noisy input
        X_target: ndarray (n_trials, n_channels, n_samples) - teacher target
        y_labels: ndarray (n_trials,) - trial labels
        config: PipelineConfig instance
        n_epochs: finetune epochs (defaults to config.dae_finetune_epochs)
        lambda_rec: reconstruction weight (defaults to config.dae_lambda_rec)

    Returns:
        (fine-tuned model, classification head)
    """
    from torch.utils.data import TensorDataset

    if config is None:
        from config import PipelineConfig
        config = PipelineConfig()

    device = get_device()
    n_epochs = n_epochs or config.dae_finetune_epochs
    lam = lambda_rec if lambda_rec is not None else config.dae_lambda_rec

    n_classes = len(np.unique(y_labels))
    head = ClassificationHead(config.latent_dim, n_classes).to(device)

    # Same normalization as train_dae's supervised teacher scheme.
    noisy_norm, noisy_means, noisy_stds = normalize_trials(X_input)
    clean_norm = (X_target - noisy_means) / noisy_stds

    dataset = TensorDataset(
        torch.FloatTensor(noisy_norm),
        torch.FloatTensor(clean_norm),
        torch.LongTensor(np.asarray(y_labels)),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=config.dae_finetune_lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    model.train()
    head.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for bx, bt, by in loader:
            bx, bt, by = bx.to(device), bt.to(device), by.to(device)
            optimizer.zero_grad()
            recon = model(bx)
            z = model.encode(bx)
            logits = head(z)
            loss = lam * mse(recon, bt) + (1.0 - lam) * ce(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * bx.shape[0]
        print(f"  [DAE-Finetune] Epoch {epoch + 1}/{n_epochs} | "
              f"Loss: {epoch_loss / len(dataset):.6f}")

    model.eval()
    head.eval()
    return model, head
