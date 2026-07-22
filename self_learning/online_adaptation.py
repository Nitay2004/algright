import numpy as np
import torch
from collections import deque


class OnlineAdaptationEngine:
    """Self-learning engine for continuous model adaptation.

    Implements the feedback loop:
    1. System makes a classification
    2. Receives confidence score or true label
    3. If confidence is low OR distribution shift detected, trigger adaptation
    4. Update DAE weights using mini-batch online learning
    """

    def __init__(self, denoiser, classifier=None,
                 confidence_threshold=0.7,
                 shift_detector_threshold=0.15,
                 adaptation_interval=10,
                 online_lr=1e-4,
                 mini_batch_size=16):
        """
        Args:
            denoiser: OnlineDenoiser instance (PyTorch model)
            classifier: EEGClassifier instance
            confidence_threshold: below this, trigger adaptation
            shift_detector_threshold: KL-div threshold for distribution shift
            adaptation_interval: epochs between forced adaptation
            online_lr: learning rate for online updates
            mini_batch_size: mini-batch size for online updates
        """
        self.denoiser = denoiser
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.shift_threshold = shift_detector_threshold
        self.adaptation_interval = adaptation_interval
        self.online_lr = online_lr
        self.mini_batch_size = mini_batch_size

        self.epoch_buffer = deque(maxlen=mini_batch_size * 5)
        self.adaptation_count = 0
        self.total_adaptations = 0
        self.shift_history = []

    def _detect_distribution_shift(self, current_epoch):
        """Simple distribution shift detection using running statistics.

        Compares current epoch's power spectrum to stored reference.
        Returns True if shift is detected.
        """
        if len(self.epoch_buffer) < 10:
            return False

        # Compute power spectral density of current epoch
        current_psd = np.mean(np.abs(np.fft.rfft(current_epoch.astype(np.float64), axis=-1)) ** 2, axis=0)

        # Compute reference PSD from buffer
        buffer_psd = np.array([
            np.mean(np.abs(np.fft.rfft(ep.astype(np.float64), axis=-1)) ** 2, axis=0)
            for ep in list(self.epoch_buffer)[-20:]
        ])
        ref_psd = np.mean(buffer_psd, axis=0)

        # Normalize (use log to avoid overflow)
        current_psd = np.log1p(current_psd)
        ref_psd = np.log1p(ref_psd)
        current_psd = current_psd / (np.sum(current_psd) + 1e-10)
        ref_psd = ref_psd / (np.sum(ref_psd) + 1e-10)

        # KL divergence (simplified)
        eps = 1e-10
        kl = np.sum(current_psd * np.log((current_psd + eps) / (ref_psd + eps)))

        self.shift_history.append(kl)
        return kl > self.shift_threshold

    def _compute_confidence(self, features):
        """Get classification confidence score."""
        if self.classifier is None or not self.classifier.is_fitted:
            return 1.0  # Assume confident if no classifier

        proba = self.classifier.predict_proba(features)
        return np.max(proba, axis=-1).mean()

    def process_epoch(self, epoch_data, features=None, true_label=None,
                      clean_target=None):
        """Process an epoch and decide whether to adapt.

        Args:
            epoch_data: ndarray (n_channels, n_samples) - denoised epoch
            features: ndarray (1, n_features) - extracted features
            true_label: int or None - ground truth label
            clean_target: ndarray or None - clean target for supervised adaptation

        Returns:
            dict with adaptation decision and results
        """
        self.epoch_buffer.append(epoch_data)
        self.adaptation_count += 1

        result = {
            "adapted": False,
            "shift_detected": False,
            "low_confidence": False,
            "force_adapt": False,
            "loss": None
        }

        # Check adaptation triggers
        shift = self._detect_distribution_shift(epoch_data)
        result["shift_detected"] = shift

        low_conf = False
        if features is not None:
            conf = self._compute_confidence(features)
            low_conf = conf < self.confidence_threshold
            result["low_confidence"] = low_conf

        force = (self.adaptation_count % self.adaptation_interval == 0)
        result["force_adapt"] = force

        # Decide whether to adapt
        should_adapt = shift or low_conf or force

        if should_adapt and clean_target is not None:
            loss = self.denoiser.online_adapt_step(
                epoch_data, clean_target, lr=self.online_lr
            )
            result["adapted"] = True
            result["loss"] = loss
            self.total_adaptations += 1

        return result

    def get_stats(self):
        """Return adaptation statistics."""
        return {
            "total_epochs_processed": len(self.epoch_buffer),
            "total_adaptations": self.total_adaptations,
            "avg_shift": np.mean(self.shift_history) if self.shift_history else 0.0
        }
