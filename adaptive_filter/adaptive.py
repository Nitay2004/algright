import numpy as np
import padasip as pa


class AdaptiveFilter:
    """Adaptive Line-Noise Canceller (ANC) using Padasip LMS/RLS.

    Uses a *synthesized* reference signal at the mains frequency (sin/cos
    pair at ``notch_freq``) so no secondary noise sensor is required. For
    each channel the filter estimates the line-noise component present in
    the signal and subtracts it:

        output = signal - y(estimated line-noise)

    This removes narrowband (50/60 Hz) interference without destroying the
    EEG itself. (A prediction-error / delay-line setup, where the error is
    returned as the output, whitens the signal and collapses classification;
    that is intentionally avoided here.)

    Because the reference is shared across channels and has only two taps,
    ``filter_length`` is overridden to match the reference dimensionality.
    """

    def __init__(self, n_channels, method="lms", filter_length=2,
                 mu=0.02, rls_lambda=0.98, rls_delta=0.1,
                 sfreq=250.0, notch_freq=50.0):
        self.n_channels = n_channels
        self.method = method.lower()
        self.filter_length = 2  # sin + cos reference taps
        self.mu = mu
        self.rls_lambda = rls_lambda
        self.rls_delta = rls_delta
        self.sfreq = float(sfreq)
        self.notch_freq = float(notch_freq)
        self._n_processed = 0
        self.filters = []

        for _ in range(n_channels):
            self.filters.append(self._new_filter())

    def _new_filter(self):
        if self.method == "lms":
            return pa.filters.AdaptiveFilter(
                model="LMS", n=self.filter_length, mu=self.mu)
        elif self.method == "rls":
            return pa.filters.AdaptiveFilter(
                model="RLS", n=self.filter_length, mu=self.mu,
                rlamb=self.rls_lambda, delta=self.rls_delta)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'lms' or 'rls'.")

    def _reference(self, n_samples):
        """Synthesized mains-frequency reference: (n_samples, 2).

        A sin/cos pair tracks both amplitude and phase of the line noise
        simultaneously, so the ANC can converge without a dedicated sensor.
        """
        t = np.arange(n_samples) / self.sfreq
        omega = 2.0 * np.pi * self.notch_freq * t
        return np.stack([np.sin(omega), np.cos(omega)], axis=1)

    def filter_signal(self, data):
        """Filter multi-channel EEG data with the adaptive line-noise canceller.

        Args:
            data: ndarray (n_channels, n_samples)

        Returns:
            ndarray (n_channels, n_samples) - signal minus estimated line-noise
        """
        n_channels, n_samples = data.shape
        output = np.zeros_like(data)
        x = self._reference(n_samples)

        for ch in range(n_channels):
            signal = np.nan_to_num(data[ch], nan=0.0, posinf=0.0, neginf=0.0)
            sig_max = np.max(np.abs(signal))
            if sig_max > 1e-10:
                signal = signal / sig_max

            # desired = (unit-scale) signal; adaptive filter estimates the
            # line-noise y; error e = d - y is the cleaned signal.
            y, e, w = self.filters[ch].run(signal, x)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

            output[ch] = e * sig_max

        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

    def filter_online_sample(self, sample):
        """Filter a single time-step across all channels.

        Args:
            sample: ndarray (n_channels,) - single time-point

        Returns:
            ndarray (n_channels,) - cleaned output
        """
        t = self._n_processed / self.sfreq
        ref = np.array([np.sin(2.0 * np.pi * self.notch_freq * t),
                        np.cos(2.0 * np.pi * self.notch_freq * t)]).reshape(1, -1)
        self._n_processed += 1

        output = np.zeros(self.n_channels)
        for ch in range(self.n_channels):
            d = np.array([sample[ch]])
            y, e, w = self.filters[ch].run(d, ref)
            output[ch] = e[0]
        return output

    def reset(self):
        """Reset all filter states."""
        self._n_processed = 0
        self.filters = []
        for _ in range(self.n_channels):
            self.filters.append(self._new_filter())
