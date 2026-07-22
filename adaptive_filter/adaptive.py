import numpy as np
import padasip as pa


class AdaptiveFilter:
    """Adaptive filter wrapper using Padasip for LMS/RLS algorithms.

    Handles multi-channel EEG by filtering each channel independently.
    """

    def __init__(self, n_channels, method="lms", filter_length=32,
                 mu=0.01, rls_lambda=0.98, rls_delta=0.1):
        self.n_channels = n_channels
        self.method = method.lower()
        self.filter_length = filter_length
        self.mu = mu
        self.rls_lambda = rls_lambda
        self.rls_delta = rls_delta
        self.filters = []

        for _ in range(n_channels):
            if self.method == "lms":
                f = pa.filters.AdaptiveFilter(
                    model="LMS", n=filter_length, mu=mu
                )
            elif self.method == "rls":
                f = pa.filters.AdaptiveFilter(
                    model="RLS", n=filter_length, mu=mu,
                    rlamb=rls_lambda, delta=rls_delta
                )
            else:
                raise ValueError(f"Unknown method: {method}. Use 'lms' or 'rls'.")
            self.filters.append(f)

    def _make_delay_line(self, signal):
        """Create a tap-delay line matrix for padasip.

        Padasip expects x to be (N, n) where n = filter_length.
        Each column is a delayed version of the signal.

        Args:
            signal: (n_samples,)

        Returns:
            ndarray (n_samples - filter_length, filter_length)
        """
        N = len(signal)
        n = self.filter_length
        x = np.zeros((N, n))
        for i in range(n):
            x[i:, i] = signal[:N - i]
        return x[n - 1:]  # trim first (n-1) rows

    def filter_signal(self, data):
        """Filter multi-channel EEG data.

        Args:
            data: ndarray (n_channels, n_samples)

        Returns:
            ndarray (n_channels, n_samples) - filtered output
        """
        n_channels, n_samples = data.shape
        output = np.zeros_like(data)

        for ch in range(n_channels):
            signal = data[ch].copy()
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            sig_max = np.max(np.abs(signal))
            if sig_max > 1e-10:
                signal = signal / sig_max

            x = self._make_delay_line(signal)
            d = signal[self.filter_length - 1:]

            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

            y, e, w = self.filters[ch].run(d, x)

            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

            output[ch, self.filter_length - 1:] = e * sig_max

        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        return output

    def filter_online_sample(self, sample):
        """Filter a single time-step across all channels.

        Args:
            sample: ndarray (n_channels,) - single time-point

        Returns:
            ndarray (n_channels,) - filtered output
        """
        output = np.zeros(self.n_channels)
        for ch in range(self.n_channels):
            x = np.zeros((1, self.filter_length))
            y, e, w = self.filters[ch].run(np.array([sample[ch]]), x)
            output[ch] = e[0]
        return output

    def reset(self):
        """Reset all filter states."""
        self.filters = []
        for _ in range(self.n_channels):
            if self.method == "lms":
                f = pa.filters.AdaptiveFilter(
                    model="LMS", n=self.filter_length, mu=self.mu
                )
            else:
                f = pa.filters.AdaptiveFilter(
                    model="RLS", n=self.filter_length, mu=self.mu,
                    rlamb=self.rls_lambda, delta=self.rls_delta
                )
            self.filters.append(f)
