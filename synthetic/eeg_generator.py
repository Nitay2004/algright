import numpy as np
from scipy.signal import butter, sosfiltfilt


def generate_synthetic_eeg(n_channels=64, n_samples=250,
                           sfreq=250, duration_sec=None):
    """Generate synthetic EEG data with realistic oscillatory components.

    Produces signals containing delta, theta, alpha, beta, and gamma
    oscillations at physiologically plausible amplitudes.

    Args:
        n_channels: number of EEG channels
        n_samples: samples per trial (or total if duration_sec given)
        sfreq: sampling frequency
        duration_sec: if given, override n_samples

    Returns:
        ndarray (n_channels, n_samples)
    """
    if duration_sec is not None:
        n_samples = int(duration_sec * sfreq)

    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Delta (0.5-4 Hz) - large amplitude
        delta_amp = np.random.uniform(20, 40)
        delta_freq = np.random.uniform(1, 3)
        ch_signal = delta_amp * np.sin(2 * np.pi * delta_freq * t + np.random.uniform(0, 2 * np.pi))

        # Theta (4-8 Hz)
        theta_amp = np.random.uniform(10, 20)
        theta_freq = np.random.uniform(5, 7)
        ch_signal += theta_amp * np.sin(2 * np.pi * theta_freq * t + np.random.uniform(0, 2 * np.pi))

        # Alpha (8-13 Hz) - dominant in posterior regions
        alpha_amp = np.random.uniform(5, 15)
        alpha_freq = np.random.uniform(9, 11)
        ch_signal += alpha_amp * np.sin(2 * np.pi * alpha_freq * t + np.random.uniform(0, 2 * np.pi))

        # Beta (13-30 Hz)
        beta_amp = np.random.uniform(2, 8)
        beta_freq = np.random.uniform(15, 25)
        ch_signal += beta_amp * np.sin(2 * np.pi * beta_freq * t + np.random.uniform(0, 2 * np.pi))

        # Gamma (30-50 Hz) - small amplitude
        gamma_amp = np.random.uniform(1, 4)
        gamma_freq = np.random.uniform(35, 45)
        ch_signal += gamma_amp * np.sin(2 * np.pi * gamma_freq * t + np.random.uniform(0, 2 * np.pi))

        data[ch] = ch_signal

    return data


def generate_eeg_trials(n_trials=100, n_channels=64, n_samples=250,
                        sfreq=250):
    """Generate multiple synthetic EEG trials.

    Returns:
        ndarray (n_trials, n_channels, n_samples)
    """
    trials = np.zeros((n_trials, n_channels, n_samples))
    for i in range(n_trials):
        trials[i] = generate_synthetic_eeg(n_channels, n_samples, sfreq)
    return trials


def inject_noise(data, noise_type="all", snr_db=-5.0, sfreq=250):
    """Inject realistic noise into EEG data.

    Args:
        data: ndarray (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        noise_type: "powerline", "emg", "eog", "all"
        snr_db: desired signal-to-noise ratio
        sfreq: sampling frequency

    Returns:
        ndarray - noisy data, same shape as input
    """
    if data.ndim == 2:
        return _inject_noise_single(data, noise_type, snr_db, sfreq)
    elif data.ndim == 3:
        noisy = np.zeros_like(data)
        for i in range(data.shape[0]):
            noisy[i] = _inject_noise_single(data[i], noise_type, snr_db, sfreq)
        return noisy
    else:
        raise ValueError("Data must be 2D (ch, samples) or 3D (trials, ch, samples)")


def _inject_noise_single(data, noise_type, snr_db, sfreq):
    """Inject noise into a single epoch."""
    n_channels, n_samples = data.shape
    noise = np.zeros_like(data)
    t = np.arange(n_samples) / sfreq

    if noise_type in ("powerline", "all"):
        # 50Hz power-line interference
        for ch in range(n_channels):
            amp = np.random.uniform(5, 15)
            noise[ch] += amp * np.sin(2 * np.pi * 50 * t + np.random.uniform(0, 2 * np.pi))

    if noise_type in ("emg", "all"):
        # EMG (muscle) noise - high frequency broadband
        for ch in range(n_channels):
            emg = np.random.randn(n_samples) * np.random.uniform(3, 10)
            # High-pass filter to make it broadband
            sos = butter(4, 20 / (sfreq / 2), btype='high', output='sos')
            emg = sosfiltfilt(sos, emg)
            noise[ch] += emg

    if noise_type in ("eog", "all"):
        # EOG (eye blink) artifacts - intermittent sharp deflections
        for ch in range(n_channels):
            n_blinks = np.random.randint(2, 6)
            for _ in range(n_blinks):
                blink_pos = np.random.randint(0, n_samples - 30)
                blink_amp = np.random.uniform(30, 80)
                blink = blink_amp * np.exp(-0.5 * ((np.arange(30) - 15) / 5) ** 2)
                end = min(blink_pos + 30, n_samples)
                noise[ch][blink_pos:end] += blink[:end - blink_pos]

    # Scale noise to desired SNR
    signal_power = np.mean(data ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        snr_linear = 10 ** (snr_db / 10.0)
        scale = np.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
        noise *= scale

    return data + noise
