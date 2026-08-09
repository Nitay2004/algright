import numpy as np
from scipy.signal import butter, sosfiltfilt


def generate_synthetic_eeg(n_channels=64, n_samples=250,
                           sfreq=250, duration_sec=None, class_label=0):
    """Generate synthetic EEG data with realistic oscillatory components.

    Produces signals containing delta, theta, alpha, beta, and gamma
    oscillations at physiologically plausible amplitudes.

    The ``class_label`` controls a lateralized (left/right) alpha/beta
    pattern so the synthetic data is class-discriminative for CSP-based
    motor-imagery classification:
        class 0 -> stronger alpha over left channels, beta over right
        class 1 -> stronger alpha over right channels, beta over left

    Args:
        n_channels: number of EEG channels
        n_samples: samples per trial (or total if duration_sec given)
        sfreq: sampling frequency
        duration_sec: if given, override n_samples
        class_label: 0 or 1 - lateralization pattern for this trial

    Returns:
        ndarray (n_channels, n_samples)
    """
    if duration_sec is not None:
        n_samples = int(duration_sec * sfreq)

    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    left = n_channels // 2

    # Coherent trial-level rhythms: one frequency AND phase per band per
    # trial, shared across all channels. A real EEG oscillation is a source
    # that projects onto all sensors with the same phase, differing only in
    # spatial amplitude - which is exactly the structure CSP exploits. Random
    # per-channel phases make every trial spatially unique, so the class
    # pattern learned on training trials does not transfer to test trials.
    delta_freq = np.random.uniform(1, 3)
    theta_freq = np.random.uniform(5, 7)
    alpha_freq = np.random.uniform(9.5, 10.5)
    beta_freq = np.random.uniform(15, 25)
    gamma_freq = np.random.uniform(35, 45)
    delta_phase = np.random.uniform(0, 2 * np.pi)
    theta_phase = np.random.uniform(0, 2 * np.pi)
    alpha_phase = np.random.uniform(0, 2 * np.pi)
    beta_phase = np.random.uniform(0, 2 * np.pi)
    gamma_phase = np.random.uniform(0, 2 * np.pi)

    for ch in range(n_channels):
        is_left = ch < left

        # Delta (0.5-4 Hz) - large amplitude
        delta_amp = np.random.uniform(20, 40)
        ch_signal = delta_amp * np.sin(2 * np.pi * delta_freq * t + delta_phase)

        # Theta (4-8 Hz)
        theta_amp = np.random.uniform(10, 20)
        ch_signal += theta_amp * np.sin(2 * np.pi * theta_freq * t + theta_phase)

        # Alpha (8-13 Hz) - strongly lateralized by class
        alpha_amp = np.random.uniform(5, 15)
        if class_label == 0:
            alpha_amp *= 3.0 if is_left else 0.4
        else:
            alpha_amp *= 3.0 if not is_left else 0.4
        ch_signal += alpha_amp * np.sin(2 * np.pi * alpha_freq * t + alpha_phase)

        # Beta (13-30 Hz) - lateralized opposite to alpha
        beta_amp = np.random.uniform(2, 8)
        if class_label == 0:
            beta_amp *= 2.0 if not is_left else 0.5
        else:
            beta_amp *= 2.0 if is_left else 0.5
        ch_signal += beta_amp * np.sin(2 * np.pi * beta_freq * t + beta_phase)

        # Gamma (30-50 Hz) - small amplitude
        gamma_amp = np.random.uniform(1, 4)
        ch_signal += gamma_amp * np.sin(2 * np.pi * gamma_freq * t + gamma_phase)

        data[ch] = ch_signal

    return data


def generate_eeg_trials(n_trials=100, n_channels=64, n_samples=250,
                        sfreq=250, labels=None):
    """Generate multiple synthetic EEG trials.

    Args:
        labels: optional array-like of class labels (0/1) per trial. When
            given, each trial is generated with the matching lateralization
            pattern. When None, labels are assigned at random.

    Returns:
        ndarray (n_trials, n_channels, n_samples)
    """
    trials = np.zeros((n_trials, n_channels, n_samples))
    if labels is None:
        labels = np.random.randint(0, 2, size=n_trials)
    for i in range(n_trials):
        trials[i] = generate_synthetic_eeg(
            n_channels, n_samples, sfreq, class_label=int(labels[i]))
    return trials


def inject_noise(data, noise_type="all", snr_db=-5.0, sfreq=250):
    """Inject realistic noise into EEG data.

    Args:
        data: ndarray (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        noise_type: "powerline", "emg", "eog", "common", "all"
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

    if noise_type in ("common", "all"):
        # Common-mode source: ONE broadband oscillation projected onto all
        # channels with different weights (diffuse muscle activity or amplifier
        # common-mode). Unlike the per-channel noises above, this is spatially
        # CORRELATED, so it corrupts the inter-channel covariance that CSP
        # depends on - and therefore actually hurts classification accuracy.
        src = np.random.randn(n_samples)
        sos = butter(4, 8 / (sfreq / 2), btype='high', output='sos')
        src = sosfiltfilt(sos, src)
        for ch in range(n_channels):
            noise[ch] += np.random.uniform(0.5, 1.5) * src

    # Scale noise to desired SNR
    signal_power = np.mean(data ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        snr_linear = 10 ** (snr_db / 10.0)
        scale = np.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
        noise *= scale

    return data + noise
