import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt, tf2sos


def apply_bandpass(data, lowcut=0.5, highcut=50.0, sfreq=250, order=5):
    """Apply zero-phase Butterworth bandpass filter.

    Args:
        data: ndarray (n_channels, n_samples) or (n_samples,)
        lowcut: lower cutoff Hz
        highcut: upper cutoff Hz
        sfreq: sampling frequency
        order: filter order

    Returns:
        Filtered signal, same shape as input.
    """
    nyq = sfreq / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=-1)


def apply_notch(data, freq=50.0, sfreq=250, quality=30.0):
    """Apply notch filter to remove power-line interference.

    Args:
        data: ndarray (n_channels, n_samples) or (n_samples,)
        freq: notch center frequency (50 or 60 Hz)
        sfreq: sampling frequency
        quality: quality factor (higher = narrower notch)

    Returns:
        Filtered signal.
    """
    b, a = iirnotch(freq, quality, sfreq)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, data, axis=-1)
