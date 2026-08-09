import numpy as np
from mne.decoding import CSP

from preprocessing import apply_bandpass


def extract_csp_features(data, labels, n_components=8):
    """Extract Common Spatial Pattern features.

    Args:
        data: ndarray (n_trials, n_channels, n_samples)
        labels: ndarray (n_trials,)
        n_components: number of CSP components

    Returns:
        features: ndarray (n_trials, n_components * 2)
        csp: fitted CSP object (for reuse on test data)
    """
    csp = CSP(n_components=n_components, reg=0.1, log=True, norm_trace=False)
    features = csp.fit_transform(data, labels)
    return features, csp


def transform_csp_features(data, csp):
    """Transform new data using a fitted CSP object."""
    return csp.transform(data)


class FBCSP:
    """Filter-Bank Common Spatial Patterns (Ang et al., 2008).

    Motor-imagery discriminative activity lives in narrow bands (mu 8-12 Hz,
    beta 13-30 Hz). FBCSP band-passes each band, fits a CSP per band and stacks
    the log-variance features. This is the standard feature front-end used by
    BCI Competition IV winners and consistently beats single full-band CSP on
    real motor-imagery data.
    """

    def __init__(self, n_components=8, sfreq=160.0, bands=None):
        self.n_components = n_components
        self.sfreq = sfreq
        self.bands = bands if bands is not None else [(8.0, 12.0), (13.0, 30.0)]

    def fit(self, data, labels):
        """Fit a CSP per frequency band."""
        self.csps_ = []
        for low, high in self.bands:
            band_data = apply_bandpass(data, low, high, self.sfreq)
            _, csp = extract_csp_features(band_data, labels, self.n_components)
            self.csps_.append(csp)
        return self

    def transform(self, data):
        """Extract stacked band-wise CSP features."""
        feats = []
        for (low, high), csp in zip(self.bands, self.csps_):
            band_data = apply_bandpass(data, low, high, self.sfreq)
            feats.append(csp.transform(band_data))
        return np.hstack(feats)

    def fit_transform(self, data, labels):
        """Fit and extract features in one call."""
        self.fit(data, labels)
        return self.transform(data)
