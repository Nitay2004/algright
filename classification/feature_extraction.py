import numpy as np
from mne.decoding import CSP


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
