import numpy as np
import time


def compute_snr(signal, noise):
    """Compute Signal-to-Noise Ratio in dB.

    Args:
        signal: ndarray - clean/reference signal
        noise: ndarray - noise component

    Returns:
        float - SNR in dB
    """
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return float('inf')
    return 10 * np.log10(sig_power / noise_power)


def compute_snr_improvement(clean, noisy, denoised):
    """Compute SNR improvement after denoising.

    Args:
        clean: ndarray - ground truth clean signal
        noisy: ndarray - noisy input
        denoised: ndarray - denoised output

    Returns:
        dict with before/after SNR and improvement
    """
    snr_before = compute_snr(clean, noisy - clean)
    snr_after = compute_snr(clean, denoised - clean)

    return {
        "snr_before": snr_before,
        "snr_after": snr_after,
        "snr_improvement_db": snr_after - snr_before
    }


def measure_latency(func, *args, n_runs=10, **kwargs):
    """Measure average execution latency of a function.

    Args:
        func: callable to benchmark
        *args, **kwargs: arguments to pass

    Returns:
        dict with latency stats in milliseconds
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "result": result
    }


def evaluate_pipeline(noisy_data, denoised_data, clean_data,
                      classifier=None, features=None, labels=None):
    """Comprehensive evaluation of the denoising pipeline.

    Args:
        noisy_data: ndarray (n_trials, n_channels, n_samples)
        denoised_data: ndarray (n_trials, n_channels, n_samples)
        clean_data: ndarray (n_trials, n_channels, n_samples)
        classifier: trained EEGClassifier
        features: ndarray (n_trials, n_features) - extracted features from denoised
        labels: ndarray (n_trials,)

    Returns:
        dict with all metrics
    """
    results = {}

    # SNR metrics (averaged over trials and channels)
    snr_before_list = []
    snr_after_list = []

    for i in range(len(clean_data)):
        for ch in range(clean_data.shape[1]):
            snr_b = compute_snr(
                clean_data[i, ch],
                noisy_data[i, ch] - clean_data[i, ch]
            )
            snr_a = compute_snr(
                clean_data[i, ch],
                denoised_data[i, ch] - clean_data[i, ch]
            )
            snr_before_list.append(snr_b)
            snr_after_list.append(snr_a)

    results["avg_snr_before"] = np.mean(snr_before_list)
    results["avg_snr_after"] = np.mean(snr_after_list)
    results["avg_snr_improvement_db"] = results["avg_snr_after"] - results["avg_snr_before"]

    # MSE
    results["mse_noisy"] = np.mean((noisy_data - clean_data) ** 2)
    results["mse_denoised"] = np.mean((denoised_data - clean_data) ** 2)

    # Classification accuracy (if provided)
    if classifier is not None and features is not None and labels is not None:
        acc = classifier.score(features, labels)
        results["classification_accuracy"] = acc

    return results


def print_evaluation(results):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 55)
    print("  PIPELINE EVALUATION RESULTS")
    print("=" * 55)

    if "avg_snr_before" in results:
        print(f"  SNR Before Denoising : {results['avg_snr_before']:.2f} dB")
        print(f"  SNR After Denoising  : {results['avg_snr_after']:.2f} dB")
        print(f"  SNR Improvement      : {results['avg_snr_improvement_db']:.2f} dB")

    if "mse_noisy" in results:
        print(f"  MSE (Noisy)          : {results['mse_noisy']:.6f}")
        print(f"  MSE (Denoised)       : {results['mse_denoised']:.6f}")

    if "classification_accuracy" in results:
        print(f"  Classification Acc   : {results['classification_accuracy']:.4f}")

    print("=" * 55 + "\n")
