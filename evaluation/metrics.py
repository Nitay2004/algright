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


def compute_rmse(clean, estimate):
    """Root mean squared error between clean and estimated signal."""
    return float(np.sqrt(np.mean((clean - estimate) ** 2)))


def compute_correlation(clean, estimate):
    """Pearson correlation coefficient between clean and estimated signal.

    Returns 0.0 if either signal is constant.
    """
    c = clean.ravel().astype(np.float64)
    e = estimate.ravel().astype(np.float64)
    if c.std() < 1e-12 or e.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(c, e)[0, 1])


def compute_psnr(clean, estimate, data_range=None):
    """Peak Signal-to-Noise Ratio in dB.

    Args:
        clean: ndarray - reference signal
        estimate: ndarray - reconstructed signal
        data_range: peak-to-peak value (defaults to clean range)
    """
    mse = np.mean((clean - estimate) ** 2)
    if mse < 1e-20:
        return float('inf')
    if data_range is None:
        data_range = np.max(clean) - np.min(clean)
    if data_range <= 0:
        data_range = 1.0
    return float(20 * np.log10(data_range / np.sqrt(mse)))


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
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "runs_ms": times,
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

    # Per-trial averaged denoising metrics (against clean ground truth)
    snr_before_list = []
    snr_after_list = []
    rmse_noisy_list = []
    rmse_denoised_list = []
    corr_noisy_list = []
    corr_denoised_list = []
    psnr_noisy_list = []
    psnr_denoised_list = []

    for i in range(len(clean_data)):
        for ch in range(clean_data.shape[1]):
            clean = clean_data[i, ch]
            noisy = noisy_data[i, ch]
            denoised = denoised_data[i, ch]

            snr_before_list.append(compute_snr(clean, noisy - clean))
            snr_after_list.append(compute_snr(clean, denoised - clean))
            rmse_noisy_list.append(compute_rmse(clean, noisy))
            rmse_denoised_list.append(compute_rmse(clean, denoised))
            corr_noisy_list.append(compute_correlation(clean, noisy))
            corr_denoised_list.append(compute_correlation(clean, denoised))
            psnr_noisy_list.append(compute_psnr(clean, noisy))
            psnr_denoised_list.append(compute_psnr(clean, denoised))

    results["avg_snr_before"] = float(np.nanmean(snr_before_list))
    results["avg_snr_after"] = float(np.nanmean(snr_after_list))
    results["avg_snr_improvement_db"] = results["avg_snr_after"] - results["avg_snr_before"]
    results["snr_before"] = results["avg_snr_before"]
    results["snr_after"] = results["avg_snr_after"]

    results["mse_noisy"] = float(np.mean((noisy_data - clean_data) ** 2))
    results["mse_denoised"] = float(np.mean((denoised_data - clean_data) ** 2))
    results["rmse_noisy"] = float(np.sqrt(results["mse_noisy"]))
    results["rmse_denoised"] = float(np.sqrt(results["mse_denoised"]))
    results["corr_noisy"] = float(np.nanmean(corr_noisy_list))
    results["corr_denoised"] = float(np.nanmean(corr_denoised_list))
    results["psnr_noisy"] = float(np.nanmean(psnr_noisy_list))
    results["psnr_denoised"] = float(np.nanmean(psnr_denoised_list))

    # Classification metrics (if provided)
    if classifier is not None and features is not None and labels is not None:
        y_pred = classifier.predict(features)
        acc = classifier.score(features, labels)
        results["classification_accuracy"] = float(acc)
        results["classification_accuracy_before"] = None  # set by caller if needed
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            results["precision"] = float(precision_score(labels, y_pred, average="macro", zero_division=0))
            results["recall"] = float(recall_score(labels, y_pred, average="macro", zero_division=0))
            results["f1_score"] = float(f1_score(labels, y_pred, average="macro", zero_division=0))
        except Exception:
            results["precision"] = float('nan')
            results["recall"] = float('nan')
            results["f1_score"] = float('nan')
        results["y_pred"] = y_pred

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
        print(f"  RMSE (Noisy)         : {results['rmse_noisy']:.6f}")
        print(f"  RMSE (Denoised)      : {results['rmse_denoised']:.6f}")
        print(f"  Corr (Noisy)         : {results['corr_noisy']:.4f}")
        print(f"  Corr (Denoised)      : {results['corr_denoised']:.4f}")
        print(f"  PSNR (Noisy)         : {results['psnr_noisy']:.2f} dB")
        print(f"  PSNR (Denoised)      : {results['psnr_denoised']:.2f} dB")

    if "classification_accuracy" in results:
        print(f"  Classification Acc   : {results['classification_accuracy']:.4f}")
        print(f"  Precision            : {results.get('precision', float('nan')):.4f}")
        print(f"  Recall               : {results.get('recall', float('nan')):.4f}")
        print(f"  F1-Score             : {results.get('f1_score', float('nan')):.4f}")

    print("=" * 55 + "\n")


def evaluate_stages(stages, clean_data, output_dir="results", stage_accs=None):
    """Compute denoising metrics for every pipeline stage and print a table.

    Args:
        stages: dict {stage_name: signal ndarray (n_trials, n_channels, n_samples)}
        clean_data: ndarray - ground truth clean signal (same shape)
        output_dir: folder where a CSV copy is saved
        stage_accs: optional dict {stage_name: classification accuracy} to
            include an Accuracy column in the consolidated table

    Returns:
        list of dicts (one per stage) with all metrics
    """
    import csv
    import os

    rows = []
    for name, signal in stages.items():
        row = {
            "stage": name,
            "snr_db": float(np.nanmean([compute_snr(clean_data[i, ch], signal[i, ch] - clean_data[i, ch])
                                        for i in range(len(clean_data))
                                        for ch in range(clean_data.shape[1])])),
            "rmse": float(np.sqrt(np.mean((signal - clean_data) ** 2))),
            "mse": float(np.mean((signal - clean_data) ** 2)),
            "correlation": float(np.nanmean([compute_correlation(clean_data[i, ch], signal[i, ch])
                                             for i in range(len(clean_data))
                                             for ch in range(clean_data.shape[1])])),
            "psnr_db": float(np.nanmean([compute_psnr(clean_data[i, ch], signal[i, ch])
                                         for i in range(len(clean_data))
                                         for ch in range(clean_data.shape[1])])),
        }
        if stage_accs is not None and name in stage_accs:
            row["accuracy"] = stage_accs[name]
        rows.append(row)

    header = ["Stage", "SNR (dB)", "RMSE", "MSE", "Correlation", "PSNR (dB)"]
    if stage_accs is not None:
        header.append("Accuracy")
    widths = [len(h) for h in header]

    print("\n" + "=" * 90)
    print("  STAGE-BY-STAGE QUANTITATIVE COMPARISON (vs Clean Ground Truth)")
    print("=" * 90)
    for r in rows:
        widths[0] = max(widths[0], len(r["stage"]))
    line = "  " + " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(header))
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        parts = [
            f"{r['stage']:<{widths[0]}}",
            f"{r['snr_db']:>{widths[1]}.2f}",
            f"{r['rmse']:>{widths[2]}.4f}",
            f"{r['mse']:>{widths[3]}.6f}",
            f"{r['correlation']:>{widths[4]}.4f}",
            f"{r['psnr_db']:>{widths[5]}.2f}",
        ]
        if stage_accs is not None:
            parts.append(f"{r.get('accuracy', float('nan')):>{widths[6]}.4f}")
        print("  " + " | ".join(parts))
    print("=" * 90 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics_stage_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            out = [r["stage"], f"{r['snr_db']:.2f}", f"{r['rmse']:.4f}",
                   f"{r['mse']:.6f}", f"{r['correlation']:.4f}", f"{r['psnr_db']:.2f}"]
            if stage_accs is not None:
                out.append(f"{r.get('accuracy', float('nan')):.4f}")
            writer.writerow(out)
    print(f"  Table saved: {csv_path}")
    return rows
