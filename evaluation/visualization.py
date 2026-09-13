"""
Comprehensive Visualization Suite for Adaptive EEG Denoising Pipeline

Generates publication-quality figures for PhD thesis / conference presentation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "noisy": "#e74c3c",
    "clean": "#2ecc71",
    "denoised": "#3498db",
    "adapted": "#f39c12",
    "primary": "#2c3e50",
    "secondary": "#9b59b6",
    "accent": "#1abc9c",
    "bg": "#f8f9fa",
    "highlight": "#e74c3c",
}


def ensure_output_dir(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _fig_path(out, label, filename):
    """Build figure path with an optional label prefix (e.g. dataset name)."""
    if label:
        return os.path.join(out, f"{label}_{filename}")
    return os.path.join(out, filename)


def robust_ylim(*signals, lo_pct=1.0, hi_pct=99.0, pad=0.1):
    """Robust shared y-limits across signals (ignores extreme outliers)."""
    vals = np.concatenate([np.asarray(s).ravel() for s in signals])
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    span = max(hi - lo, 1e-12)
    return (lo - pad * span, hi + pad * span)


def fig1_raw_eeg_comparison(X_noisy, X_clean, X_denoised, sfreq,
                             channel_idx=0, trial_idx=0, output_dir="results",
                             clean_label="Ground Truth (Clean)", label=""):
    """Fig 1: Multi-stage EEG signal comparison - noisy vs clean vs denoised."""
    out = ensure_output_dir(output_dir)
    t_noisy = np.arange(X_noisy.shape[2]) / sfreq
    t_clean = np.arange(X_clean.shape[2]) / sfreq
    t_denoised = np.arange(X_denoised.shape[2]) / sfreq

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Figure 1: Multi-Stage EEG Signal Comparison", fontweight="bold", y=0.98)

    axes[0].plot(t_noisy, X_noisy[trial_idx, channel_idx], color=COLORS["noisy"], linewidth=0.6, alpha=0.85)
    axes[0].set_ylabel("Amplitude (\u00b5V)")
    axes[0].set_title("(a) Noisy Input Signal")
    axes[0].fill_between(t_noisy, X_noisy[trial_idx, channel_idx], alpha=0.15, color=COLORS["noisy"])

    axes[1].plot(t_denoised, X_denoised[trial_idx, channel_idx], color=COLORS["denoised"], linewidth=0.6, alpha=0.85)
    axes[1].set_ylabel("Amplitude (\u00b5V)")
    axes[1].set_title("(b) DAE-Denoised Signal")
    axes[1].fill_between(t_denoised, X_denoised[trial_idx, channel_idx], alpha=0.15, color=COLORS["denoised"])

    axes[2].plot(t_clean, X_clean[trial_idx, channel_idx], color=COLORS["clean"], linewidth=0.6, alpha=0.85)
    axes[2].set_ylabel("Amplitude (\u00b5V)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title(f"(c) {clean_label}")
    axes[2].fill_between(t_clean, X_clean[trial_idx, channel_idx], alpha=0.15, color=COLORS["clean"])

    for ax in axes:
        ax.set_xlim([0, t_noisy[-1]])

    ylo, yhi = robust_ylim(
        X_noisy[trial_idx, channel_idx], X_denoised[trial_idx, channel_idx],
        X_clean[trial_idx, channel_idx])
    for ax in axes:
        ax.set_ylim([ylo, yhi])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = _fig_path(out, label, "fig1_eeg_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig2_pipeline_stages(X_noisy, X_pre, X_adapted, X_denoised, X_clean,
                          sfreq, channel_idx=0, trial_idx=0, output_dir="results",
                          clean_label="Ground Truth (Clean)", label=""):
    """Fig 2: Full pipeline stage progression."""
    out = ensure_output_dir(output_dir)
    stages = [
        ("(a) Raw Noisy", X_noisy, COLORS["noisy"]),
        ("(b) Bandpass + Notch", X_pre, COLORS["primary"]),
        ("(c) LMS Adaptive Filter", X_adapted, COLORS["adapted"]),
        ("(d) DAE Neural Denoiser", X_denoised, COLORS["denoised"]),
        (f"(e) {clean_label}", X_clean, COLORS["clean"]),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Figure 2: Signal Progression Through Pipeline Stages",
                 fontweight="bold", y=0.99)

    t = np.arange(stages[0][1].shape[2]) / sfreq

    for i, (title, data, color) in enumerate(stages):
        sig = data[trial_idx, channel_idx]
        axes[i].plot(t, sig, color=color, linewidth=0.5, alpha=0.85)
        axes[i].fill_between(t, sig, alpha=0.1, color=color)
        axes[i].set_ylabel("\u00b5V", fontsize=9)
        axes[i].set_title(title, fontsize=10, loc="left", fontweight="bold")
        axes[i].set_xlim([0, t[-1]])

    ylo, yhi = robust_ylim(*[st[1][trial_idx, channel_idx] for st in stages])
    for ax in axes:
        ax.set_ylim([ylo, yhi])

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = _fig_path(out, label, "fig2_pipeline_stages.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig3_multi_channel_heatmap(X_data, title_str, sfreq, output_dir="results",
                                 n_channels=None, trial_idx=0, vmin=None, vmax=None,
                                 label=""):
    """Fig 3: Multi-channel heatmap (topoplot-style).

    Args:
        vmin/vmax: shared color-scale limits. When provided, all heatmaps in a
            comparison use the SAME scale so they can be compared directly.
    """
    out = ensure_output_dir(output_dir)
    if n_channels is None:
        n_channels = min(X_data.shape[1], 20)

    fig, ax = plt.subplots(figsize=(14, 6))
    data_slice = X_data[trial_idx, :n_channels, :]
    t = np.arange(data_slice.shape[1]) / sfreq

    if vmin is None or vmax is None:
        vmin = -np.std(data_slice) * 2
        vmax = np.std(data_slice) * 2

    im = ax.imshow(data_slice, aspect="auto", cmap="RdBu_r",
                   extent=[t[0], t[-1], n_channels, 0],
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel Index")
    ax.set_title(f"Figure 3: {title_str} - Multi-Channel Heatmap", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Amplitude (\u00b5V)")
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels([f"Ch {i+1}" for i in range(n_channels)], fontsize=7)

    plt.tight_layout()
    path = _fig_path(out, label, f"fig3_heatmap_{title_str.replace(' ', '_').replace(':', '')}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig4_frequency_spectrum(X_noisy, X_denoised, X_clean, sfreq,
                              channel_idx=0, output_dir="results",
                              clean_label="Clean", label=""):
    """Fig 4: Power Spectral Density comparison."""
    out = ensure_output_dir(output_dir)

    from scipy.signal import welch

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 4: Power Spectral Density Analysis", fontweight="bold", y=1.02)

    n_avg = min(X_noisy.shape[0], 20)
    psd_noisy_list, psd_denoised_list, psd_clean_list = [], [], []
    for i in range(n_avg):
        f, p_n = welch(X_noisy[i, channel_idx], fs=sfreq, nperseg=min(256, X_noisy.shape[2]))
        _, p_d = welch(X_denoised[i, channel_idx], fs=sfreq, nperseg=min(256, X_denoised.shape[2]))
        _, p_c = welch(X_clean[i, channel_idx], fs=sfreq, nperseg=min(256, X_clean.shape[2]))
        psd_noisy_list.append(p_n)
        psd_denoised_list.append(p_d)
        psd_clean_list.append(p_c)

    psd_noisy = np.mean(psd_noisy_list, axis=0)
    psd_denoised = np.mean(psd_denoised_list, axis=0)
    psd_clean = np.mean(psd_clean_list, axis=0)

    axes[0].semilogy(f, psd_noisy, color=COLORS["noisy"], linewidth=1.2, label="Noisy", alpha=0.8)
    axes[0].semilogy(f, psd_denoised, color=COLORS["denoised"], linewidth=1.2, label="Denoised", alpha=0.8)
    axes[0].semilogy(f, psd_clean, color=COLORS["clean"], linewidth=1.2, label=clean_label, linestyle="--", alpha=0.8)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (\u00b5V\u00b2/Hz)")
    axes[0].set_title("(a) PSD Comparison")
    axes[0].legend(framealpha=0.9)
    axes[0].set_xlim([0, 60])

    freq_bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13),
                  "Beta": (13, 30), "Gamma": (30, 50)}
    band_names = list(freq_bands.keys())

    band_power_noisy = []
    band_power_denoised = []
    band_power_clean = []
    for bname, (lo, hi) in freq_bands.items():
        idx = np.logical_and(f >= lo, f <= hi)
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        band_power_noisy.append(_trapz(psd_noisy[idx], f[idx]))
        band_power_denoised.append(_trapz(psd_denoised[idx], f[idx]))
        band_power_clean.append(_trapz(psd_clean[idx], f[idx]))

    x_pos = np.arange(len(band_names))
    width = 0.25
    axes[1].bar(x_pos - width, band_power_noisy, width, color=COLORS["noisy"], label="Noisy", alpha=0.8)
    axes[1].bar(x_pos, band_power_denoised, width, color=COLORS["denoised"], label="Denoised", alpha=0.8)
    axes[1].bar(x_pos + width, band_power_clean, width, color=COLORS["clean"], label=clean_label, alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(band_names)
    axes[1].set_ylabel("Band Power (\u00b5V\u00b2)")
    axes[1].set_title("(b) Band Power Distribution")
    axes[1].legend(framealpha=0.9)
    axes[1].set_yscale("log")

    plt.tight_layout()
    path = _fig_path(out, label, "fig4_frequency_spectrum.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig5_dae_training(losses, val_losses=None, output_dir="results", label=""):
    """Fig 5: DAE training convergence curve."""
    out = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(losses) + 1)

    ax.plot(epochs, losses, color=COLORS["primary"], linewidth=2, marker="o",
            markersize=4, markerfacecolor=COLORS["accent"], markeredgecolor="white",
            markeredgewidth=0.8, label="Training Loss")

    if val_losses is not None and len(val_losses) == len(losses):
        ax.plot(epochs, val_losses, color=COLORS["highlight"], linewidth=2,
                marker="s", markersize=4, markerfacecolor=COLORS["secondary"],
                markeredgecolor="white", markeredgewidth=0.8, label="Validation Loss")

    min_loss_idx = np.argmin(losses)
    ax.axvline(x=min_loss_idx + 1, color=COLORS["highlight"], linestyle="--",
               alpha=0.7, label=f"Best: Epoch {min_loss_idx+1} ({losses[min_loss_idx]:.6f})")
    ax.scatter([min_loss_idx + 1], [losses[min_loss_idx]], color=COLORS["highlight"],
               s=100, zorder=5, edgecolors="white", linewidths=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Figure 5: Denoising Autoencoder Training Convergence", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.set_yscale("log")

    plt.tight_layout()
    path = _fig_path(out, label, "fig5_dae_training.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig6_snr_comparison(metrics_dict, output_dir="results", label=""):
    """Fig 6: SNR improvement visualization."""
    out = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 6: Signal Quality Metrics", fontweight="bold", y=1.02)

    snr_before = metrics_dict.get("snr_before", 0)
    snr_after = metrics_dict.get("snr_after", 0)
    mse_noisy = metrics_dict.get("mse_noisy", 0)
    mse_denoised = metrics_dict.get("mse_denoised", 0)

    categories = ["Before Denoising", "After Denoising"]
    snr_vals = [snr_before, snr_after]
    bar_colors = [COLORS["noisy"], COLORS["denoised"]]

    bars = axes[0].bar(categories, snr_vals, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("SNR (dB)")
    axes[0].set_title("(a) Signal-to-Noise Ratio")
    for bar, val in zip(bars, snr_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.2f} dB", ha="center", fontweight="bold", fontsize=11)

    if snr_after > snr_before:
        improvement = snr_after - snr_before
        axes[0].annotate(f"+{improvement:.2f} dB", xy=(1, snr_after),
                         xytext=(1.3, snr_after * 0.5),
                         arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=2),
                         fontsize=12, color=COLORS["accent"], fontweight="bold")

    mse_cats = ["Noisy", "Denoised"]
    mse_vals = [mse_noisy, mse_denoised]
    mse_colors = [COLORS["noisy"], COLORS["denoised"]]
    bars2 = axes[1].bar(mse_cats, mse_vals, color=mse_colors, width=0.5, edgecolor="white", linewidth=1.5)
    axes[1].set_ylabel("MSE")
    axes[1].set_title("(b) Mean Squared Error")
    for bar, val in zip(bars2, mse_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", fontweight="bold", fontsize=11)

    plt.tight_layout()
    path = _fig_path(out, label, "fig6_snr_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig7_confusion_matrix(y_true, y_pred, output_dir="results", label=""):
    """Fig 7: Confusion matrix heatmap."""
    out = ensure_output_dir(output_dir)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Class 0", "Class 1"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)

    ax.set_title("Figure 7: Classification Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    path = _fig_path(out, label, "fig7_confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig8_roc_curve(y_true, y_scores, output_dir="results", label=""):
    """Fig 8: ROC curve with AUC."""
    out = ensure_output_dir(output_dir)

    if y_scores.ndim > 1 and y_scores.shape[1] >= 2:
        y_score = y_scores[:, 1]
    else:
        y_score = y_scores.ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--",
            label="Random Classifier")

    idx_optimal = np.argmax(tpr - fpr)
    ax.scatter(fpr[idx_optimal], tpr[idx_optimal], color=COLORS["highlight"],
               s=120, zorder=5, edgecolors="white", linewidths=2,
               label=f"Optimal Threshold ({thresholds[idx_optimal]:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 8: Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    path = _fig_path(out, label, "fig8_roc_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig9_latency_benchmark(latency_stats, output_dir="results", label=""):
    """Fig 9: Real inference latency per pipeline stage (measured, not estimated)."""
    out = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 9: Real-Time Inference Latency", fontweight="bold", y=1.02)

    breakdown = latency_stats.get("breakdown", {})
    runs = latency_stats.get("runs_ms", None)

    # Panel (a): real measured per-stage breakdown
    stage_labels = {
        "preprocessing": "Bandpass+Notch",
        "lms": "LMS\nAdaptive",
        "dae": "DAE\nInference",
        "csp": "CSP\nTransform",
        "svm": "SVM\nPredict",
    }
    categories = []
    latencies = []
    for key in ["preprocessing", "lms", "dae", "csp", "svm"]:
        if key in breakdown and breakdown[key] is not None:
            categories.append(stage_labels.get(key, key))
            latencies.append(breakdown[key])
    if "total" in breakdown and breakdown["total"] is not None:
        categories.append("Total\nPipeline")
        latencies.append(breakdown["total"])

    if categories:
        bar_colors = [COLORS["adapted"], COLORS["denoised"], COLORS["secondary"],
                      COLORS["accent"], COLORS["primary"], COLORS["highlight"]]
        bars = axes[0].bar(categories, latencies, color=bar_colors[:len(categories)],
                           width=0.6, edgecolor="white", linewidth=1.5)
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_title("(a) Measured Per-Stage Latency")
        for bar, val in zip(bars, latencies):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                         f"{val:.2f} ms", ha="center", fontweight="bold", fontsize=9)
        axes[0].axhline(y=100, color=COLORS["highlight"], linestyle="--", linewidth=1.5,
                        label="100 ms Real-Time Threshold")
        axes[0].legend(framealpha=0.9)
    else:
        axes[0].text(0.5, 0.5, "No measured latency breakdown available",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].axis("off")

    # Panel (b): real per-run latencies (repeated DAE inference measurements)
    if runs is not None and len(runs) > 1:
        mean_ms = float(np.mean(runs))
        std_ms = float(np.std(runs))
        x = range(1, len(runs) + 1)
        axes[1].plot(x, runs, color=COLORS["primary"], linewidth=1.5, marker="o",
                     markersize=4, markerfacecolor=COLORS["accent"])
        axes[1].axhline(y=mean_ms, color=COLORS["denoised"], linewidth=1.5,
                        label=f"Mean: {mean_ms:.2f} ms")
        axes[1].fill_between(x, mean_ms - std_ms, mean_ms + std_ms, alpha=0.2,
                             color=COLORS["denoised"], label=f"\u00b1{std_ms:.2f} ms")
        axes[1].set_xlabel("Run #")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("(b) DAE Inference Over Repeated Runs")
        axes[1].legend(framealpha=0.9)
    else:
        axes[1].text(0.5, 0.5, "No per-run measurements available",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")

    plt.tight_layout()
    path = _fig_path(out, label, "fig9_latency_benchmark.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig10_latent_space(X_denoised, y_labels, output_dir="results", label=""):
    """Fig 10: Latent space visualization (t-SNE + PCA)."""
    out = ensure_output_dir(output_dir)

    n_samples = X_denoised.shape[0]
    X_flat = X_denoised.reshape(n_samples, -1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 10: Latent Space Visualization of Denoised EEG",
                 fontweight="bold", y=1.02)

    max_samples = min(n_samples, 100)
    X_sub = X_flat[:max_samples]
    y_sub = y_labels[:max_samples]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sub)
    unique_labels = np.unique(y_sub)
    color_map = [COLORS["denoised"], COLORS["highlight"]]
    for i, lbl in enumerate(unique_labels):
        mask = y_sub == lbl
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color_map[i % len(color_map)],
                        label=f"Class {lbl}", alpha=0.7, s=50, edgecolors="white", linewidths=0.5)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title("(a) PCA Projection")
    axes[0].legend(framealpha=0.9)

    if max_samples >= 4:
        perplexity = min(30, max_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_sub)
        for i, lbl in enumerate(unique_labels):
            mask = y_sub == lbl
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                            c=color_map[i % len(color_map)],
                            label=f"Class {lbl}", alpha=0.7, s=50,
                            edgecolors="white", linewidths=0.5)
        axes[1].set_xlabel("t-SNE Dim 1")
        axes[1].set_ylabel("t-SNE Dim 2")
        axes[1].set_title("(b) t-SNE Projection")
        axes[1].legend(framealpha=0.9)

    plt.tight_layout()
    path = _fig_path(out, label, "fig10_latent_space.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig11_adaptation_curve(adaptation_data, output_dir="results", label=""):
    """Fig 11: Self-learning adaptation curve.

    Shows (a) model parameter shift, (b) classification confidence before
    and after each adaptation, and (c) reconstruction MSE before/after each
    adaptation to demonstrate that the feedback loop actually improves the
    model on distribution-shifted inputs.
    """
    out = ensure_output_dir(output_dir)

    n_epochs = len(adaptation_data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Figure 11: Online Self-Learning Adaptation", fontweight="bold", y=1.02)

    shifts = [d.get("shift", 0) for d in adaptation_data]
    confidences = [d.get("confidence", 0) for d in adaptation_data]
    confidences_after = [d.get("confidence_after", 0) for d in adaptation_data]
    mse_before = [d.get("mse_before", 0) for d in adaptation_data]
    mse_after = [d.get("mse_after", 0) for d in adaptation_data]
    adapted_flags = [d.get("adapted", False) for d in adaptation_data]
    adapted_x = [i + 1 for i, a in enumerate(adapted_flags) if a]

    axes[0].plot(range(1, n_epochs + 1), shifts, color=COLORS["primary"],
                 linewidth=1.5, marker="o", markersize=3)
    adapted_y = [shifts[i] for i, a in enumerate(adapted_flags) if a]
    axes[0].scatter(adapted_x, adapted_y, color=COLORS["highlight"], s=60,
                    zorder=5, edgecolors="white", linewidths=1, label="Adaptation Triggered")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Parameter Shift")
    axes[0].set_title("(a) Model Parameter Shift Over Time")
    axes[0].legend(framealpha=0.9)

    axes[1].plot(range(1, n_epochs + 1), confidences, color=COLORS["accent"],
                 linewidth=1.5, marker="s", markersize=3, label="Confidence (before)")
    for i in adapted_x:
        axes[1].plot([i - 0.2, i + 0.2],
                     [confidences[i - 1], confidences_after[i - 1]],
                     color=COLORS["highlight"], linewidth=2, zorder=5)
    axes[1].axhline(y=0.7, color=COLORS["highlight"], linestyle="--",
                    linewidth=1.5, label="Confidence Threshold (0.7)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Classification Confidence")
    axes[1].set_title("(b) Confidence Before/After Adaptation")
    axes[1].legend(framealpha=0.9, loc="lower right")
    axes[1].set_ylim([0, 1.05])

    axes[2].plot(range(1, n_epochs + 1), mse_before, color=COLORS["noisy"],
                 linewidth=1.5, marker="o", markersize=3, label="MSE before update")
    axes[2].plot(range(1, n_epochs + 1), mse_after, color=COLORS["denoised"],
                 linewidth=1.5, marker="o", markersize=3, label="MSE after update")
    for i in adapted_x:
        axes[2].plot([i, i], [mse_before[i - 1], mse_after[i - 1]],
                     color=COLORS["highlight"], linewidth=1.5, alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Reconstruction MSE (\u00b5V\u00b2)")
    axes[2].set_title("(c) Reconstruction Error Before/After Adaptation")
    axes[2].legend(framealpha=0.9)

    plt.tight_layout()
    path = _fig_path(out, label, "fig11_adaptation_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig12_architecture_diagram(output_dir="results", latency=None, metrics=None,
                               label=""):
    """Fig 12: Architecture summary figure.

    When ``latency`` (dict with mean_ms/breakdown) and ``metrics`` (dict from
    evaluate_pipeline) are provided, the measured numbers are annotated on
    the diagram so claims like "real-time" are backed by actual numbers
    instead of unsupported labels.
    """
    out = ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Figure 12: Adaptive AI EEG Denoising & Classification Architecture",
                 fontweight="bold", fontsize=13, pad=20)

    boxes = [
        (0.5, 6.5, 2.5, 1.2, "EEG Input\n(64 channels\n160-250 Hz)", COLORS["noisy"]),
        (3.8, 6.5, 2.5, 1.2, "Preprocessing\nBandpass +\nNotch Filter", COLORS["primary"]),
        (7.1, 6.5, 2.5, 1.2, "LMS Adaptive\nFilter\n(Padasip)", COLORS["adapted"]),
        (10.4, 6.5, 2.5, 1.2, "DAE Neural\nDenoiser\n(PyTorch)", COLORS["denoised"]),
        (13.7, 6.5, 1.8, 1.2, "Clean\nEEG", COLORS["clean"]),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color,
                              facecolor=color, alpha=0.15, zorder=2,
                              joinstyle="round")
        ax.add_patch(rect)
        border = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color,
                                facecolor="none", zorder=3, joinstyle="round")
        ax.add_patch(border)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        y_mid = boxes[i][1] + boxes[i][3] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))

    bottom_boxes = [
        (2.0, 3.5, 3.0, 1.5, "CSP Feature\nExtraction\n(MNE-Python)", COLORS["secondary"]),
        (6.0, 3.5, 3.0, 1.5, "SVM / LDA\nClassification\n(scikit-learn)", COLORS["accent"]),
        (10.0, 3.5, 3.5, 1.5, "Online Self-Learning\nAdaptation Engine\n(PyTorch)", COLORS["highlight"]),
    ]

    for x, y, w, h, text, color in bottom_boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color,
                              facecolor=color, alpha=0.15, zorder=2,
                              joinstyle="round")
        ax.add_patch(rect)
        border = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color,
                                facecolor="none", zorder=3, joinstyle="round")
        ax.add_patch(border)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=4)

    ax.annotate("", xy=(3.5, 3.5), xytext=(3.5, 5.0),
                 arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))
    ax.annotate("", xy=(7.5, 5.0), xytext=(7.5, 3.5),
                 arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))
    ax.annotate("", xy=(11.75, 5.0), xytext=(11.75, 3.5),
                 arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))

    # --- Feedback loop: classifier confidence -> adaptation -> model update ---
    # SVM/LDA output (right edge) -> Adaptation Engine (left edge)
    ax.annotate("", xy=(10.0, 4.2), xytext=(9.0, 4.2),
                 arrowprops=dict(arrowstyle="-|>", color=COLORS["highlight"],
                                 lw=2, connectionstyle="arc3,rad=0.0"))
    ax.text(9.5, 4.55, "prediction\nconfidence", ha="center", va="bottom",
            fontsize=8, color=COLORS["highlight"], fontweight="bold")

    # Adaptation Engine (top) -> DAE (bottom): model weight update
    ax.annotate("", xy=(11.75, 6.5), xytext=(11.75, 5.0),
                 arrowprops=dict(arrowstyle="-|>", color=COLORS["highlight"],
                                 lw=2, connectionstyle="arc3,rad=0.0"))
    ax.text(12.9, 5.75, "weight update", ha="center", va="center",
            fontsize=8, color=COLORS["highlight"], fontweight="bold")

    tech_text = (
        "Tech Stack:  Python  |  PyTorch  |  MNE-Python  |  MOABB  |  "
        "scikit-learn  |  Padasip  |  SciPy  |  Matplotlib"
    )
    ax.text(8, 1.5, tech_text, ha="center", va="center", fontsize=9,
            style="italic", color="gray",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    key_features = (
        "Key Features:  Real-time processing  |  Adaptive  |  "
        "Self-learning  |  No hardware required (offline mode)"
    )
    ax.text(8, 0.7, key_features, ha="center", va="center", fontsize=9,
            style="italic", color="gray",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan",
                      edgecolor="gray", alpha=0.8))

    # --- Measured results annotation (real numbers, not claims) ---
    notes = []
    if latency is not None and latency.get("mean_ms") is not None:
        bd = latency.get("breakdown", {})
        notes.append(
            f"Measured latency: {latency['mean_ms']:.2f} ms "
            f"(+/- {latency.get('std_ms', 0):.2f}) per 1-s epoch "
            f"[pre {bd.get('preprocessing', 0):.1f} | LMS {bd.get('lms', 0):.1f} | "
            f"DAE {bd.get('dae', 0):.1f} | CSP {bd.get('csp', 0):.1f} | "
            f"SVM {bd.get('svm', 0):.1f} ms]"
        )
    if metrics is not None:
        snr = metrics.get("snr_after")
        if snr is not None:
            notes.append(
                f"Measured denoising: SNR {metrics.get('snr_before', 0):.1f} -> "
                f"{snr:.1f} dB | Corr {metrics.get('corr_noisy', 0):.2f} -> "
                f"{metrics.get('corr_denoised', 0):.2f} | "
                f"MSE {metrics.get('mse_noisy', 0):.0f} -> "
                f"{metrics.get('mse_denoised', 0):.0f}"
            )
    for i, note in enumerate(notes):
        ax.text(8, 2.2 - i * 0.55, note, ha="center", va="center", fontsize=8.5,
                style="italic", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1",
                          edgecolor="#1abc9c", alpha=0.9))

    plt.tight_layout()
    path = _fig_path(out, label, "fig12_architecture.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def fig13_before_after_channel(X_noisy, X_denoised, sfreq, n_show=6,
                                 output_dir="results", label=""):
    """Fig 13: Before vs After denoising across multiple channels."""
    out = ensure_output_dir(output_dir)
    n_channels = min(X_noisy.shape[1], n_show)
    trial_idx = 0
    t = np.arange(X_noisy.shape[2]) / sfreq

    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 2.5 * n_channels), sharex=True)
    fig.suptitle("Figure 13: Per-Channel Denoising Results",
                 fontweight="bold", y=1.01)

    for ch in range(n_channels):
        ylo, yhi = robust_ylim(X_noisy[trial_idx, ch], X_denoised[trial_idx, ch])
        axes[ch, 0].plot(t, X_noisy[trial_idx, ch], color=COLORS["noisy"],
                         linewidth=0.4, alpha=0.8)
        axes[ch, 0].fill_between(t, X_noisy[trial_idx, ch], alpha=0.1,
                                  color=COLORS["noisy"])
        axes[ch, 0].set_ylabel(f"Ch {ch+1}", fontsize=8, rotation=0, labelpad=35)
        axes[ch, 0].set_ylim([ylo, yhi])
        if ch == 0:
            axes[ch, 0].set_title("Noisy Input", fontweight="bold")

        axes[ch, 1].plot(t, X_denoised[trial_idx, ch], color=COLORS["denoised"],
                         linewidth=0.4, alpha=0.8)
        axes[ch, 1].fill_between(t, X_denoised[trial_idx, ch], alpha=0.1,
                                  color=COLORS["denoised"])
        axes[ch, 1].set_ylim([ylo, yhi])
        if ch == 0:
            axes[ch, 1].set_title("DAE Denoised", fontweight="bold")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    path = _fig_path(out, label, "fig13_per_channel.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_all_figures(pipeline_data, output_dir="results", mode="run", label="",
                         run_dir=None):
    """Master function: generate all figures from pipeline run data.

    Args:
        pipeline_data: dict with all required data arrays and metrics.
        output_dir: folder to save figures.
        mode: execution mode label (e.g. "offline", "dataset", "streaming").
            Appended to the timestamped run folder name.
        label: optional name of the dataset (or other run tag). Appended to
            the run folder name and prefixed to every figure filename so that
            multi-dataset runs keep results separated and identifiable.
    """
    print("\n" + "=" * 60)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    # Save each run's figures into a timestamped sub-folder so previous runs
    # are never overwritten. Old run folders are preserved alongside. The
    # mode label is appended so offline vs dataset runs are identifiable, and
    # the dataset label (if given) names the folder + figure files.
    if run_dir is None:
        run_name = "run_{}_{}".format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), mode)
        if label:
            run_name = f"{run_name}_{label}"
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

    # Real EEG has no clean ground truth: in dataset mode the "reference" is
    # the preprocessed (bandpass+notch) signal, not a true clean signal.
    clean_label = ("Reference (Preprocessed)"
                   if mode == "dataset" else "Ground Truth (Clean)")
    output_dir = run_dir

    paths = []

    required_keys = ["X_noisy", "X_clean", "X_adapted", "X_denoised", "sfreq"]
    missing = [k for k in required_keys if k not in pipeline_data]
    if missing:
        print(f"  WARNING: Missing keys: {missing}. Skipping some figures.")

    if "X_noisy" in pipeline_data and "X_clean" in pipeline_data and "X_denoised" in pipeline_data:
        paths.append(fig1_raw_eeg_comparison(
            pipeline_data["X_noisy"], pipeline_data["X_clean"],
            pipeline_data["X_denoised"], pipeline_data["sfreq"],
            output_dir=output_dir, clean_label=clean_label, label=label
        ))

    if all(k in pipeline_data for k in ["X_noisy", "X_pre", "X_adapted", "X_denoised", "X_clean"]):
        paths.append(fig2_pipeline_stages(
            pipeline_data["X_noisy"], pipeline_data["X_pre"],
            pipeline_data["X_adapted"], pipeline_data["X_denoised"],
            pipeline_data["X_clean"], pipeline_data["sfreq"],
            output_dir=output_dir, clean_label=clean_label, label=label
        ))

    if "X_denoised" in pipeline_data:
        n_ch = min(pipeline_data["X_denoised"].shape[1], 20)
        trial_idx = 0

        # Shared color scale so noisy vs denoised heatmaps are comparable
        shared_vmin, shared_vmax = None, None
        if "X_noisy" in pipeline_data:
            noisy_slice = pipeline_data["X_noisy"][trial_idx, :n_ch, :]
            denoised_slice = pipeline_data["X_denoised"][trial_idx, :n_ch, :]
            combined = np.concatenate([noisy_slice.ravel(), denoised_slice.ravel()])
            std = np.std(combined)
            shared_vmin, shared_vmax = -2 * std, 2 * std

        paths.append(fig3_multi_channel_heatmap(
            pipeline_data["X_denoised"], "Denoised Signal",
            pipeline_data["sfreq"], output_dir=output_dir, n_channels=n_ch,
            vmin=shared_vmin, vmax=shared_vmax, label=label
        ))
        if "X_noisy" in pipeline_data:
            paths.append(fig3_multi_channel_heatmap(
                pipeline_data["X_noisy"], "Noisy Input",
                pipeline_data["sfreq"], output_dir=output_dir, n_channels=n_ch,
                vmin=shared_vmin, vmax=shared_vmax, label=label
            ))

    if all(k in pipeline_data for k in ["X_noisy", "X_denoised", "X_clean"]):
        paths.append(fig4_frequency_spectrum(
            pipeline_data["X_noisy"], pipeline_data["X_denoised"],
            pipeline_data["X_clean"], pipeline_data["sfreq"],
            output_dir=output_dir, clean_label=clean_label, label=label
        ))

    if "losses" in pipeline_data:
        paths.append(fig5_dae_training(
            pipeline_data["losses"],
            val_losses=pipeline_data.get("val_losses"),
            output_dir=output_dir, label=label
        ))

    if "metrics" in pipeline_data:
        m = pipeline_data["metrics"]
        if m.get("snr_after") is not None and m.get("snr_before") is not None:
            paths.append(fig6_snr_comparison(m, output_dir=output_dir, label=label))
        else:
            print("  Skipped fig6 (SNR needs clean ground truth; not available on real data)")

    if "y_true" in pipeline_data and "y_pred" in pipeline_data:
        paths.append(fig7_confusion_matrix(
            pipeline_data["y_true"], pipeline_data["y_pred"], output_dir=output_dir,
            label=label
        ))

    if "y_true" in pipeline_data and "y_scores" in pipeline_data:
        paths.append(fig8_roc_curve(
            pipeline_data["y_true"], pipeline_data["y_scores"], output_dir=output_dir,
            label=label
        ))

    if "latency" in pipeline_data:
        paths.append(fig9_latency_benchmark(pipeline_data["latency"], output_dir=output_dir, label=label))

    if "X_denoised" in pipeline_data and "y_true" in pipeline_data:
        paths.append(fig10_latent_space(
            pipeline_data["X_denoised"], pipeline_data["y_true"], output_dir=output_dir,
            label=label
        ))

    if "adaptation_data" in pipeline_data:
        paths.append(fig11_adaptation_curve(
            pipeline_data["adaptation_data"], output_dir=output_dir, label=label
        ))

    paths.append(fig12_architecture_diagram(
        latency=pipeline_data.get("latency"),
        metrics=pipeline_data.get("metrics"),
        output_dir=output_dir, label=label
    ))

    if "X_noisy" in pipeline_data and "X_denoised" in pipeline_data:
        paths.append(fig13_before_after_channel(
            pipeline_data["X_noisy"], pipeline_data["X_denoised"],
            pipeline_data["sfreq"], output_dir=output_dir, label=label
        ))

    print(f"\n  Total figures generated: {len(paths)}")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    return paths
