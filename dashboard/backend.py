"""
Real-Time EEG Dashboard Backend
Flask + Socket.IO server that runs the adaptive EEG pipeline
and streams results to a React frontend in real-time.
"""
import sys
import os
import re
import json
import time
import numpy as np
import base64
import io
import threading
import traceback
import subprocess
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(BASE_DIR, "static")
UDA_RESULTS_DIR = Path(REPO_ROOT) / "results_uda"

sys.path.insert(0, REPO_ROOT)

from flask import Flask, send_from_directory, jsonify, request, abort
from flask_socketio import SocketIO, emit

from config import PipelineConfig
from synthetic import generate_eeg_trials, inject_noise
from preprocessing import apply_bandpass, apply_notch
from adaptive_filter import AdaptiveFilter
from ai_denoising import DenoisingAutoencoder, train_dae, OnlineDenoiser
from classification import extract_csp_features, EEGClassifier
from self_learning import OnlineAdaptationEngine
from evaluation import evaluate_pipeline, measure_latency
from data_pipeline.dataset_fetcher import DATASETS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

app = Flask(__name__, static_folder=STATIC_DIR)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

ALL_DATASETS = ("all",) + tuple(DATASETS)

# Separate state for the UDA experiment runner so it cannot clash with the
# original live-pipeline "running" flag.
uda_state = {
    "running": False,
    "logs": [],
    "returncode": None,
    "dataset": None,
}
uda_lock = threading.Lock()

pipeline_state = {
    "running": False,
    "mode": None,
    "progress": 0,
    "phase": "",
    "metrics": {},
    "chart_data": {
        "eeg_noisy": [],
        "eeg_clean": [],
        "eeg_denoised": [],
        "loss_curve": [],
        "snr_history": [],
        "adaptation_events": [],
        "confusion_matrix": [],
        "band_power": {"delta": 0, "theta": 0, "alpha": 0, "beta": 0, "gamma": 0},
        "latency": {"mean_ms": 0, "std_ms": 0},
        "classification": {"train_acc": 0, "cv_acc": 0},
    },
}

lock = threading.Lock()


def _to_serializable(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if not np.isfinite(v):
            return 0.0
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def emit_update(event, data):
    socketio.emit(event, _to_serializable(data))


def normalize_per_trial(data):
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        std = np.std(data[i])
        if std > 1e-10:
            out[i] = data[i] / std
        else:
            out[i] = data[i]
    return out


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def run_pipeline_thread(config, mode):
    try:
        with lock:
            pipeline_state["running"] = True
            pipeline_state["mode"] = mode

        emit_update("pipeline_status", {"running": True, "mode": mode, "phase": "Starting..."})

        if mode == "offline":
            _run_offline(config)
        elif mode == "dataset":
            _run_dataset(config)
        elif mode == "synthetic_stream":
            _run_synthetic_stream(config)

    except Exception as e:
        traceback.print_exc()
        emit_update("pipeline_error", {"error": str(e)})
    finally:
        with lock:
            pipeline_state["running"] = False
        emit_update("pipeline_status", {"running": False, "phase": "Complete"})


def _run_offline(config):
    # Phase 1: Generate Data
    emit_update("pipeline_status", {"phase": "Phase 1: Generating synthetic EEG data...", "progress": 5})
    labels = np.random.RandomState(42).randint(0, 2, size=200)
    clean_data = generate_eeg_trials(
        n_trials=200, n_channels=config.n_channels,
        n_samples=config.epoch_samples, sfreq=config.sfreq, labels=labels
    )
    noisy_data = inject_noise(clean_data, noise_type="all",
                              snr_db=config.noise_snr_db, sfreq=config.sfreq)

    n_train = int(0.8 * len(clean_data))
    train_clean, test_clean = clean_data[:n_train], clean_data[n_train:]
    train_noisy, test_noisy = noisy_data[:n_train], noisy_data[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    sample_data = {
        "noisy": test_noisy[0].tolist(),
        "clean": test_clean[0].tolist(),
        "channels": test_noisy.shape[1],
        "samples": test_noisy.shape[2],
        "sfreq": config.sfreq,
    }
    emit_update("eeg_initial", sample_data)
    emit_update("pipeline_status", {"phase": "Phase 1: Data generated", "progress": 15})

    # Phase 2: Preprocessing
    emit_update("pipeline_status", {"phase": "Phase 2: Baseline preprocessing...", "progress": 20})

    def preprocess(data):
        out = apply_bandpass(data, config.l_freq, config.h_freq, config.sfreq)
        out = apply_notch(out, config.notch_freq, config.sfreq)
        return out

    train_pre = preprocess(train_noisy)
    test_pre = preprocess(test_noisy)
    emit_update("pipeline_status", {"phase": "Phase 2: Preprocessing complete", "progress": 30})

    # Phase 3: Adaptive Filtering
    emit_update("pipeline_status", {"phase": "Phase 3: LMS adaptive filtering...", "progress": 35})
    adapt_filter = AdaptiveFilter(
        n_channels=config.n_channels, method=config.filter_type,
        filter_length=config.lms_length, mu=config.lms_mu
    )

    train_norm = normalize_per_trial(train_pre)
    test_norm = normalize_per_trial(test_pre)

    train_adapted = np.zeros_like(train_norm)
    for i in range(train_norm.shape[0]):
        train_adapted[i] = adapt_filter.filter_signal(train_norm[i])
        if i % 20 == 0:
            emit_update("filter_progress", {"trial": i, "total": train_norm.shape[0], "stage": "train"})

    adapt_filter.reset()
    test_adapted = np.zeros_like(test_norm)
    for i in range(test_norm.shape[0]):
        test_adapted[i] = adapt_filter.filter_signal(test_norm[i])

    train_adapted = np.nan_to_num(train_adapted, nan=0.0)
    test_adapted = np.nan_to_num(test_adapted, nan=0.0)
    emit_update("pipeline_status", {"phase": "Phase 3: Adaptive filtering complete", "progress": 45})

    # Phase 4: AI Denoising
    emit_update("pipeline_status", {"phase": "Phase 4: Training DAE neural network...", "progress": 50})

    losses = []
    dae_model, full_losses = train_dae_stream(config, train_clean, config.noise_snr_db, emit_update, losses)
    denoiser = OnlineDenoiser(dae_model)

    train_denoised = np.nan_to_num(denoiser.denoise_batch(train_adapted), nan=0.0)
    test_denoised = np.nan_to_num(denoiser.denoise_batch(test_adapted), nan=0.0)
    emit_update("pipeline_status", {"phase": "Phase 4: DAE training complete", "progress": 65})

    with lock:
        pipeline_state["chart_data"]["loss_curve"] = full_losses

    # Phase 5: Classification
    emit_update("pipeline_status", {"phase": "Phase 5: CSP + SVM classification...", "progress": 70})
    n_comp = min(config.csp_n_components, train_denoised.shape[1] // 2)
    train_features, csp = extract_csp_features(train_denoised, train_labels, n_comp)
    test_features = csp.transform(test_denoised)

    train_features_clean, csp_clean = extract_csp_features(train_clean, train_labels, n_comp)
    test_features_clean = csp_clean.transform(test_clean)

    classifier = EEGClassifier(method="svm", kernel=config.svm_kernel)
    results_denoised = classifier.train(train_features, train_labels)

    classifier_clean = EEGClassifier(method="svm", kernel=config.svm_kernel)
    results_clean = classifier_clean.train(train_features_clean, train_labels)

    y_pred = classifier.predict(test_features)
    cm = _compute_confusion_matrix(test_labels, y_pred)
    emit_update("confusion_matrix", {"matrix": cm, "labels": ["Class 0", "Class 1"]})

    y_scores = classifier.predict_proba(test_features)
    roc_data = _compute_roc(test_labels, y_scores)
    emit_update("roc_curve", roc_data)

    with lock:
        pipeline_state["chart_data"]["classification"] = {
            "train_acc": results_denoised["train_accuracy"],
            "cv_acc": results_denoised["cv_mean"],
        }
        pipeline_state["chart_data"]["confusion_matrix"] = cm

    emit_update("pipeline_status", {"phase": "Phase 5: Classification complete", "progress": 80})

    # Band power analysis
    bp = _compute_band_power(test_noisy, test_denoised, test_clean, config.sfreq)
    emit_update("band_power", bp)

    with lock:
        pipeline_state["chart_data"]["band_power"] = bp

    # Phase 6: Self-learning
    emit_update("pipeline_status", {"phase": "Phase 6: Online adaptation...", "progress": 85})
    engine = OnlineAdaptationEngine(
        denoiser=denoiser, classifier=classifier,
        confidence_threshold=0.7, adaptation_interval=5,
        online_lr=config.online_lr, mini_batch_size=config.mini_batch_size
    )

    adapt_events = []
    for i in range(min(50, test_adapted.shape[0])):
        result = engine.process_epoch(
            test_denoised[i], features=test_features[i:i+1],
            clean_target=test_clean[i]
        )
        adapt_events.append({
            "epoch": i + 1,
            "adapted": result["adapted"],
            "confidence": 1.0 if not result.get("low_confidence", False) else 0.5,
            "shift": result.get("shift_detected", False),
        })
        if i % 10 == 0:
            emit_update("adaptation_progress", {"epoch": i, "total": min(50, test_adapted.shape[0])})

    with lock:
        pipeline_state["chart_data"]["adaptation_events"] = adapt_events

    # Evaluation
    emit_update("pipeline_status", {"phase": "Phase 6: Evaluation...", "progress": 90})
    metrics = evaluate_pipeline(
        test_adapted, test_denoised, test_clean,
        classifier=classifier, features=test_features, labels=test_labels
    )
    lat = measure_latency(denoiser.denoise_epoch, test_denoised[0], n_runs=20)

    # SNR history
    snr_before = metrics.get("avg_snr_before", 0)
    snr_after = metrics.get("avg_snr_after", 0)
    acc = metrics.get("classification_accuracy", 0)

    with lock:
        pipeline_state["metrics"] = {
            "snr_before": float(snr_before) if np.isfinite(snr_before) else 0,
            "snr_after": float(snr_after) if np.isfinite(snr_after) else 0,
            "mse_noisy": float(metrics.get("mse_noisy", 0)),
            "mse_denoised": float(metrics.get("mse_denoised", 0)),
            "accuracy": float(acc),
        }
        pipeline_state["chart_data"]["latency"] = _to_serializable(lat)

    # Send denoised EEG for live chart
    denoised_sample = {
        "denoised": test_denoised[0].tolist(),
        "adapted": test_adapted[0].tolist(),
    }
    emit_update("eeg_denoised", denoised_sample)

    emit_update("pipeline_metrics", pipeline_state["metrics"])
    emit_update("latency", lat)
    emit_update("pipeline_status", {"phase": "Pipeline Complete!", "progress": 100})
    emit_update("pipeline_complete", {"mode": "offline"})


def _run_dataset(config):
    dataset_name = getattr(config, "dataset_name", None) or "physionet"
    subjects = getattr(config, "dataset_subjects", None) or [1, 2, 3]
    if dataset_name == "all":
        dataset_name = "physionet"
        emit_update("uda_log", {
            "line": "Live-pipeline mode supports one dataset at a time; "
                    "using PhysioNet. Use the UDA tab to run every dataset."
        })

    emit_update("pipeline_status", {
        "phase": f"Loading {dataset_name} dataset (subjects {subjects})...",
        "progress": 5,
    })

    from data_pipeline import fetch_eeg_dataset
    X, y, meta = fetch_eeg_dataset(dataset_name=dataset_name, subjects=subjects)

    if X.shape[0] == 0:
        emit_update("pipeline_error", {"error": "No trials loaded from dataset"})
        return

    config.n_channels = X.shape[1]
    config.sfreq = meta["sfreq"]
    config.notch_freq = 60.0 if meta["sfreq"] > 125 else meta["sfreq"] / 2.0 - 1.0

    emit_update("dataset_loaded", {
        "dataset": dataset_name,
        "n_trials": X.shape[0], "n_channels": X.shape[1],
        "n_samples": X.shape[2], "sfreq": meta["sfreq"],
        "labels": np.unique(y).tolist(),
    })
    emit_update("pipeline_status", {"phase": f"Dataset loaded: {X.shape[0]} trials", "progress": 15})

    unique_labels = np.unique(y)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_int = np.array([label_map[lbl] for lbl in y])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    # Send raw data sample
    sample_data = {
        "noisy": X_test[0].tolist(),
        "clean": X_test[0].tolist(),
        "channels": X_test.shape[1],
        "samples": X_test.shape[2],
        "sfreq": config.sfreq,
    }
    emit_update("eeg_initial", sample_data)

    # Preprocess
    emit_update("pipeline_status", {"phase": "Preprocessing...", "progress": 20})

    def preprocess(data):
        out = apply_bandpass(data, config.l_freq, config.h_freq, config.sfreq)
        out = apply_notch(out, config.notch_freq, config.sfreq)
        return out

    X_train_pre = np.nan_to_num(preprocess(X_train), nan=0.0)
    X_test_pre = np.nan_to_num(preprocess(X_test), nan=0.0)

    # Adaptive Filter
    emit_update("pipeline_status", {"phase": "LMS Adaptive Filter...", "progress": 30})
    adapt_filter = AdaptiveFilter(
        n_channels=config.n_channels, method=config.filter_type,
        filter_length=config.lms_length, mu=config.lms_mu
    )

    X_train_norm = normalize_per_trial(X_train_pre)
    X_test_norm = normalize_per_trial(X_test_pre)

    X_train_adapted = np.zeros_like(X_train_norm)
    for i in range(X_train_norm.shape[0]):
        X_train_adapted[i] = adapt_filter.filter_signal(X_train_norm[i])
    adapt_filter.reset()
    X_test_adapted = np.zeros_like(X_test_norm)
    for i in range(X_test_norm.shape[0]):
        X_test_adapted[i] = adapt_filter.filter_signal(X_test_norm[i])

    X_train_adapted = np.nan_to_num(X_train_adapted, nan=0.0)
    X_test_adapted = np.nan_to_num(X_test_adapted, nan=0.0)

    # DAE Training
    emit_update("pipeline_status", {"phase": "Training DAE...", "progress": 45})
    losses = []
    dae_model, full_losses = train_dae_stream(config, X_train_adapted, config.noise_snr_db, emit_update, losses)
    denoiser = OnlineDenoiser(dae_model)

    X_train_denoised = np.nan_to_num(denoiser.denoise_batch(X_train_adapted), nan=0.0)
    X_test_denoised = np.nan_to_num(denoiser.denoise_batch(X_test_adapted), nan=0.0)

    with lock:
        pipeline_state["chart_data"]["loss_curve"] = full_losses

    # Classification
    emit_update("pipeline_status", {"phase": "CSP + SVM Classification...", "progress": 65})
    n_classes = len(np.unique(y_train))
    n_comp = min(config.csp_n_components, X_train_denoised.shape[1] // 2)
    if n_comp < 1:
        n_comp = 1

    train_features, csp = extract_csp_features(X_train_denoised, y_train, n_comp)
    test_features = csp.transform(X_test_denoised)

    classifier = EEGClassifier(method="svm", kernel=config.svm_kernel)
    results = classifier.train(train_features, y_train)

    y_pred = classifier.predict(test_features)
    cm = _compute_confusion_matrix(y_test, y_pred)
    emit_update("confusion_matrix", {"matrix": cm, "labels": ["Class 0", "Class 1"]})

    y_scores = classifier.predict_proba(test_features)
    roc_data = _compute_roc(y_test, y_scores)
    emit_update("roc_curve", roc_data)

    bp = _compute_band_power(X_test_pre, X_test_denoised, X_test_adapted, config.sfreq)
    emit_update("band_power", bp)

    with lock:
        pipeline_state["chart_data"]["band_power"] = bp
        pipeline_state["chart_data"]["classification"] = {
            "train_acc": results["train_accuracy"],
            "cv_acc": results["cv_mean"],
        }

    # Self-learning
    emit_update("pipeline_status", {"phase": "Online Adaptation...", "progress": 75})
    engine = OnlineAdaptationEngine(
        denoiser=denoiser, classifier=classifier,
        confidence_threshold=0.7, adaptation_interval=5,
        online_lr=config.online_lr, mini_batch_size=config.mini_batch_size
    )

    adapt_events = []
    for i in range(min(50, X_test_denoised.shape[0])):
        result = engine.process_epoch(
            X_test_denoised[i], features=test_features[i:i+1],
            clean_target=X_test_adapted[i]
        )
        adapt_events.append({
            "epoch": i + 1,
            "adapted": result["adapted"],
            "confidence": 1.0 if not result.get("low_confidence", False) else 0.5,
        })

    with lock:
        pipeline_state["chart_data"]["adaptation_events"] = adapt_events

    # Evaluation
    emit_update("pipeline_status", {"phase": "Evaluation...", "progress": 90})
    metrics = evaluate_pipeline(
        X_test_adapted, X_test_denoised, X_test_adapted,
        classifier=classifier, features=test_features, labels=y_test
    )
    lat = measure_latency(denoiser.denoise_epoch, X_test_denoised[0], n_runs=20)

    denoised_sample = {"denoised": X_test_denoised[0].tolist(), "adapted": X_test_adapted[0].tolist()}
    emit_update("eeg_denoised", denoised_sample)

    with lock:
        snr_b = metrics.get("avg_snr_before", 0)
        snr_a = metrics.get("avg_snr_after", 0)
        acc = metrics.get("classification_accuracy", 0)
        pipeline_state["metrics"] = {
            "snr_before": float(snr_b) if np.isfinite(snr_b) else 0,
            "snr_after": float(snr_a) if np.isfinite(snr_a) else 0,
            "mse_noisy": float(metrics.get("mse_noisy", 0)),
            "mse_denoised": float(metrics.get("mse_denoised", 0)),
            "accuracy": float(acc),
        }
        pipeline_state["chart_data"]["latency"] = _to_serializable(lat)

    emit_update("pipeline_metrics", pipeline_state["metrics"])
    emit_update("latency", lat)
    emit_update("pipeline_status", {"phase": "Pipeline Complete!", "progress": 100})
    emit_update("pipeline_complete", {"mode": "dataset"})


def _run_synthetic_stream(config):
    """Stream synthetic EEG data in real-time (simulates live BCI)."""
    emit_update("pipeline_status", {"phase": "Starting real-time stream...", "progress": 5})

    data = generate_eeg_trials(
        n_trials=50, n_channels=config.n_channels,
        n_samples=config.epoch_samples, sfreq=config.sfreq
    )
    noisy = inject_noise(data, noise_type="all", snr_db=config.noise_snr_db, sfreq=config.sfreq)
    labels = np.random.randint(0, 2, size=data.shape[0])

    emit_update("pipeline_status", {"phase": "Streaming EEG...", "progress": 15})

    adapt_filter = AdaptiveFilter(
        n_channels=config.n_channels, method=config.filter_type,
        filter_length=config.lms_length, mu=config.lms_mu
    )

    def preprocess(d):
        out = apply_bandpass(d, config.l_freq, config.h_freq, config.sfreq)
        out = apply_notch(out, config.notch_freq, config.sfreq)
        return out

    n_train = 40
    train_data = preprocess(noisy[:n_train])
    train_norm = normalize_per_trial(train_data)
    train_adapted = np.zeros_like(train_norm)
    for i in range(train_norm.shape[0]):
        train_adapted[i] = adapt_filter.filter_signal(train_norm[i])
    train_adapted = np.nan_to_num(train_adapted, nan=0.0)

    dae_model, full_losses = train_dae_stream(config, data[:n_train], config.noise_snr_db, emit_update, [])
    denoiser = OnlineDenoiser(dae_model)
    train_denoised = np.nan_to_num(denoiser.denoise_batch(train_adapted), nan=0.0)

    n_comp = min(config.csp_n_components, train_denoised.shape[1] // 2)
    if n_comp < 1:
        n_comp = 1
    train_features, csp = extract_csp_features(train_denoised, labels[:n_train], n_comp)

    classifier = EEGClassifier(method="svm", kernel=config.svm_kernel)
    classifier.train(train_features, labels[:n_train])

    adapt_filter.reset()

    with lock:
        pipeline_state["chart_data"]["loss_curve"] = full_losses

    emit_update("pipeline_status", {"phase": "Streaming live data...", "progress": 60})

    for trial_idx in range(n_train, min(n_train + 10, data.shape[0])):
        test_raw = noisy[trial_idx:trial_idx + 1]
        test_pre = preprocess(test_raw)
        test_norm = normalize_per_trial(test_pre)
        test_adapted = adapt_filter.filter_signal(test_norm[0])
        test_adapted = np.nan_to_num(test_adapted, nan=0.0)
        test_denoised = denoiser.denoise_epoch(test_adapted)
        test_denoised = np.nan_to_num(test_denoised, nan=0.0)

        emit_update("stream_trial", {
            "trial": trial_idx - n_train + 1,
            "noisy": test_raw[0].tolist(),
            "adapted": test_adapted.tolist(),
            "denoised": test_denoised.tolist(),
            "clean": data[trial_idx].tolist(),
            "true_label": int(labels[trial_idx]),
        })

        test_feat = csp.transform(test_denoised.reshape(1, *test_denoised.shape))
        pred = classifier.predict(test_feat)
        proba = classifier.predict_proba(test_feat)

        emit_update("stream_prediction", {
            "predicted": int(pred[0]),
            "true_label": int(labels[trial_idx]),
            "confidence": float(np.max(proba)),
            "probabilities": proba[0].tolist(),
        })

        progress = 60 + int(40 * (trial_idx - n_train + 1) / 10)
        emit_update("pipeline_status", {"phase": f"Streaming trial {trial_idx - n_train + 1}/10...", "progress": min(progress, 99)})

        socketio.sleep(0.3)

    emit_update("pipeline_status", {"phase": "Stream complete!", "progress": 100})
    emit_update("pipeline_complete", {"mode": "synthetic_stream"})


def train_dae_stream(config, clean_data, noise_level_db, emit_fn, loss_collector):
    import torch
    from torch.utils.data import DataLoader
    from ai_denoising.trainer import DenoisingDataset, create_training_data
    from ai_denoising.model import DenoisingAutoencoder
    from utils import get_device

    device = get_device()
    noisy_data, target_data = create_training_data(clean_data, noise_level_db, config.sfreq)
    dataset = DenoisingDataset(target_data, noisy_data)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    n_channels = clean_data.shape[1]
    model = DenoisingAutoencoder(
        n_channels=n_channels, n_samples=clean_data.shape[2], latent_dim=config.latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.MSELoss()

    model.train()
    losses = []

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_noisy, batch_clean in dataloader:
            batch_noisy = batch_noisy.to(device)
            batch_clean = batch_clean.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_noisy)
            loss = criterion(reconstructed, batch_clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_noisy.shape[0]

        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            emit_fn("training_progress", {
                "epoch": epoch + 1,
                "total_epochs": config.epochs,
                "loss": avg_loss,
                "losses": losses.copy(),
            })

    model.eval()
    if loss_collector is not None:
        loss_collector.extend(losses)
    return model, losses


def _compute_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    return cm.tolist()


def _compute_roc(y_true, y_scores):
    from sklearn.metrics import roc_curve, auc
    if y_scores.ndim > 1 and y_scores.shape[1] >= 2:
        y_score = y_scores[:, 1]
    else:
        y_score = y_scores.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}


def _compute_band_power(X_noisy, X_denoised, X_clean, sfreq):
    f, p_noisy = welch(X_noisy[0, 0], fs=sfreq, nperseg=min(256, X_noisy.shape[2]))
    _, p_denoised = welch(X_denoised[0, 0], fs=sfreq, nperseg=min(256, X_denoised.shape[2]))
    _, p_clean = welch(X_clean[0, 0], fs=sfreq, nperseg=min(256, X_clean.shape[2]))

    freq_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
                  "beta": (13, 30), "gamma": (30, 50)}

    result = {}
    for name, (lo, hi) in freq_bands.items():
        idx = np.logical_and(f >= lo, f <= hi)
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        result[name] = {
            "noisy": float(_trapz(p_noisy[idx], f[idx])),
            "denoised": float(_trapz(p_denoised[idx], f[idx])),
            "clean": float(_trapz(p_clean[idx], f[idx])),
        }

    return result


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


@app.route("/api/datasets")
def api_datasets():
    return jsonify({"datasets": list(ALL_DATASETS)})


@app.route("/api/status")
def api_status():
    import json as _json
    data = _to_serializable(dict(pipeline_state))
    return _json.dumps(data, default=str), 200, {"Content-Type": "application/json"}


@socketio.on("connect")
def handle_connect():
    print("[WS] Client connected")
    emit("connected", {"status": "ok"})


@socketio.on("disconnect")
def handle_disconnect():
    print("[WS] Client disconnected")


@socketio.on("start_pipeline")
def handle_start_pipeline(data):
    if pipeline_state["running"]:
        emit("pipeline_error", {"error": "Pipeline already running"})
        return

    mode = data.get("mode", "offline")
    channels = data.get("channels", 64)
    sfreq = data.get("sfreq", 250)
    snr = data.get("snr", -5.0)
    filter_type = data.get("filter", "lms")
    dataset_name = data.get("dataset", "physionet")
    raw_subjects = data.get("subjects", "1 2 3")
    if isinstance(raw_subjects, str):
        subject_list = [int(s) for s in raw_subjects.replace(",", " ").split() if s]
    else:
        subject_list = [int(s) for s in raw_subjects]
    if not subject_list:
        subject_list = [1, 2, 3]

    config = PipelineConfig(
        n_channels=channels,
        sfreq=sfreq,
        noise_snr_db=snr,
        filter_type=filter_type,
        dataset_name=dataset_name,
        dataset_subjects=subject_list,
    )

    t = threading.Thread(target=run_pipeline_thread, args=(config, mode), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# UDA results browser + experiment runner
# ---------------------------------------------------------------------------
def _uda_run_dir(name):
    """Resolve a run folder safely (no path traversal)."""
    if not name or Path(name).name != name or not re.fullmatch(r"run_[A-Za-z0-9_\-]+", name):
        return None
    path = UDA_RESULTS_DIR / name
    return path if path.is_dir() else None


def _read_csv_records(path):
    try:
        import pandas as pd
        return _to_serializable(pd.read_csv(path).to_dict(orient="records"))
    except Exception:
        return []


def _config_of(run_dir):
    try:
        with open(run_dir / "config.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def list_uda_runs():
    """Summarise every results_uda/run_* folder for the dashboard."""
    if not UDA_RESULTS_DIR.exists():
        return []
    runs = []
    for path in sorted(UDA_RESULTS_DIR.iterdir(), reverse=True):
        if not path.is_dir() or not path.name.startswith("run_"):
            continue
        cfg = _config_of(path)
        summary = _read_csv_records(path / "summary.csv")
        best_accuracy = None
        if summary:
            values = [r.get("accuracy_mean") for r in summary
                      if isinstance(r.get("accuracy_mean"), (int, float))]
            best_accuracy = max(values) if values else None
        runs.append({
            "name": path.name,
            "timestamp": path.name.replace("run_", "", 1),
            "dataset": cfg.get("dataset_name") or path.name.split("_")[-1],
            "mode": cfg.get("mode", ""),
            "methods": cfg.get("methods", []),
            "proportions": cfg.get("target_proportions", []),
            "n_subjects": cfg.get("synthetic_subjects") if cfg.get("mode") == "synthetic"
                          else len(cfg.get("subjects", [])),
            "n_summary_rows": len(summary),
            "best_accuracy": best_accuracy,
            "figures": sorted(p.name for p in path.glob("fig*.png")),
            "excel": sorted(p.name for p in path.glob("*.xlsx")),
        })
    return runs


def load_uda_run(name):
    run_dir = _uda_run_dir(name)
    if run_dir is None:
        return None
    tables = {
        "summary": _read_csv_records(run_dir / "summary.csv"),
        "subject_level": _read_csv_records(run_dir / "subject_level.csv"),
        "alignment": _read_csv_records(run_dir / "alignment.csv"),
        "statistics": _read_csv_records(run_dir / "statistics.csv"),
    }
    return {
        "name": run_dir.name,
        "config": _config_of(run_dir),
        "tables": tables,
        "figures": sorted(p.name for p in run_dir.glob("fig*.png")),
        "excel": sorted(p.name for p in run_dir.glob("*.xlsx")),
        "combined_excel": sorted(p.name for p in UDA_RESULTS_DIR.glob(
            "results_uda_summary_*.xlsx"
        )) if UDA_RESULTS_DIR.exists() else [],
    }


def _append_uda_log(line):
    with uda_lock:
        uda_state["logs"].append({"line": line.rstrip()})
        uda_state["logs"] = uda_state["logs"][-800:]
    emit_update("uda_log", {"line": line.rstrip()})


def _stream_uda_process(proc, dataset):
    try:
        for line in iter(proc.stdout.readline, ""):
            if line:
                _append_uda_log(line)
        returncode = proc.wait()
    except Exception as exc:
        _append_uda_log(f"[dashboard] Failed to read process output: {exc}")
        returncode = -1
    finally:
        with uda_lock:
            uda_state["running"] = False
            uda_state["returncode"] = returncode
            uda_state["process"] = None
        emit_update("uda_status", {
            "running": False, "returncode": returncode, "dataset": dataset
        })
        emit_update("uda_complete", {
            "dataset": dataset,
            "returncode": returncode,
            "success": returncode == 0,
        })


@app.route("/api/uda/runs")
def api_uda_runs():
    return jsonify(_to_serializable({"runs": list_uda_runs()}))


@app.route("/api/uda/run/<name>")
def api_uda_run(name):
    payload = load_uda_run(name)
    if payload is None:
        abort(404)
    return jsonify(payload)


@app.route("/api/uda/figure/<name>/<filename>")
def api_uda_figure(name, filename):
    run_dir = _uda_run_dir(name)
    if run_dir is None or not re.fullmatch(r"fig[0-9A-Za-z_\-]+\.png", filename):
        abort(404)
    return send_from_directory(str(run_dir), filename)


@app.route("/api/uda/status")
def api_uda_status():
    with uda_lock:
        data = dict(uda_state)
        data["process"] = None
    return jsonify(_to_serializable(data))


@app.route("/api/uda/run", methods=["POST"])
def api_uda_start():
    with uda_lock:
        if uda_state["running"]:
            return jsonify({"error": "A UDA run is already in progress"}), 409

    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "dataset")
    if mode not in ("synthetic", "dataset"):
        return jsonify({"error": f"Unknown mode: {mode}"}), 400

    dataset = str(payload.get("dataset", "bci2a"))
    if mode == "dataset" and dataset not in ALL_DATASETS:
        return jsonify({"error": f"Unknown dataset: {dataset}"}), 400

    def _numbers(key, cast):
        raw = payload.get(key, "")
        if isinstance(raw, list):
            return [cast(v) for v in raw]
        return [cast(v) for v in str(raw).replace(",", " ").split() if str(v).strip()]

    try:
        subjects = _numbers("subjects", int) or [1, 2, 3]
        proportions = _numbers("proportions", float) or [1.0]
        epochs = int(payload.get("epochs", 60))
        lambda_coral = float(payload.get("lambda_coral", 1000.0))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid parameter: {exc}"}), 400

    if not proportions or any(p <= 0 or p > 1 for p in proportions):
        return jsonify({"error": "Proportions must be numbers in (0, 1]"}), 400

    cmd = [
        sys.executable, "-m", "uda_cross_subject.run_uda",
        "--mode", mode,
        "--output-dir", str(UDA_RESULTS_DIR),
        "--epochs", str(epochs),
        "--lambda-coral", str(lambda_coral),
        "--proportions", *(str(p) for p in proportions),
    ]
    if mode == "dataset":
        cmd += ["--dataset", dataset, "--subjects", *(str(s) for s in subjects)]

    with uda_lock:
        uda_state["running"] = True
        uda_state["logs"] = []
        uda_state["returncode"] = None
        uda_state["dataset"] = dataset
    _append_uda_log("[dashboard] " + " ".join(cmd))

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        with uda_lock:
            uda_state["running"] = False
        return jsonify({"error": f"Could not start UDA process: {exc}"}), 500

    with uda_lock:
        uda_state["process"] = proc
    threading.Thread(
        target=_stream_uda_process, args=(proc, dataset), daemon=True
    ).start()
    return jsonify({"started": True, "dataset": dataset, "command": cmd})


@app.route("/api/uda/stop", methods=["POST"])
def api_uda_stop():
    with uda_lock:
        proc = uda_state.get("process")
    if proc is None or proc.poll() is not None:
        return jsonify({"stopped": False})
    proc.terminate()
    _append_uda_log("[dashboard] UDA run terminated by user")
    return jsonify({"stopped": True})


if __name__ == "__main__":
    print("=" * 50)
    print("  EEG Dashboard Server")
    print("  Open http://localhost:5000 in your browser")
    print("  Tabs: Live Pipeline | UDA Results")
    print("=" * 50)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
