"""
Main Pipeline Runner: Adaptive EEG Denoising & Classification Framework

Usage:
    python run_pipeline.py --mode offline        # Run offline with synthetic data
    python run_pipeline.py --mode dataset        # Run with MOABB dataset
    python run_pipeline.py --mode streaming      # Run with LSL mock streaming
"""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from synthetic import generate_eeg_trials, inject_noise
from preprocessing import apply_bandpass, apply_notch
from adaptive_filter import AdaptiveFilter
from ai_denoising import DenoisingAutoencoder, train_dae, OnlineDenoiser
from classification import extract_csp_features, EEGClassifier
from self_learning import OnlineAdaptationEngine
from evaluation import evaluate_pipeline, print_evaluation, measure_latency
from evaluation.visualization import generate_all_figures


def normalize_per_trial(data):
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        std = np.std(data[i])
        if std > 1e-10:
            out[i] = data[i] / std
        else:
            out[i] = data[i]
    return out


def run_offline_pipeline(config):
    """Run the full pipeline on synthetic data (no hardware needed)."""
    print("\n" + "=" * 60)
    print("  PHASE 1: Data Generation")
    print("=" * 60)

    clean_data = generate_eeg_trials(
        n_trials=200, n_channels=config.n_channels,
        n_samples=config.epoch_samples, sfreq=config.sfreq
    )
    print(f"  Generated {clean_data.shape[0]} trials | "
          f"{clean_data.shape[1]} channels | {clean_data.shape[2]} samples")

    labels = np.random.randint(0, 2, size=clean_data.shape[0])

    noisy_data = inject_noise(clean_data, noise_type="all",
                              snr_db=config.noise_snr_db, sfreq=config.sfreq)
    print(f"  Injected noise at SNR = {config.noise_snr_db} dB")

    n_train = int(0.8 * len(clean_data))
    train_clean, test_clean = clean_data[:n_train], clean_data[n_train:]
    train_noisy, test_noisy = noisy_data[:n_train], noisy_data[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    print("\n" + "=" * 60)
    print("  PHASE 2: Baseline Preprocessing")
    print("=" * 60)

    def preprocess(data):
        out = apply_bandpass(data, config.l_freq, config.h_freq, config.sfreq)
        out = apply_notch(out, config.notch_freq, config.sfreq)
        return out

    train_pre = preprocess(train_noisy)
    test_pre = preprocess(test_noisy)
    print("  Bandpass: {}-{} Hz | Notch: {} Hz".format(
        config.l_freq, config.h_freq, config.notch_freq))

    print("\n" + "=" * 60)
    print("  PHASE 3: Adaptive Filtering")
    print("=" * 60)

    adapt_filter = AdaptiveFilter(
        n_channels=config.n_channels, method=config.filter_type,
        filter_length=config.lms_length, mu=config.lms_mu
    )

    from ai_denoising import normalize_trials, denormalize_trials

    train_norm, train_means, train_stds = normalize_trials(train_pre)
    test_norm, test_means, test_stds = normalize_trials(test_pre)

    train_adapted = np.zeros_like(train_norm)
    for i in range(train_norm.shape[0]):
        train_adapted[i] = adapt_filter.filter_signal(train_norm[i])

    adapt_filter.reset()
    test_adapted = np.zeros_like(test_norm)
    for i in range(test_norm.shape[0]):
        test_adapted[i] = adapt_filter.filter_signal(test_norm[i])

    train_adapted = np.nan_to_num(train_adapted, nan=0.0)
    test_adapted = np.nan_to_num(test_adapted, nan=0.0)

    print(f"  {config.filter_type.upper()} filter | Length: {config.lms_length} | Mu: {config.lms_mu}")

    print("\n" + "=" * 60)
    print("  PHASE 4: AI-Based Denoising (Training)")
    print("=" * 60)

    # --- PHASE 4: AI-Based Denoising (Training) ---
    # Train DAE: input=normalized_noisy (train_adapted), target=train_clean
    # The trainer will automatically normalize train_clean to match the input distribution
    dae_model, losses = train_dae(clean_data=train_clean, config=config, noisy_data=train_adapted)

    denoiser = OnlineDenoiser(dae_model)

    train_denoised = denoiser.denoise_batch(train_adapted)
    test_denoised  = denoiser.denoise_batch(test_adapted)
    train_denoised = np.nan_to_num(train_denoised, nan=0.0)
    test_denoised  = np.nan_to_num(test_denoised,  nan=0.0)
    print(f"  Model trained | Final loss: {losses[-1]:.6f}")
    print(f"  Denoised scale check: std={test_denoised.std():.4f} | adapted std={test_adapted.std():.4f}")

    print("\n" + "=" * 60)
    print("  PHASE 5: Feature Extraction & Classification")
    print("=" * 60)

    n_comp_offline = min(config.csp_n_components, train_denoised.shape[1] // 2, len(np.unique(train_labels)))
    train_features, csp = extract_csp_features(train_denoised, train_labels, n_comp_offline)
    test_features = csp.transform(test_denoised)

    print("  CSP features extracted")

    train_features_clean, csp_clean = extract_csp_features(train_clean, train_labels, n_comp_offline)
    test_features_clean = csp_clean.transform(test_clean)

    classifier = EEGClassifier(method="svm", kernel=config.svm_kernel)
    print("\n  --- Classifier on DENOISED data ---")
    results_denoised = classifier.train(train_features, train_labels)

    classifier_clean = EEGClassifier(method="svm", kernel=config.svm_kernel)
    print("\n  --- Classifier on CLEAN data (upper bound) ---")
    results_clean = classifier_clean.train(train_features_clean, train_labels)

    print("\n" + "=" * 60)
    print("  PHASE 6: Self-Learning / Online Adaptation")
    print("=" * 60)

    engine = OnlineAdaptationEngine(
        denoiser=denoiser, classifier=classifier,
        confidence_threshold=0.7, adaptation_interval=5,
        online_lr=config.online_lr, mini_batch_size=config.mini_batch_size
    )

    adaptation_count = 0
    adaptation_data = []
    for i in range(min(50, test_adapted.shape[0])):
        result = engine.process_epoch(
            test_denoised[i],
            features=test_features[i:i+1],
            clean_target=test_clean[i]
        )
        adaptation_data.append({
            "shift": result.get("shift_detected", False),
            "confidence": 1.0 if not result.get("low_confidence", False) else 0.5,
            "adapted": result["adapted"],
        })
        if result["adapted"]:
            adaptation_count += 1

    stats = engine.get_stats()
    print(f"  Adaptations triggered: {adaptation_count}")
    print(f"  Stats: {stats}")

    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    test_denoised_raw = denormalize_trials(test_denoised, test_means, test_stds)

    metrics = evaluate_pipeline(
        test_pre,           # raw uV space reference
        test_denoised_raw,  # denormalized back to raw uV space
        test_clean,         # raw uV ground truth
        classifier=classifier, features=test_features, labels=test_labels
    )
    print_evaluation(metrics)

    lat = measure_latency(denoiser.denoise_epoch, test_denoised[0], n_runs=20)
    print(f"  Inference Latency: {lat['mean_ms']:.2f} ms (+/- {lat['std_ms']:.2f} ms)")
    real_time = "YES" if lat['mean_ms'] < 100 else "NO"
    print(f"  Real-time capable (< 100ms): {real_time}")

    y_pred = classifier.predict(test_features)
    y_scores = classifier.predict_proba(test_features)

    pipeline_data = {
        "X_noisy": test_noisy,
        "X_pre": test_pre,
        "X_adapted": test_adapted,
        "X_denoised": test_denoised,
        "X_clean": test_clean,
        "sfreq": config.sfreq,
        "losses": losses,
        "metrics": metrics,
        "y_true": test_labels,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "latency": lat,
        "adaptation_data": adaptation_data,
    }

    generate_all_figures(pipeline_data, output_dir="results")

    return metrics


def run_dataset_pipeline(X, y, meta, config):
    """Run the full pipeline on real MOABB dataset."""
    import torch

    config.n_channels = X.shape[1]
    config.sfreq = meta["sfreq"]
    config.notch_freq = 60.0 if meta["sfreq"] > 125 else meta["sfreq"] / 2.0 - 1.0

    print(f"  Shape: {X.shape} | Labels: {np.unique(y)} | sfreq: {config.sfreq}")

    unique_labels = np.unique(y)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_int = np.array([label_map[lbl] for lbl in y])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    print("\n" + "=" * 60)
    print("  PHASE 2: Baseline Preprocessing")
    print("=" * 60)

    def preprocess(data):
        out = apply_bandpass(data, config.l_freq, config.h_freq, config.sfreq)
        out = apply_notch(out, config.notch_freq, config.sfreq)
        return out

    X_train_pre = preprocess(X_train)
    X_test_pre = preprocess(X_test)
    X_train_pre = np.nan_to_num(X_train_pre, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_pre = np.nan_to_num(X_test_pre, nan=0.0, posinf=0.0, neginf=0.0)
    print("  Bandpass + Notch applied")

    print("\n" + "=" * 60)
    print("  PHASE 3: Adaptive Filtering")
    print("=" * 60)

    adapt_filter = AdaptiveFilter(
        n_channels=config.n_channels, method=config.filter_type,
        filter_length=config.lms_length, mu=config.lms_mu
    )

    from ai_denoising import normalize_trials, denormalize_trials

    X_train_norm, train_means, train_stds = normalize_trials(X_train_pre)
    X_test_norm, test_means, test_stds = normalize_trials(X_test_pre)

    X_train_adapted = np.zeros_like(X_train_norm)
    for i in range(X_train_norm.shape[0]):
        X_train_adapted[i] = adapt_filter.filter_signal(X_train_norm[i])

    adapt_filter.reset()
    X_test_adapted = np.zeros_like(X_test_norm)
    for i in range(X_test_norm.shape[0]):
        X_test_adapted[i] = adapt_filter.filter_signal(X_test_norm[i])

    X_train_adapted = np.nan_to_num(X_train_adapted, nan=0.0)
    X_test_adapted = np.nan_to_num(X_test_adapted, nan=0.0)
    print(f"  {config.filter_type.upper()} filter applied on {X_train.shape[0]} train / {X_test.shape[0]} test trials")

    print("\n" + "=" * 60)
    print("  PHASE 4: AI-Based Denoising (Training)")
    print("=" * 60)

    # For dataset mode, we don't have true clean data. 
    # Use self-supervised learning (denoising synthetic noise added to the already noisy X_train_adapted)
    dae_model, losses = train_dae(clean_data=X_train_adapted, config=config, noise_level_db=config.noise_snr_db)
    denoiser = OnlineDenoiser(dae_model)

    X_train_denoised = denoiser.denoise_batch(X_train_adapted)
    X_test_denoised = denoiser.denoise_batch(X_test_adapted)
    X_train_denoised = np.nan_to_num(X_train_denoised, nan=0.0)
    X_test_denoised = np.nan_to_num(X_test_denoised, nan=0.0)
    print(f"  Model trained | Final loss: {losses[-1]:.6f}")

    print("\n" + "=" * 60)
    print("  PHASE 5: Feature Extraction & Classification")
    print("=" * 60)

    n_classes = len(np.unique(y_train))
    n_comp = min(config.csp_n_components, X_train_denoised.shape[1] // 2, n_classes)
    if n_comp < 1:
        n_comp = 1

    train_features, csp = extract_csp_features(X_train_denoised, y_train, n_comp)
    test_features = csp.transform(X_test_denoised)
    print(f"  CSP features: {train_features.shape[1]} components")

    classifier = EEGClassifier(method="svm", kernel=config.svm_kernel)
    results = classifier.train(train_features, y_train)

    print("\n" + "=" * 60)
    print("  PHASE 6: Self-Learning / Online Adaptation")
    print("=" * 60)

    engine = OnlineAdaptationEngine(
        denoiser=denoiser, classifier=classifier,
        confidence_threshold=0.7, adaptation_interval=5,
        online_lr=config.online_lr, mini_batch_size=config.mini_batch_size
    )

    adapt_count = 0
    adaptation_data = []
    for i in range(min(50, X_test_denoised.shape[0])):
        result = engine.process_epoch(
            X_test_denoised[i], features=test_features[i:i+1],
            clean_target=X_test_adapted[i]
        )
        adaptation_data.append({
            "shift": result.get("shift_detected", False),
            "confidence": 1.0 if not result.get("low_confidence", False) else 0.5,
            "adapted": result["adapted"],
        })
        if result["adapted"]:
            adapt_count += 1
    print(f"  Adaptations triggered: {adapt_count}")

    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    X_test_denoised_raw = denormalize_trials(X_test_denoised, test_means, test_stds)

    metrics = evaluate_pipeline(
        X_test_pre,           # Since we don't have true clean, evaluate SNR against preprocessed signal
        X_test_denoised_raw,
        X_test_pre,           # Dummy ground truth
        classifier=classifier, features=test_features, labels=y_test
    )
    print_evaluation(metrics)

    lat = measure_latency(denoiser.denoise_epoch, X_test_denoised[0], n_runs=20)
    print(f"  Inference Latency: {lat['mean_ms']:.2f} ms (+/- {lat['std_ms']:.2f} ms)")
    real_time = "YES" if lat['mean_ms'] < 100 else "NO"
    print(f"  Real-time capable (< 100ms): {real_time}")

    y_pred = classifier.predict(test_features)
    y_scores = classifier.predict_proba(test_features)

    pipeline_data = {
        "X_noisy": X_test_pre,
        "X_pre": X_test_pre,
        "X_adapted": X_test_adapted,
        "X_denoised": X_test_denoised,
        "X_clean": X_test_adapted,
        "sfreq": config.sfreq,
        "losses": losses,
        "metrics": metrics,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "latency": lat,
        "adaptation_data": adaptation_data,
    }

    generate_all_figures(pipeline_data, output_dir="results")

    return metrics


def run_streaming_pipeline(config):
    """Run pipeline with LSL mock streaming."""
    from data_pipeline import mock_lsl_stream, LSLReceiver
    from threading import Thread

    print("\n[Streaming] Generating synthetic data for LSL...")
    data = generate_eeg_trials(
        n_trials=10, n_channels=config.n_channels,
        n_samples=config.epoch_samples, sfreq=config.sfreq
    )

    stream_data = data[0]

    streamer_thread = Thread(
        target=mock_lsl_stream,
        args=(stream_data, config.sfreq, config.lsl_stream_name, config.lsl_chunk_size),
        daemon=True
    )
    streamer_thread.start()

    receiver = LSLReceiver(stream_name=config.lsl_stream_name,
                           epoch_samples=config.epoch_samples)
    try:
        receiver.connect(timeout=5.0)
        print("[Streaming] Waiting for epoch...")
        epoch = receiver.receive_epoch(timeout=10.0)
        if epoch is not None:
            print(f"[Streaming] Received epoch: {epoch.shape}")

            processed = apply_bandpass(epoch, config.l_freq, config.h_freq, config.sfreq)
            processed = apply_notch(processed, config.notch_freq, config.sfreq)
            print("[Streaming] Preprocessed epoch through pipeline")
        else:
            print("[Streaming] No epoch received (timeout)")
    except Exception as e:
        print(f"[Streaming] LSL not available: {e}")
        print("[Streaming] Falling back to offline mode...")
        return run_offline_pipeline(config)

    return None


def main():
    parser = argparse.ArgumentParser(description="EEG Adaptive Denoising Pipeline")
    parser.add_argument("--mode", choices=["offline", "dataset", "streaming"],
                        default="offline", help="Pipeline mode")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--noise-snr", type=float, default=-5.0,
                        help="Noise level in dB")
    parser.add_argument("--filter", choices=["lms", "rls"], default="lms")
    parser.add_argument("--classifier", choices=["svm", "lda"], default="svm")

    args = parser.parse_args()

    config = PipelineConfig(
        n_channels=args.channels,
        sfreq=args.sfreq,
        noise_snr_db=args.noise_snr,
        filter_type=args.filter,
    )

    print("=" * 62)
    print("  Adaptive AI EEG Denoising & Classification Pipeline")
    print(f"  Mode: {args.mode}")
    print("=" * 62)

    if args.mode == "offline":
        run_offline_pipeline(config)
    elif args.mode == "streaming":
        run_streaming_pipeline(config)
    elif args.mode == "dataset":
        print("[Dataset] Loading MOABB dataset...")
        from data_pipeline import fetch_eeg_dataset
        X, y, meta = fetch_eeg_dataset()
        if X.shape[0] == 0:
            print("  ERROR: No trials loaded from dataset!")
            return
        print(f"  Loaded: {X.shape} | sfreq={meta['sfreq']}")
        run_dataset_pipeline(X, y, meta, config)


if __name__ == "__main__":
    main()
