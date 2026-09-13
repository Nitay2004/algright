from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # Sampling rate
    sfreq: int = 250

    # Bandpass filter
    l_freq: float = 0.5
    h_freq: float = 50.0

    # Notch filter
    notch_freq: float = 50.0

    # Adaptive filter (line-noise canceller: synthesized 50/60 Hz reference)
    filter_type: str = "lms"  # "lms" or "rls"
    lms_length: int = 2  # sin+cos reference taps
    lms_mu: float = 0.02
    rls_length: int = 2
    rls_lambda: float = 0.98
    rls_delta: float = 0.1

    # AI Denoising model
    n_channels: int = 64
    epoch_samples: int = 250  # 1 second
    latent_dim: int = 32

    # Guarded blend for the DAE output: final = alpha*dae + (1-alpha)*ref
    # (ref = bandpass+notch). Prevents over-smoothing on real data.
    dae_blend_alpha: float = 0.8

    # Task-aware finetuning: reconstruction + classification multi-task loss
    dae_finetune: bool = True
    dae_finetune_epochs: int = 10
    dae_lambda_rec: float = 0.5  # weight on the reconstruction term
    dae_finetune_lr: float = 1e-4

    # Training
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 30

    # Self-learning
    adaptation_interval: int = 10  # epochs between adaptation
    mini_batch_size: int = 16
    online_lr: float = 1e-4

    # CSP / Classification
    csp_n_components: int = 8
    svm_kernel: str = "rbf"
    use_fbcsp: bool = True
    fbcsp_bands: list = field(default_factory=lambda: [(8.0, 12.0), (13.0, 30.0)])
    classifier: str = "auto"  # svm | lda | auto (CV-based best)

    # Real-data preprocessing
    use_car: bool = True
    artifact_rejection_ratio: float = 4.0  # reject trials where a channel's
    # peak-to-peak exceeds this multiple of its dataset-wide median

    # Real-time inference chain: which stages run inside the live path.
    # LMS is excluded by default: on real EEG there is no correlated noise
    # reference (a secondary sensor), so LMS ANC degrades both latency
    # (217 ms on real data) and classification. The DAE is included but can
    # be disabled if it does not help a given dataset.
    use_lms_in_real_time: bool = False
    use_dae_in_real_time: bool = True

    # Evaluation
    noise_snr_db: float = -5.0
    test_noise_shift_db: float = 3.0  # extra noise on test (session shift)

    # LSL Streaming
    lsl_stream_name: str = "SimulatedEEG"
    lsl_chunk_size: int = 250  # 1 second chunks
    lsl_overlap: int = 0

    # Dataset
    dataset_name: str = "physionet"  # physionet | bci2a | openbmi | weibo2014
    dataset_subjects: list = field(default_factory=lambda: [1, 2, 3])
