"""Configuration for the UDA cross-subject experiment.

Every value that the paper leaves as "[To be determined]" is exposed here
with a documented default.  See section 5.3 / Table 4 of the paper.
"""

from dataclasses import dataclass, field


METHODS = ("source_only", "coral", "mmd", "dann")


@dataclass
class UDAConfig:
    # ------------------------------------------------------------------
    # Data source
    # ------------------------------------------------------------------
    mode: str = "synthetic"          # "synthetic" (quick check) | "dataset" (MOABB)
    dataset_name: str = "bci2a"      # key into data_pipeline.dataset_fetcher.DATASETS
    subjects: list = field(default_factory=lambda: [1, 2, 3])
    tmin: float = 0.0
    tmax: float = 4.0
    n_classes: int = 2

    # Synthetic mode (used for a dependency-free smoke test of the framework)
    synthetic_subjects: int = 3
    synthetic_trials_per_subject: int = 60
    synthetic_channels: int = 16
    synthetic_samples: int = 250
    synthetic_sfreq: float = 250.0

    # ------------------------------------------------------------------
    # Preprocessing (paper section 4.3)
    # ------------------------------------------------------------------
    band: tuple = (8.0, 30.0)        # mu + beta rhythms
    use_car: bool = True             # Common Average Referencing
    use_notch: bool = True
    notch_freq: float = 50.0

    # ------------------------------------------------------------------
    # Source / target domain construction (paper sections 4.2, 4.7)
    # ------------------------------------------------------------------
    target_test_ratio: float = 0.2   # held-out target test partition
    target_split_mode: str = "stratified"  # "stratified" | "chronological"
    target_proportions: tuple = (1.0,)     # paper: 0.25 / 0.5 / 0.75 / 1.0
    source_val_ratio: float = 0.2    # source-only validation for model selection

    # ------------------------------------------------------------------
    # Feature preparation (paper section 4.3 / related work 2.1)
    # ------------------------------------------------------------------
    feature_kind: str = "fbcsp"      # "fbcsp" | "logpower" (robust fallback)
    fbcsp_bands: list = field(default_factory=lambda: [(8.0, 12.0), (13.0, 30.0)])
    csp_components: int = 4
    logpower_bands: list = field(default_factory=lambda: [(8.0, 12.0), (13.0, 30.0)])

    # ------------------------------------------------------------------
    # Network architecture (paper section 4.6: F, C and D)
    # ------------------------------------------------------------------
    hidden_dims: tuple = (64,)
    feature_dim: int = 32            # d in the CORAL loss
    dropout: float = 0.1

    # ------------------------------------------------------------------
    # Training and adaptation losses (paper sections 4.4 - 4.6)
    # ------------------------------------------------------------------
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Features are standardised and CORAL is divided by 4*d^2, so its raw value
    # is very small (~1e-5). A larger lambda keeps it comparable to L_cls.
    lambda_coral: float = 1000.0     # lambda_CORAL in L = L_cls + lambda * L_CORAL
    lambda_mmd: float = 1.0          # lambda_MMD
    lambda_dann: float = 1.0         # lambda_DANN (adversarial coefficient)
    mmd_kernel: str = "rbf"
    mmd_sigmas: tuple = (0.1, 0.5, 1.0, 2.0)
    use_grl_schedule: bool = True    # DANN lambda ramp: 2/(1+exp(-10p)) - 1
    early_stop_patience: int = 10

    # ------------------------------------------------------------------
    # Experiment control
    # ------------------------------------------------------------------
    methods: tuple = METHODS
    seed: int = 42
    device: str = "auto"             # "auto" | "cpu" | "cuda"
    output_dir: str = "results_uda"
    verbose: bool = True

    def __post_init__(self):
        self.methods = tuple(self.methods)
        self.target_proportions = tuple(float(p) for p in self.target_proportions)
        self.hidden_dims = tuple(self.hidden_dims)
        self.mmd_sigmas = tuple(float(s) for s in self.mmd_sigmas)
        self.band = tuple(float(b) for b in self.band)
        for name in ("methods", "target_proportions"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for p in self.target_proportions:
            if not 0.0 < p <= 1.0:
                raise ValueError(f"target proportion must be in (0, 1], got {p}")
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown method(s): {sorted(unknown)}. Choose from {METHODS}")
        if not 0.0 < self.target_test_ratio < 1.0:
            raise ValueError("target_test_ratio must be in (0, 1)")
        if not 0.0 <= self.source_val_ratio < 1.0:
            raise ValueError("source_val_ratio must be in [0, 1)")

    def lambda_for(self, method: str) -> float:
        return {
            "source_only": 0.0,
            "coral": self.lambda_coral,
            "mmd": self.lambda_mmd,
            "dann": self.lambda_dann,
        }[method]
