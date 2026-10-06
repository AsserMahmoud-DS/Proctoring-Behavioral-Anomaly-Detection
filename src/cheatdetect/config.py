from dataclasses import dataclass

from cheatdetect.utils import _find_project_root

_project_root = _find_project_root()

# Environment / deployment constants
NORMAL_DIR = _project_root / "dataset/raw/pure normal"
MIXED_DIR = _project_root / "dataset/raw/mixed"
PROCESSED_DIR = _project_root / "dataset/processed"
MODELS_DIR = _project_root / "best_models"
REPORTS_DIR = _project_root / "reports"

# Processed data paths (per-split)
TRAIN_NORMAL_PATH = PROCESSED_DIR / "train_normal.pkl"
VAL_NORMAL_PATH = PROCESSED_DIR / "val_normal.pkl"
VAL_MIXED_PATH = PROCESSED_DIR / "val_mixed.pkl"
TEST_MIXED_PATH = PROCESSED_DIR / "test_mixed.pkl"
SPLIT_INFO_PATH = PROCESSED_DIR / "split_info.json"
TRAIN_AUGMENTED_PATH = PROCESSED_DIR / "train_augmented.pkl"
FEATURE_LISTS_PATH = PROCESSED_DIR / "feature_lists.json"

# LSTM sequence caches (raw micro-chunk sequences, pre-feature-selection)
LSTM_TRAIN_PATH = PROCESSED_DIR / "lstm_train.npz"
LSTM_VAL_PATH = PROCESSED_DIR / "lstm_val.npz"
LSTM_TEST_PATH = PROCESSED_DIR / "lstm_test.npz"
LSTM_ES_PATH = PROCESSED_DIR / "lstm_es.npz"

# Reports directories
EDA_DIR = REPORTS_DIR / "eda"
VAL_DIR = REPORTS_DIR / "val_results"
TEST_DIR = REPORTS_DIR / "test_results"


@dataclass(frozen=True)
class ExperimentConfig:
    chunk_size: int = 50
    step_size: int = 25
    cheating_threshold: float = 0.5
    random_state: int = 42
    normal_val_size: float = 0.2
    mixed_val_size: float = 0.3
    aug_enabled: bool = True
    aug_n_copies: int = 2
    aug_sigma_min: float = 2.0
    aug_sigma_max: float = 5.0

    # Feature selection
    high_corr_threshold: float = 0.85
    skew_threshold: float = 2.0

    # Threshold tuning
    precision_floor: float = 0.5

    # Isolation Forest grid
    if_n_estimators: tuple[int, ...] = (100, 200, 300)
    if_max_samples: tuple = (256, 0.8, "auto")
    if_contamination: tuple[float, ...] = (0.01, 0.05, 0.1)

    # One-Class SVM grid
    ocsvm_nu: tuple[float, ...] = (0.01, 0.05, 0.1)
    ocsvm_gamma: tuple = ("scale", "auto", 0.1, 0.01)
    ocsvm_kernel: tuple[str, ...] = ("rbf",)

    # Ensemble
    ensemble_weights: tuple[float, ...] = (0.3, 0.5, 0.7)

    # LSTM autoencoder (research-only comparison model).
    # Disabled by default: torch is a dev-only dependency and the 108-config
    # grid is expensive. The train_evaluate notebook opts in explicitly.
    lstm_enabled: bool = False
    lstm_sub_chunk: int = 10
    lstm_sub_step: int = 5
    lstm_hidden_dims: tuple[int, ...] = (8, 16, 32)
    lstm_num_layers: tuple[int, ...] = (1, 2)
    lstm_dropouts: tuple[float, ...] = (0.1, 0.3)
    lstm_lrs: tuple[float, ...] = (1e-3, 5e-4, 1e-4)
    lstm_batch_sizes: tuple[int, ...] = (32, 64, 128)
    lstm_epochs: int = 100
    lstm_patience: int = 10
    # Early-stopping source for the AE: held-out normal sequences
    # ("normal_val"), the combined (mixed) val ("mixed_val"), or no early
    # stopping at all ("none"). "none" is supported but not evaluated.
    lstm_es_source: str = "normal_val"
    # "standard" (AE recipe) or "log1p_robust" (flat-model recipe).
    lstm_preprocessing: str = "standard"

# if __name__ == "__main__":
#     cfg1 = ExperimentConfig(chunk_size = 50, if_n_estimators = (20,30,40), ocsvm_nu = (1,2,3))
#     print(cfg1.if_n_estimators,cfg1.ocsvm_nu )
