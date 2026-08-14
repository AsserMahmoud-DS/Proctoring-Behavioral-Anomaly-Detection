from dataclasses import dataclass

from cheatdetect.utils import _find_project_root

_project_root = _find_project_root()

# Environment / deployment constants
NORMAL_DIR = _project_root / "data/raw/pure normal"
MIXED_DIR = _project_root / "data/raw/mixed"
PROCESSED_DIR = _project_root / "data/processed"
MODELS_DIR = _project_root / "best_models"
REPORTS_DIR = _project_root / "reports"

# Processed data paths (per-split)
TRAIN_NORMAL_PATH = PROCESSED_DIR / "train_normal.pkl"
VAL_NORMAL_PATH = PROCESSED_DIR / "val_normal.pkl"
VAL_MIXED_PATH = PROCESSED_DIR / "val_mixed.pkl"
TEST_MIXED_PATH = PROCESSED_DIR / "test_mixed.pkl"
SPLIT_INFO_PATH = PROCESSED_DIR / "split_info.json"

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

# if __name__ == "__main__":
#     cfg1 = ExperimentConfig(chunk_size = 50, if_n_estimators = (20,30,40), ocsvm_nu = (1,2,3))
#     print(cfg1.if_n_estimators,cfg1.ocsvm_nu )
