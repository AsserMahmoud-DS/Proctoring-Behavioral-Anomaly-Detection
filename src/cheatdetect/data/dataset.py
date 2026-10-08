"""Fresh paired study representations; never use the legacy feature/sequence caches.

Builds the single manifest-guarded dataset consumed by both the experiment
runner and (later) the production pipeline: it fits every preprocessing recipe on
original normal training only and freezes the result for synthetic and held-out
inputs.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from cheatdetect.config import (
    AUDIT_DIR,
    EXPERIMENT_1_DIR,
    MIXED_DIR,
    NORMAL_DIR,
    PACKAGE_DIR,
    STUDY_ARTIFACTS_DIR,
)
from cheatdetect.utils import _find_project_root
from cheatdetect.data.feature_schema import SOURCE_FEATURES
from cheatdetect.data.paired import (
    PairedRepresentation as Representation,
    build_paired_representation as build_representation,
    extract_paired_features as _paired_features,
    load_parent_sessions,
)
from cheatdetect.data.preprocessing import FeaturePreprocessor, gamma_center

ARTIFACT_ROOT = STUDY_ARTIFACTS_DIR
PARTITIONS = ("train_original", "train_synthetic", "normal_val", "mixed_val", "mixed_test")
SPLIT_KEYS = ("normal_train", "normal_val", "mixed_val", "mixed_test")


@dataclass
class PreparedStudy:
    manifest: dict
    raw: dict[str, Representation]
    preprocessors: dict[str, FeaturePreprocessor]
    transformed: dict[str, dict[str, pd.DataFrame | np.ndarray]]
    gamma_centers: dict[str, float]
    session_report: pd.DataFrame

    def training(self, recipe: str, augmented: bool):
        original = self.transformed[recipe]["train_original"]
        if not augmented:
            return original
        synthetic = self.transformed[recipe]["train_synthetic"]
        if isinstance(original, pd.DataFrame):
            return pd.concat([original, synthetic], ignore_index=True)
        return np.concatenate([original, synthetic], axis=0)

    def training_identities(self, augmented: bool) -> pd.DataFrame:
        original = self.raw["train_original"].identities
        if not augmented:
            return original
        return pd.concat([original, self.raw["train_synthetic"].identities], ignore_index=True)


def input_manifest(
    split: dict, normal_dir: Path, mixed_dir: Path,
    n_copies: int = 2, sigma_range: tuple[float, float] = (2.0, 5.0),
) -> dict:
    if any(not split.get(key) for key in SPLIT_KEYS):
        raise ValueError("All frozen study splits must contain session files")
    if set(split["normal_train"]) & set(split["normal_val"]):
        raise ValueError("Normal train/validation sessions overlap")
    if set(split["mixed_val"]) & set(split["mixed_test"]):
        raise ValueError("Mixed validation/test sessions overlap")
    files = {}
    for key in SPLIT_KEYS:
        directory = normal_dir if key.startswith("normal") else mixed_dir
        if len(set(split[key])) != len(split[key]):
            raise ValueError("Duplicate session in study split")
        for name in split[key]:
            if Path(name).name != name:
                raise ValueError("Study session names must be basenames")
            files[f"{key}/{name}"] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    root = _find_project_root()
    production = PACKAGE_DIR / "data"
    source_files = [
        Path(__file__),
        production / "preprocessing.py",
        EXPERIMENT_1_DIR / "audit.py",
    ]
    source_files.extend([production / "cleaning.py", production / "build.py", production / "augment.py",
                         production / "feature_schema.py", production / "preprocessing.py", production / "paired.py"])
    source_files.extend(sorted((production / "features").glob("*.py")))
    return {
        "version": 2, "split": {key: list(split[key]) for key in SPLIT_KEYS},
        "file_hashes": files, "source_features": list(SOURCE_FEATURES),
        "parent_events": 50, "parent_stride": 25, "micro_events": 10,
        "micro_stride": 5, "label_fraction": 0.5,
        "augmentation": {"copies": n_copies, "sigma_range": list(sigma_range),
                         "seed": 42, "screen_bounds": [1920, 1080]},
        "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }


def prepare_study(
    split: dict | None = None, normal_dir: Path = NORMAL_DIR, mixed_dir: Path = MIXED_DIR,
    n_copies: int = 2, sigma_range: tuple[float, float] = (2.0, 5.0),
) -> PreparedStudy:
    """Build fresh data and fit each recipe once; never fit on synthetic/held-out rows."""
    if split is None:
        split = json.loads((AUDIT_DIR / "manifest.json").read_text())["split"]
        frozen_hashes = pd.read_csv(AUDIT_DIR / "sessions.csv").set_index("session_file")["sha256"]
        for name in split["normal_train"]:
            if hashlib.sha256((normal_dir / name).read_bytes()).hexdigest() != frozen_hashes[name]:
                raise ValueError("Original training contents changed since the frozen Phase 1 audit")
    manifest = input_manifest(split, normal_dir, mixed_dir, n_copies, sigma_range)
    loaded, session_reports = {}, []
    for key in SPLIT_KEYS:
        directory = normal_dir if key.startswith("normal") else mixed_dir
        loaded[key], report = load_parent_sessions(
            [directory / name for name in split[key]], partition=key,
            normal_only=key.startswith("normal"),
        )
        session_reports.append(report)
    raw = {
        "train_original": build_representation(loaded["normal_train"]),
        "train_synthetic": build_representation(loaded["normal_train"], True, n_copies, sigma_range),
        **{key: build_representation(loaded[key]) for key in SPLIT_KEYS[1:]},
    }
    preprocessors, transformed, centers = {}, {}, {}
    for name, recipe, scale in (
        ("IF-raw", "base", False), ("IF-log", "log", False),
        ("SVM-base", "base", True), ("SVM-log", "log", True),
        ("SVM-YJ", "yj", True), ("SVM-quantile", "quantile", True),
        ("AE-base", "base", True), ("AE-log", "log", True), ("AE-YJ", "yj", True),
    ):
        original = raw["train_original"]
        fit_frame = pd.DataFrame(original.sequences.reshape(-1, 25), columns=SOURCE_FEATURES) if name.startswith("AE-") else original.parent
        processor = FeaturePreprocessor(recipe, scale).fit(fit_frame)
        preprocessors[name] = processor
        transformed[name] = {
            key: processor.transform_sequences(view.sequences) if name.startswith("AE-") else processor.transform(view.parent)
            for key, view in raw.items()
        }
        if name.startswith("SVM-"):
            centers[name] = gamma_center(transformed[name]["train_original"])
    return PreparedStudy(manifest, raw, preprocessors, transformed, centers, pd.concat(session_reports, ignore_index=True))


def _output_directory(directory: Path) -> Path:
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("Study artifacts must stay inside study/artifacts")
    return directory


def save_prepared(study: PreparedStudy, directory: Path) -> None:
    directory = _output_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, directory / "prepared.joblib")
    (directory / "manifest.json").write_text(json.dumps(study.manifest, indent=2) + "\n")
    study.session_report.to_csv(directory / "session_readiness.csv", index=False)
    (directory / "preprocessing.json").write_text(json.dumps({
        "gamma_centers": study.gamma_centers,
        "output_features": next(iter(study.preprocessors.values())).output_features_,
        "original_fit_rows": {name: processor.n_original_rows_ for name, processor in study.preprocessors.items()},
    }, indent=2) + "\n")


def load_prepared(directory: Path, expected_manifest: dict) -> PreparedStudy:
    """Load trusted local joblib only after matching fresh input/specification hashes."""
    directory = _output_directory(directory)
    saved_manifest = json.loads((directory / "manifest.json").read_text())
    if saved_manifest != expected_manifest:
        raise ValueError("Prepared study manifest does not match current inputs/specification")
    prepared = joblib.load(directory / "prepared.joblib")
    if prepared.manifest != expected_manifest:
        raise ValueError("Prepared bundle and manifest disagree")
    return prepared
