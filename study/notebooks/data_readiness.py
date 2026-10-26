# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: cheatdetect
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Study Phase 2: data and frozen preprocessing
#
# Run from the project environment. This rebuilds fresh paired representations
# without legacy caches, fits preprocessing on original training only, and saves
# an isolated readiness bundle. It loads held-out inputs for automatic validity
# checks, but fits no detectors and computes no validation/test metrics or EDA.

# %%
import sys

import numpy as np
import pandas as pd

from cheatdetect.utils import _find_project_root

sys.path.insert(0, str(_find_project_root()))

from cheatdetect.data.dataset import ARTIFACT_ROOT, prepare_study, save_prepared

prepared = prepare_study()
directory = ARTIFACT_ROOT / "phase_02_readiness"
save_prepared(prepared, directory)

# %% [markdown]
# ## Automated readiness summary
#
# Each parent has one aligned nine-timestep sequence. Synthetic copy indices
# identify copies, not additional independent sessions. Both augmentation states
# reuse the same preprocessing objects and the same held-out representations.
# Zero-window files remain in the frozen split and are explicitly reported below;
# they contribute no windows. An entirely unusable required partition fails.

# %%
session_summary = prepared.session_report.groupby("partition", sort=False).agg(
    assigned_session_files=("session_file", "size"),
    contributing_session_files=("included", "sum"),
    parent_rows=("parent_windows", "sum"),
)
session_summary["excluded_session_files"] = (
    session_summary["assigned_session_files"] - session_summary["contributing_session_files"]
)
session_summary

# %%
prepared.session_report.loc[~prepared.session_report["included"]]

# %%
pd.DataFrame([
    {
        "partition": name,
        "parent_rows": len(view.parent),
        "sequence_rows": len(view.sequences),
        "contributing_original_session_files": view.identities["session_file"].nunique(),
        "nonfinite_raw_cells": int((~np.isfinite(view.parent.to_numpy())).sum()),
    }
    for name, view in prepared.raw.items()
])

# %%
pd.DataFrame([
    {
        "recipe": name,
        "original_fit_rows": processor.n_original_rows_,
        "train_off_rows": len(prepared.training(name, False)),
        "train_on_rows": len(prepared.training(name, True)),
        "encoded_columns": len(processor.output_features_),
        "gamma_center": prepared.gamma_centers.get(name),
    }
    for name, processor in prepared.preprocessors.items()
])

# %% [markdown]
# ## Reuse contract
#
# Later study runs may load this trusted local bundle only with a freshly computed
# `input_manifest` matching raw-file hashes, ordered splits, extraction/recipe
# code, geometry, augmentation settings, and library versions. No artifact under
# `dataset/processed/` or `best_models/` is overwritten.
