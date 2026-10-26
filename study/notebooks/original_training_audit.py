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
# # Study Phase 1: original normal-training audit
#
# This reads saved training-only diagnostics, not the legacy augmented EDA.
# To regenerate from the repository root: `python -m cheatdetect.experiments.experiment_1.audit`.
# No validation/test contents, scores, detectors, or augmentation are used.
# Specification corrections remain subject to review before Phase 2.

# %%
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

from cheatdetect.utils import _find_project_root

sys.path.insert(0, str(_find_project_root()))

from cheatdetect.experiments.experiment_1.audit import AUDIT_DIR

manifest = json.loads((AUDIT_DIR / "manifest.json").read_text())
pd.Series({key: manifest[key] for key in (
    "original_training_sessions", "parent_rows", "micro_rows_with_overlap",
    "encoded_dimension", "if_effective_max_samples", "specification_status",
)})

# %% [markdown]
# ## Definitions and numerical validity
#
# Negative jerk is expected: it differentiates acceleration magnitude, so it
# measures increasing/decreasing acceleration rather than vector jerk magnitude.
# Direction 0 is zero endpoint displacement/insufficient mouse events; it does
# not imply no movement. Codes 1–8 are 45-degree sectors starting at +x.
# Read `study/plans/phase_01_findings.md` for units and caveats.

# %%
parent_features = pd.read_csv(AUDIT_DIR / "parent_features.csv")
micro_features = pd.read_csv(AUDIT_DIR / "micro_features.csv")
pd.concat({"parent": parent_features, "micro": micro_features}, names=["representation"])

# %%
pd.read_csv(AUDIT_DIR / "sessions.csv")

# %% [markdown]
# ## Recipe effects, not feature re-selection
#
# These summaries describe all retained source features and all nine direction
# indicators. Constants remain. Micro rows overlap and are not independent.

# %%
transformed_features = pd.read_csv(AUDIT_DIR / "transformed_features.csv")
transformed_features["absolute_extreme"] = transformed_features[["min", "max"]].abs().max(axis=1)
transformed_features

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for axis, representation in zip(axes, ("parent", "micro")):
    summaries = transformed_features[
        transformed_features["representation"].eq(representation)
        & ~transformed_features["recipe"].str.startswith("IF-")
    ]
    summaries.pivot(index="feature", columns="recipe", values="absolute_extreme").plot.bar(
        ax=axis, logy=True, fontsize=7,
    )
    axis.set_title(f"{representation}: absolute column maximum after recipe")
    axis.set_ylabel("Absolute maximum (log axis)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## OCSVM training geometry
#
# Gamma uses variance over **all entries**, not average per-column variance.
# Off-diagonal kernels use up to 512 training rows (seed 42), not held-out labels.
# A broad kernel range supports the bounded draft; it is not proof of accuracy.

# %%
pd.read_csv(AUDIT_DIR / "kernel_geometry.csv")
