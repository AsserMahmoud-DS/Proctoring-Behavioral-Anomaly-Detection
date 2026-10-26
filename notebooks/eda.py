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
# # Exploratory Data Analysis
#
# Feature-level EDA on the **original normal training data only** (leakage
# prevention). Reuses the exact data path as training via `prepare_data`, so
# the split and window geometry match what the models see. The feature schema
# is frozen at 25 source features (33 encoded by `FeaturePreprocessor`), so
# this notebook inspects domains and redundancy rather than selecting features.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from cheatdetect.config import EDA_DIR, ExperimentConfig
from cheatdetect.data.feature_schema import SOURCE_FEATURES
from cheatdetect.pipeline import prepare_data

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

config = ExperimentConfig()

# %%
# ============================================================
# Prepare data (shared with training for reproducibility)
# ============================================================
data = prepare_data(config)
study = data["study"]

# %%
# ============================================================
# Dataset Summary — paired windows per partition
# ============================================================
summary = pd.DataFrame(
    [
        {
            "partition": partition,
            "chunks": len(view.parent),
            "anomalous_chunks": int(view.labels.sum()),
            "sessions": view.identities["session_file"].nunique(),
        }
        for partition, view in study.raw.items()
    ]
).set_index("partition")
summary["normal_chunks"] = summary["chunks"] - summary["anomalous_chunks"]
summary["anomalous_pct"] = (
    summary["anomalous_chunks"] / summary["chunks"] * 100
).round(1)
summary

# %%
# ============================================================
# Domain Check + Zero-Variance Features (Original Normal Train Only)
# ============================================================
X_train = study.raw["train_original"].parent[list(SOURCE_FEATURES)]

zero_variance = [
    feature for feature in SOURCE_FEATURES if X_train[feature].var() == 0
]
print(f"Source features: {len(SOURCE_FEATURES)}")
print(f"Zero-variance features: {len(zero_variance)}")
for feature in zero_variance:
    print(f"  - {feature}")
if not zero_variance:
    print("  (none)")

# %%
# ============================================================
# EDA: Feature Distributions (Original Normal Train Only)
# ============================================================
n_cols = 4
n_rows = (len(SOURCE_FEATURES) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, feature in enumerate(SOURCE_FEATURES):
    axes[i].hist(X_train[feature].dropna(), bins=30, alpha=0.7, edgecolor="black")
    axes[i].set_title(feature, fontsize=9)
    axes[i].tick_params(labelsize=7)

for j in range(len(SOURCE_FEATURES), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(EDA_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

print("Note: Sparse features (e.g. copy_events, window_switch_events) are normal.")
print("They are valuable discriminators for anomaly detection — not noise.")

# %%
# ============================================================
# EDA: Correlation Analysis (Original Normal Train Only)
# ============================================================
corr = X_train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

pairs = corr.where(mask).stack()
pairs = (
    pairs[pairs.abs() > config.high_corr_threshold]
    .rename("correlation")
    .reset_index()
    .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
    .sort_values("correlation", key=lambda values: values.abs(), ascending=False)
)
print(f"=== Correlated Pairs (|r| > {config.high_corr_threshold}) ===")
print(f"Found {len(pairs)} pairs\n")
if len(pairs):
    display(pairs)
else:
    print("None.")

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr, mask=mask, cmap="RdBu_r", center=0,
    xticklabels=False, yticklabels=False,
)
plt.title("Feature Correlation Matrix (Original Normal Train Only)")
plt.tight_layout()
plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ============================================================
# Per-Feature Summary
# ============================================================
max_abs_corr = corr.abs().where(mask).max()
schema_summary = pd.DataFrame(
    {
        "feature": list(SOURCE_FEATURES),
        "zero_variance": [feature in zero_variance for feature in SOURCE_FEATURES],
        "max_abs_correlation": [
            max_abs_corr[feature] for feature in SOURCE_FEATURES
        ],
    }
)
print("The 25-source schema is frozen; correlated pairs are reported above but")
print("redundancy is handled by the audited recipes, not dropped here.")
schema_summary
