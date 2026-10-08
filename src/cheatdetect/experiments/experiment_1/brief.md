# Experiment 1 — Preprocessing & Model Feasibility Study

## Aim

Determine whether Isolation Forest (IF), One-Class SVM (OCSVM), and an LSTM
autoencoder (AE) can learn a useful behavioral anomaly signal from this small
session dataset, and quantify how complete preprocessing recipes, augmentation,
and a bounded AE depth comparison affect the result.

This is a proof of concept. It is **not** exhaustive optimization and **not** a
production-readiness assessment.

## Motivation

Earlier ad-hoc runs were confounded: the flat models applied a
`Log1pSkewed + RobustScaler` recipe while the AE used `StandardScaler`, and
augmentation was tangled with preprocessing. A single ablation on the validation
set showed preprocessing — not augmentation — collapsed the AE (ROC ≈ 0.46 vs
0.77). Experiment 1 replaces that ad-hoc pipeline with a frozen, audited,
manifest-guarded protocol so every recipe is compared under identical data,
preprocessing-fit rules, and selection criteria.

## Research questions

1. Which complete preprocessing recipe (raw / selective log / Yeo–Johnson /
   quantile) gives the best validation ranking per model family?
2. Does training-only Gaussian coordinate augmentation help, measured both as a
   matched-configuration intervention and as independently tuned condition
   winners?
3. Does a two-layer AE encoder/decoder differ from one layer under a bounded
   depth comparison?
4. How sensitive are the locked IF/AE finalists to the model-initialization seed?

## Evaluation contract (frozen)

- **Split:** seed 42, normal-val fraction 0.2, mixed-val fraction 0.3; the actual
  ordered file assignments are frozen, not just the seed.
- **Windows:** parent 50 events / stride 25; AE micro-windows 10 events / stride 5
  → 9 timesteps per parent.
- **Preprocessing:** nine fixed recipes, fitted on original normal training only,
  then frozen for synthetic/held-out inputs. IF recipes are unscaled; OCSVM/AE
  use block-specific `StandardScaler`.
- **Selection:** validation ROC-AUC, first-candidate tie-break; PR-AUC reported
  with anomaly prevalence as its baseline.
- **Threshold:** validation recall-maximizing subject to precision ≥ 0.5, with
  the existing F1 fallback; classification is `score >= threshold` everywhere.
- **Augmentation:** training-only, two copies per parent, coordinate-noise sigma
  in [2, 5] pixels, seed 42; flat and AE views share the same realized noisy
  windows.
- **Seeds:** 42 for the main search; 123 and 2026 only as predefined repeats of
  locked IF/AE finalists (never choose the best seed).
- **Test data:** never used for selection; evaluated once in the final reporting
  stage against a frozen manifest.

## Phased protocol

| Phase | Description | Status |
|---|---|---|
| 0 | Persist the phased plan | Done |
| 1 | Original-training audit + specification freeze | Done |
| 2 | Data & preprocessing readiness (frozen recipes, paired aug off/on) | Done |
| 3 | Model controls + isolated validation-only study runner | Not started |
| 4 | Seed-42 full paired grids (204 fits); lock 18 recipe winners + 2 ensembles | Not started |
| 5 | Locked-finalist seed sensitivity (8 fits; 212 total) | Not started |
| 6 | Final test reporting and interpretation | Not started |

## Scope / non-goals

No repeated-split study, nested cross-validation, exhaustive optimization,
per-feature ablation, duplicate augmented-data EDA, or experiment-tracking
platform. No automatic deployment promotion.

## Current status

- Study Phases 1–2 complete: the original-training audit and the paired
  readiness bundle are built and independently verified.
- Integration Phase A complete: reusable experiment code is now tracked under
  `src/cheatdetect/` (`data/dataset.py`, `experiments/experiment_1/audit.py`),
  so a clean clone can reproduce the study.
- The frozen readiness bundle predates the Phase A code move; rebuild it when the
  study next runs so its provenance hashes match the tracked sources.

See `files.md` for the file inventory.
