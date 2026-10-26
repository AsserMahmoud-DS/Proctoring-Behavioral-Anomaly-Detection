# Experiment 1 — Selected models and conclusions

The four locked final variants, why they were selected, what the study
concludes, its limitations, and the provenance needed to reproduce or cite it.
Method and complete result tables: `03_experiments_and_models.md`.

## 1. Selection criteria applied

- One winner per family/recipe/augmentation cell by **validation ROC-AUC**
  (first declared candidate on ties); no threshold metric overrides it.
- Among each family's recipe winners, the per-condition overall finalist is the
  highest validation ROC-AUC (first tie) — used for the ensemble and repeats.
- Ensemble: fitted family winners reused, robust score normalization frozen on
  the 197 pure-normal validation windows, IF weight chosen on combined
  validation ROC-AUC.
- Operating threshold per selected approach: maximize validation recall subject
  to validation precision ≥ 0.5, F1 fallback if the floor is unmet; classification
  is `score >= threshold`.
- Test is evaluated once at the frozen configurations, weights, and thresholds;
  test metrics never change the selection.

## 2. The four final variants

All four are augmentation-on. Each entry lists the exact selected parameters,
the full-precision threshold, validation values, and the single test evaluation.
Figures are in `figures/`.

### 2.1 `if:IF-raw:aug_on`

- **Parameters:** `{"max_samples": 256, "max_features": 1.0, "n_estimators": 100, "contamination": "auto"}`
- **Validation:** ROC-AUC 0.7847, AP 0.6273; at the selected threshold
  (precision 0.50, recall 0.6707) — no F1 fallback.
- **Threshold:** −0.06215092372224379
- **Test:** ROC-AUC 0.7138, AP 0.7500, precision 0.7391, recall 0.6634,
  F1 0.6992, FPR 0.2981; TP/FP/TN/FN = 136/48/113/69.
- **Why:** best IF validation winner (vs IF-log/on 0.7818); raw input preserves
  extreme-value separation, and augmentation gave the largest matched IF gain
  (12/12 pairs positive, median ΔROC +0.0229).
- **Figures:** `figures/roc_if_IF-raw_aug_on.png`, `pr_if_IF-raw_aug_on.png`,
  `confusion_if_IF-raw_aug_on.png`.

### 2.2 `ocsvm:SVM-base:aug_on`

- **Parameters:** `{"nu": 0.01, "gamma": 0.04359031420070035, "kernel": "rbf", "tol": 0.001, "max_iter": -1}`
- **Validation:** ROC-AUC 0.7664, AP 0.5134; threshold at precision 0.50,
  recall 0.6098 — no fallback.
- **Threshold:** −0.019054717746805228
- **Test:** ROC-AUC 0.7323, AP 0.7461, precision 0.7798, recall 0.6390,
  F1 0.7024, FPR 0.2298; TP/FP/TN/FN = 131/37/124/74.
- **Why:** best OCSVM validation winner (compression/YJ/quantile variants did
  not improve ranking). It also has the highest primary **test** ROC-AUC
  (0.7323); that is reported for completeness and does not replace the
  validation-selected overall winner.
- **Figures:** `figures/roc_ocsvm_SVM-base_aug_on.png`,
  `pr_ocsvm_SVM-base_aug_on.png`, `confusion_ocsvm_SVM-base_aug_on.png`.

### 2.3 `ae:AE-log:aug_on`

- **Parameters:** `{"num_layers": 2, "hidden_dim": 32, "batch_size": 32, "lr": 0.001}`
- **Validation:** ROC-AUC 0.7445, AP 0.5717; threshold at precision 0.50,
  recall 0.5732 — no fallback.
- **Threshold:** 0.3605460226535797
- **Test:** ROC-AUC 0.6314, AP 0.6607, precision 0.7099, recall 0.4537,
  F1 0.5536, FPR 0.2360; TP/FP/TN/FN = 93/38/123/112.
- **Why:** best AE validation winner (vs AE-base/on 0.7419 and AE-YJ/on 0.6546);
  the selective-log recipe also improved most consistently under augmentation
  (14/14 matched pairs positive).
- **Figures:** `figures/roc_ae_AE-log_aug_on.png`,
  `figures/pr_ae_AE-log_aug_on.png`, `figures/confusion_ae_AE-log_aug_on.png`.

### 2.4 `ensemble:IF-raw+SVM-base:aug_on`

- **Components:** fitted IF-raw/on and SVM-base/on winners (no refit);
  `if_weight = 0.7`, `ocsvm_weight = 0.30000000000000004` (floating-point
  representation of 0.3), scores normalized by the frozen normal-validation
  RobustScaler.
- **Validation:** ROC-AUC 0.7946 (best of all 20 approaches), AP 0.6158;
  threshold at precision 0.50, recall 0.6585 — no fallback.
- **Threshold:** 0.6587707622593663
- **Test:** ROC-AUC 0.7191, AP 0.7587, precision 0.7670, recall 0.6585,
  F1 0.7087, FPR 0.2547; TP/FP/TN/FN = 135/41/120/70.
- **Why:** the overall validation-selected winner. Combining the two best
  family rankings with 70% weight on IF improves validation ranking over either
  component; on test it improves AP/F1 over IF alone while SVM-base alone still
  has the higher test ROC.
- **Figures:** `figures/roc_ensemble_IF-raw_SVM-base_aug_on.png`,
  `figures/pr_ensemble_IF-raw_SVM-base_aug_on.png`,
  `figures/confusion_ensemble_IF-raw_SVM-base_aug_on.png`.
- Combined comparisons:
  `figures/roc_comparison.png`, `figures/pr_comparison.png`.

### 2.5 Summary

| Variant | Val ROC | Val AP | Test ROC | Test AP | Test F1 | Threshold |
|---|---|---|---|---|---|---|
| if:IF-raw:aug_on | 0.7847 | 0.6273 | 0.7138 | 0.7500 | 0.6992 | −0.06215092372224379 |
| ocsvm:SVM-base:aug_on | 0.7664 | 0.5134 | 0.7323 | 0.7461 | 0.7024 | −0.019054717746805228 |
| ae:AE-log:aug_on | 0.7445 | 0.5717 | 0.6314 | 0.6607 | 0.5536 | 0.3605460226535797 |
| ensemble:IF-raw+SVM-base:aug_on | 0.7946 | 0.6158 | 0.7191 | 0.7587 | 0.7087 | 0.6587707622593663 |

## 3. Why these four

- They are the augmentation-on per-family validation winners plus the overall
  validation-selected ensemble; every other primary row is a different recipe or
  condition of the same families (`03_experiments_and_models.md` §7.1).
- Augmentation-on is where the strongest validation rankings occur for each
  family in this run, and 86/102 matched pairs improve validation ROC.
- Threshold-selected alternatives inside the same families differ mainly in
  operating point, not in ranking: e.g., SVM-base/off has higher precision and
  much lower recall (0.8583 / 0.5024) than SVM-base/on (0.7798 / 0.6390).
- AE-YJ is excluded despite satisfying the precision floor because its
  validation decision used two predictions (one anomalous, one normal) and the
  transferred threshold flags every test window normal — a degenerate operating
  point, not a practical detector.

## 4. Conclusions

- **Feasibility:** normal-only trained detectors rank annotated anomalous
  windows better than chance on this split; the validation-selected classical
  approaches and ensemble land around test ROC 0.70–0.73, with the AE lower
  (~0.63).
- **Preprocessing matters:** SVM-base beats its transformed variants on
  validation; IF's preferred recipe flips between log (off) and raw (on); the
  AE's best recipe depends on augmentation state.
- **Augmentation:** usually improves validation ranking within matched
  configurations (86/102 pairs), with small or inconsistent operational/test
  changes and, for the AE, additional optimizer updates under the fixed epoch
  budget.
- **Depth:** two-layer AE gains are small (9/12 pairs positive, median ≤ +0.008
  ROC) and do not change the AE's weaker standing.
- **Seeds:** IF validation varies ~0.01–0.02 and AE ~0.01 across seeds 42/123/2026;
  AE's weaker ranking is consistent rather than one initialization.
- **Ranking vs threshold:** a good ROC ranking does not guarantee a useful
  transferred operating point (AE-YJ, §7.6 of file 03).
- **Scope:** these are descriptive proof-of-concept findings, not statistical
  significance, deployment safety, or misconduct determination claims.

## 5. Limitations

1. **Few independent sessions:** four normal-validation and five mixed-validation
   sessions (ten contributing test sessions); overlapping windows and synthetic
   copies are not independent samples.
2. **Student overlap:** file-level disjointness does not prove unseen-student
   performance.
3. **Partition composition:** validation mixes pure-normal and mixed sessions;
   test is mixed-only, with different prevalence (20.87% vs 56.01%).
4. **Selection optimism:** grids, recipes, weights, checkpoints, and thresholds
   all reuse the same small validation partition; validation maxima are not
   unbiased generalization estimates.
5. **Exploration context:** earlier validation ablations informed the design;
   the study is exploratory rather than a previously untouched confirmatory
   holdout.
6. **Whole-recipe comparisons:** nonlinear steps, count treatment, and scaling
   are not independently factorialized; a winning recipe does not isolate one
   beneficial transformation.
7. **Unequal search budgets:** 48 IF / 72 OCSVM / 84 AE main fits; more search
   opportunity can raise validation maxima.
8. **AE compute confound:** augmentation adds optimizer updates, not only data
   diversity; 77/84 AE fits reached the 100-epoch cap, so AE performance is
   reported under the declared budget.
9. **Window/label definition:** event-count windows have variable real duration;
   metrics assess the ≥50%-annotated parent label, not arbitrary real-time
   latency.
10. **Feature proxies:** zero aggregates are ambiguous (no data vs genuine zero),
    lone clicks are undercounted, and several feature names are proxies
    (`01_features.md` §6).
11. **No uncertainty estimates:** no session-bootstrap interval, repeated split,
    nested validation, or unseen-student evaluation is included.
12. **Anomaly is not misconduct:** legitimate clipboard use, navigation,
    irregular movement, accessibility needs, or device differences can produce
    similar signals.

## 6. Provenance and reproduction

See the [reproduction guide](REPRODUCING.md) for private data requirements,
checksum verification, and portability limits before executing any commands.

Original inputs and outputs selected for preservation in this snapshot
(paths are relative to the repository root unless noted):

| Artifact | Role |
|---|---|
| `study/artifacts/phase_01_audit/` | original-training audit (split, feature, transform, kernel diagnostics) |
| `study/artifacts/phase_03_study/` | prepared bundle: paired views, frozen processors, gamma centers, manifests |
| `study/artifacts/phase_04_study/` | 204 candidate rows, comparison tables, winners, reporting manifest |
| `study/artifacts/phase_04_study/phase_05_seed_sensitivity/` | 8 repeat artifacts |
| `study/artifacts/phase_04_study/phase_06_reporting/` | 28 test rows, dataset summary, grouped figures |
| `tables/` | immutable copies of the report tables and manifest used here |
| `figures/` | final-variant and EDA figures used here |

Checksums (SHA-256) of the key artifacts at report time:

- `phase_03_study/manifest.json`:
  `b58959842cb2ae41b9a7f39c9262d196de2fea9adb5ff621bc8bfc5e31a39c52`
- `phase_04_study/reporting_manifest.json`:
  `67bbee56063c5bb2316d27f5fededeeace52d46cb8df48afc77747ee827fecb4`
- `phase_04_study/candidates.csv`:
  `531950816bc5a10f55e22b5126cba22a81531ddef915d43a714a36e964540602`
- `phase_06_reporting/test_results.csv`:
  `e40f8b61070387b33a4e758be35ab7b06f3a38c0b4746a5179cace81bfdab86d`

Recorded package versions (from the prepared manifest): NumPy 2.4.6, pandas
3.0.3, scikit-learn 1.9.0. The manifest does not record the Python/PyTorch/
hardware environment, so byte-identical reruns are not guaranteed. PyTorch (CPU)
is required for AE training and AE artifact loading.

Reproduction sequence (with the raw sessions and audit prerequisites present):

```bash
uv run python -m cheatdetect.experiments.experiment_1.audit
uv run python -m cheatdetect.experiments --phase 3
uv run python -m cheatdetect.experiments --phase 4 --run-dir study/artifacts/experiment_1_reproduction
uv run python -m cheatdetect.experiments --phase 5 --run-dir study/artifacts/experiment_1_reproduction
uv run python -m cheatdetect.experiments --phase 6 --run-dir study/artifacts/experiment_1_reproduction
```

Phases 4–6 with a fresh run directory avoid overwriting the original result set;
the audit and Phase 3 use their configured default directories, so preserve the
originals or reproduce in a separate checkout. Regression tests cover
original-only fitting/audit reads, paired-noise identity, schema/recipe
correctness, ordered selection, model controls and convergence, repeats, CLI
routing, comparison tables, and reporting — test success supports implementation
checks, not statistical validity.

## 7. Final statement

**Experiment 1 demonstrates a modest learnable anomaly signal under normal-only
training with session-file-held-out evaluation. Classical models are more
convincing than the tested AE. Small coordinate augmentation usually improves
validation ranking but not every operating metric, and its AE effect includes
extra optimization work. Bounded depth gains are small; seed repeats show AE
weakness is consistent rather than unlucky. The overall winner remains the
augmentation-on IF-raw/SVM-base ensemble selected by validation ROC-AUC
(test ROC-AUC 0.7191, AP 0.7587, precision 0.7670, recall 0.6585, F1 0.7087),
with SVM-base augmentation-on reported as the strongest primary test ranking
(0.7323) without rewriting the selection. The study is exploratory,
small-session, and threshold-sensitive: it establishes feasibility, not reliable
misconduct determination or unseen-student performance.**
