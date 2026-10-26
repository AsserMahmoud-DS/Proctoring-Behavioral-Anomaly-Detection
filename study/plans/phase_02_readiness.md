# Study Phase 2 — Data and preprocessing readiness

> Historical document: This records the earlier readiness snapshot, including module paths that later changed. Its bundle is preserved as historical evidence; the current runner uses stage 3. See the [study inventory](../README.md) and [final report](../../reports/experiment1/README.md).

Implementation complete; synthetic verification passes. The user successfully
executed the readiness notebook, and the saved real-data snapshot was independently
verified. Reusable components now live in tracked main-workflow modules/tests;
study orchestration imports them. No detector fits, validation/test metric inspection,
augmentation EDA, production-cache writes, commits, or pushes.

## Delivered

- `study/data.py` builds fresh raw-data representations using the frozen Phase 1
  split. It checks original training hashes against that audit, rejects anomalous
  normal-session inputs, and never invokes the legacy cache/training pipeline.
  Files with no complete 50-event window are explicitly reported and skipped;
  an entirely unusable required partition fails. No padding, geometry changes,
  or removal from the frozen split.
- Parent identities are `(session_file, parent_start_event, copy_index)`, with
  copy 0 original and copies 1/2 synthetic. Input-file order follows the frozen
  split; rows follow event-window order. Synthetic identity suffixes do not create
  independent sessions.
- Each noisy parent is generated once. Both its aggregate features and its nine
  micro-feature vectors come from that exact raw realization, with coordinate
  bounds and schema/domain checks. No separately seeded flat/sequence augmenters.
- `src/cheatdetect/data/preprocessing.py` implements the reusable fitted operations;
  `study/preprocessing.py` only provides old snapshot import compatibility.
  The study chooses all nine fixed recipes: IF-raw/log,
  SVM-base/log/YJ/quantile, AE-base/log/YJ. Fit each processor once on original
  normal training, then only transform synthetic and held-out inputs.
- Numerical nonfinite values are imputed with persisted original-training
  medians. An all-missing original column fails rather than inventing a median.
  Direction must be a valid finite code 0–8. Source counts/ratios and signs are
  checked; no clipping of signed jerk or dropping of rare/constant features.
- YJ/quantile bypass constant original-training continuous columns. Other
  constants follow scaler scale=1; later changes remain representable. Quantile
  landmarks and YJ lambdas are persisted in the fitted joblib objects.
- The flat processors use original parent rows. AE processors use original
  micro-features flattened across all parents/timesteps, then restore `(N,9,33)`.
  All outputs share the same fixed 33-column order.
- Augmentation off/on accessors reuse the same frozen processors, gamma centres,
  and held-out matrices. On adds synthetic rows to the originals; off never does.
  OCSVM gamma is derived from transformed originals only and saved numerically.
- Readiness manifests cover raw-content hashes, ordered splits, geometry,
  augmentation settings/seed, source-code hashes, and numerical-library versions.
  Cache reuse requires a freshly calculated matching manifest before loading
  trusted local joblib; a stale bundle or mismatched bundle/manifest fails.
- Artifacts can only be saved inside `study/artifacts/`. Production cache/model
  directories cannot be passed as readiness output destinations.
- API classification now uses `score >= threshold`, matching tuning, detector
  prediction, and evaluation. Normal service fixtures use below-threshold scores;
  an explicit regression covers equality across all four paths.

## User execution and reuse

Execute the paired readiness notebook in the project environment. It prepares
and saves `study/artifacts/phase_02_readiness/`:

- `prepared.joblib`: original/synthetic/held-out paired views, fitted processors,
  transformed matrices, labels/identities, frozen gamma centres, and provenance.
- `manifest.json`: reproducibility/cache-validity contract.
- `preprocessing.json`: output order, original fit-row counts, and gamma centres.
- `session_readiness.csv`: every assigned file's raw-row count, cleaned-event
  count, parent-window count, inclusion flag, and exclusion reason. The same
  report is included in `prepared.joblib`; the notebook reports assigned versus
  contributing session counts separately and lists excluded files.

This is preprocessing and automated input validity checking, not another EDA or
model-selection study. No detector is trained and no held-out metric is computed.
Future Phase 3/4 callers should use `input_manifest` with the frozen split and
current directories, then `load_prepared`; do not pass an old saved manifest as
the expectation. If manifest checks fail, regenerate fresh rather than bypassing
the check. Only load joblib written by this trusted local study.

## Verification coverage

Latest integration verification: 152 study/production tests pass with API tests excluded from
the sandboxed run; all four API tests pass separately outside the sandbox.
Notebook pairs, Python compilation, ignored-file placement, and diff whitespace
checks also pass. The saved real-data snapshot also passes fresh input/source
manifest checks, serialization, frozen split/parent/copy identity checks, exact
re-transformation equality, finite output checks, original-only fit counts,
unchanged fitted-state checks, and original-derived gamma-centre checks.

Observed original windows: train 662, normal validation 197, mixed validation 196,
test 366. Synthetic training contributes 1,324 additional windows, giving 1,986
for augmentation on. Original contributing sessions are 15 / 4 / 5 / 10,
respectively; test has 12 assigned files, including two excluded summaries.

Synthetic tests verify exact selective logs and signed values, excluded scaling
blocks, fixed categories/column order, constant bypass and unseen changes,
train-only medians/scales, AE flatten/restore, serialization, gamma calculation,
shared realized noise, copy identities/labels, reproducibility, source/configuration
invalidation, frozen training hash checks, output isolation, and unchanged fitted
state when held-out inputs or synthetic-copy counts change. Existing extraction,
sequence, detector, pipeline, metrics, and API/service tests provide regressions.

Additional regressions cover summary-only and short files beside usable inputs,
unchanged frozen split assignments, persisted exclusion reports, aligned remaining
parent/sequence/label rows, and an unusable partition failure for all four splits.

The user's first real-data run found two summary-only files assigned to mixed_test:
`mixed - El Farouk Omar.csv` and `youssef ahmed- mixed.csv`. Both have zero cleaned
events. The old per-file hard failure has been replaced with the explicit reporting
policy above. Rerun the readiness notebook to regenerate with this policy.

## Review gate

The real-data readiness run passes. Reusable schema, preprocessing, paired
extraction, and session loading are integrated under tracked data modules, with
tracked regression tests. Study recipe/configuration orchestration, diagnostics,
manifests, and reporting remain ignored. Existing training defaults and deployed
artifacts are intentionally unchanged; model-specific wiring belongs to Phase 3.
The readiness snapshot was migrated after exact provenance/input/transform and
shared-window extraction equivalence checks, without refitting or regeneration.
Phase 2 is complete and ready for review; no Phase 3 work has begun.
If a schema or
domain check fails, investigate extraction/input validity rather than consulting
validation/test results or silently changing recipes. Stop before Phase 3.
