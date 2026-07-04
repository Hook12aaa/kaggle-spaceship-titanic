# Auto-Train Final Report — Spaceship Titanic

**Status: CONVERGED → DONE.** Winner: `exp_ensemble` (Caruana blend), validation accuracy **0.80621**, beating the best single model `tree_d1` (0.80276) and the baseline `exp_000` (0.79758). Every number below traces to an executed script (`compute_pareto.py`, `check_class_exhaustion.py`, `check_cross_class_coverage.py`, `verify_merkle_chain.py`, `caruana_ensemble.py`, and each worktree's `eval.py`).

---

## Section 1 — Objective Recap

| Field | Value |
|---|---|
| Dataset (train) | `./train.csv` (8693 rows) |
| Dataset (test) | `./test.csv` (4277 rows) |
| Target column | `Transported` (boolean) |
| Competition metric | `accuracy` |
| Metric direction | `maximize` |
| Submission id column | `PassengerId` |
| Submission prediction column | `Transported` |
| Constraints | max_iterations 15, architecture_classes_minimum 3 |

Source: `objective.yaml`.

## Section 2 — Data Quality Summary

All 8 universal checks PASS (`.auto-trainer/data-quality-report.json`, status `PASS`):

- **Shape & size:** 8693 samples, 11 raw modelling features, 790.3 samples/feature (≥10 → pass).
- **Data types:** 0 coercion failures.
- **Missing values:** all 11 feature columns carry ~2.0–2.5% missing (low band); imputed (median for numeric, most-frequent for categorical) inside each worktree's `data.py`. No column exceeded the drop threshold.
- **Target:** `Transported` boolean, 0% missing, near-balanced (True 4378 / False 4315).
- **Duplicates:** 0 exact duplicate rows.
- **Distributions:** five spending columns are right-skewed (|skew|>2) → log1p features engineered.
- **Correlations:** no redundant pairs (|r|>0.95), no leakage suspects.
- **Outliers:** 19 rows (0.22%) flagged by MAD modified-Z; retained.

A domain-research agent produced `domain_context` (inferred domain: synthetic interstellar passenger transport), column semantics, and known relationships (group encoding in PassengerId, Cabin = deck/num/side, CryoSleep ⇒ zero amenity spend, group/surname outcome correlation). The dataset is synthetic (Kaggle Playground) → no external source to join; significance thresholds were calibrated to the synthetic domain (2% relative).

## Section 3 — Exploration Summary

| Metric | Value |
|---|---|
| Total nodes | 8 (1 baseline + 6 variants + 1 ensemble) |
| Architecture classes explored | 3 (linear, tree_based, knn) |
| Max tree depth reached | 3 |
| Exploration rounds | 3 |
| Engineered features | 38 (`feature-manifest.json`) |
| Convergence | two-tier (within-class exhaustion + cross-class coverage), reached at Round 3 |

Feature engineering merged proposals from parallel domain + structural research agents into `features.py` (verified to run on train and test, row counts preserved) and locked `feature-manifest.json` with input-data and code hashes.

## Section 4 — Pareto Front Evolution

Pareto front computed over (accuracy ↑, trainable_params ↓) by `compute_pareto.py`. `pareto_history` snapshots:

| Round | Pareto front (exp_ids) |
|---|---|
| Baseline | `exp_000` |
| Round 1 | `exp_000`, `tree_d1` |
| Round 2 | `lin_d2`, `tree_d1`, `tree_d2` |
| Round 3 | `lin_d2`, `tree_d1`, `tree_d2` (unchanged → stable) |

The front stabilized between Round 2 and Round 3 (identical snapshots), satisfying `pareto_stability_rounds = 2`. `exp_000` left the front when `lin_d2` (same 62 params, higher accuracy) dominated it; `tree_d3` and `knn_d1` never entered (dominated). The final experiment Pareto front is `tree_d1`, `lin_d2`, `tree_d2`. The Caruana ensemble (`exp_ensemble`) is built post-convergence from these front members and is recorded as the recommended winner separately; it is not an exploration node and does not re-open the convergence loop.

## Section 5 — Winner Analysis

**Winner: `exp_ensemble`** — Caruana greedy blend selected by `caruana_ensemble.py`.

| Field | Value |
|---|---|
| Members & weights | `tree_d1` (RandomForest) 0.5, `lin_d2` (LogisticRegression) 0.5 |
| Architecture class | ensemble (spans tree_based + linear → 2 distinct classes) |
| Validation accuracy | **0.80621** |
| Best single (tree_d1) | 0.80276 |
| Baseline (exp_000) | 0.79758 |
| Absolute gain vs best single | +0.00345 |
| Absolute gain vs baseline | +0.00863 (relative +1.08%) |
| Train accuracy / gap_ratio | 0.83333 / 0.0326 (≤0.10, no overfitting) |
| Parent (Merkle) | `tree_d1` (best single member) |
| node sha | `3ae7259730c05dc3…` |

`caruana_ensemble.py` greedy forward selection (with replacement) settled on the two members at equal weight; the blend's `beats_best_single` flag is `true`. Ensemble gain comes from architectural diversity — a high-variance tree and a low-variance linear model make different errors.

## Section 6 — Runner-up Comparison

Top models by validation accuracy (all from executed `eval.py` runs):

| exp_id | class | depth | accuracy | params | eval verdict | on experiment front |
|---|---|---|---|---|---|---|
| **exp_ensemble** | ensemble | — | **0.80621** | 126020 | INCONCLUSIVE* | winner (post-convergence blend, not an exploration node) |
| tree_d1 | tree_based | 1 | 0.80276 | 125958 | INCONCLUSIVE* | yes |
| tree_d2 | tree_based | 2 | 0.80046 | 2994 | INCONCLUSIVE* | yes (best accuracy-per-param) |
| tree_d3 | tree_based | 3 | 0.79931 | 54490 | INCONCLUSIVE* | no (dominated by tree_d2) |
| lin_d2 | linear | 2 | 0.79873 | 62 | INCONCLUSIVE* | yes (smallest model) |
| exp_000 | linear | 0 | 0.79758 | 62 | BASELINE | no |
| lin_d1 | linear | 1 | 0.79356 | 62 | INCONCLUSIVE* | no |
| knn_d1 | knn | 1 | 0.77516 | 424194 | INCONCLUSIVE* | no (dominated) |

*Every variant passed the evaluator's hard layers (data-validation, overfitting, forensics). "INCONCLUSIVE" means the accuracy gain over the baseline did not clear the synthetic-domain 2% relative-significance bar — an honest reflection that on this dataset the realistic model ceiling (~0.81) sits close to a strong linear baseline. The winner is selected by Pareto position and absolute accuracy, with the ensemble providing the best measured score.

Why `exp_ensemble` over `tree_d1`: identical-cost (≈126k params) but +0.00345 higher validation accuracy, reproduced and cross-checked independently. Why over `tree_d2`: `tree_d2` (GradientBoosting, 2994 params) is the Pareto champion for accuracy-per-parameter (0.80046 at 1/42 the size) and is the recommended pick if model size is the priority; the headline winner optimizes raw accuracy per the objective metric.

## Section 7 — Two-Tier Convergence Evidence

**Tier 1 — within-class exhaustion** (`check_class_exhaustion.py`, final):

| class | status | best acc | depth | reason exhausted |
|---|---|---|---|---|
| linear | EXHAUSTED | 0.79873 | 2 | diminishing returns: depth-bests 0.79758 → 0.79356 → 0.79873, both step changes <1% |
| tree_based | EXHAUSTED | 0.80276 | 3 | diminishing returns: depth-bests 0.80276 → 0.80046 → 0.79931, both step changes <1% |
| knn | EXHAUSTED | 0.77516 | 1 | Pareto-dominated by linear (0.79873 ≥ 0.77516 at 62 ≤ 424194 params) |

**Tier 2 — cross-class coverage** (`check_cross_class_coverage.py`, final):

```
{"global_status": "CONVERGED", "explored_classes": 3, "exploring_classes": [],
 "pareto_stable": true, "reasons": []}
```

All three conditions hold: (a) explored classes 3 ≥ minimum 3; (b) zero classes still EXPLORING; (c) Pareto front unchanged for 2 consecutive rounds.

## Section 8 — Integrity Summary

- **Merkle chain:** `verify_merkle_chain.py` → `{"valid": true, "nodes_checked": 8, "mismatches": []}`. Every node's `sha = SHA256(config_hash + parent_sha)` verified; lineage intact from `exp_000` (root) through the ensemble.
- **Independent winner verification:** a fresh reviewer subagent (no build/eval context) deleted the winner's `metrics.json`/`predictions.csv`, re-ran `run.sh` from scratch, and reproduced accuracy **0.80621**; it then cross-checked the blend a second, independent way (averaging the two members' saved validation probabilities) → **0.8062104657849338**, identical. Member validation splits confirmed bit-for-bit aligned (`np.array_equal` True). Submission confirmed 4277 boolean rows. **No discrepancies.**
- **Evaluation discipline:** every metric originated from an executed manifest extraction command; all Pareto, exhaustion, coverage, and significance comparisons were computed by scripts, never by reading numbers.

## Section 9 — Reproducibility

- **Winner worktree:** `.auto-trainer/worktrees/exp_ensemble/`
- **Members:** `.auto-trainer/worktrees/tree_d1/` (RandomForest n_estimators=200, max_depth=12, min_samples_leaf=3) and `.auto-trainer/worktrees/lin_d2/` (LogisticRegression C=10), blended 0.5/0.5 via `ensemble_config.json`.
- **Environment:** `.venv` (Python 3.12, scikit-learn 1.9.0, pandas 3.0.3, numpy 2.5.0).
- **Reproduce the winner:**
  ```bash
  bash .auto-trainer/worktrees/tree_d1/run.sh      # member 1
  bash .auto-trainer/worktrees/lin_d2/run.sh       # member 2
  python3 .auto-trainer/scripts/caruana_ensemble.py .auto-trainer/experiment-tree.json accuracy maximize .auto-trainer/worktrees/exp_ensemble/ensemble_config.json
  bash .auto-trainer/worktrees/exp_ensemble/run.sh # blend + submission
  ```
- **Shared data pipeline:** `.auto-trainer/features.py` (38 engineered features), consumed by each worktree's `data.py`; feature lineage in `.auto-trainer/feature-manifest.json`.
- **Submission:** `.auto-trainer/submission.csv` (4277 rows, `PassengerId,Transported`, booleans, predicted-True fraction 0.538), generated from the winner's `eval` on `test.csv`.

---

*Generated by the auto-model-trainer orchestrator. Permanent audit trail: `.auto-trainer/experiment-tree.json`.*
