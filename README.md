# Kaggle Spaceship Titanic -- Transported Prediction

This solution was created by [auto-model-trainer](https://github.com/Hook12aaa/auto-model-trainer), a Claude Code plugin for autonomous ML training. Given the competition objective, it explored 3 architecture classes (linear, tree_based, knn) across 8 nodes and finished on a Caruana ensemble blend. The domain research agents worked out the dataset structure on their own -- the PassengerId group encoding, the Cabin deck/num/side split, and the CryoSleep-to-spend relationship (passengers in cryo sleep can't spend on amenities).

## Competition
- **Task:** Binary classification, predict `Transported` (boolean)
- **Metric:** Accuracy
- **Data:** 8693 train rows, 4277 test rows
- **Link:** https://www.kaggle.com/competitions/spaceship-titanic

## Results
**Leaderboard: Public LB Score: 0.79798**

The winner was a Caruana greedy blend of a tree model and a linear model. Architectural diversity did the work -- a high-variance tree and a low-variance linear model make different errors, so blending them helped.

| Experiment | Class | Depth | CV Accuracy |
|---|---|---|---|
| exp_ensemble (Caruana blend) | ensemble | -- | **0.80621** |
| tree_d1 | tree_based | 1 | 0.80276 |
| tree_d2 | tree_based | 2 | 0.80046 |
| lin_d2 | linear | 2 | 0.79873 |
| exp_000 (baseline) | linear | 0 | 0.79758 |
| lin_d1 | linear | 1 | 0.79356 |
| knn_d1 | knn | 1 | 0.77516 |

- 38 engineered features
- 3 exploration rounds before two-tier convergence
- All three architecture classes explored to exhaustion, Pareto front stable for 2 consecutive rounds

## Project Structure
```
objective.yaml       # competition spec fed to the plugin
features.py           # 38 engineered features (group size, deck/side, spend logs, etc.)
final-report.md       # full evidence trail: every number traces to an executed script
submission.csv        # winner's predictions on the test set
.auto-trainer/        # experiment tree, worktrees, convergence scripts, feature manifest
```

## Usage
Created by the [auto-model-trainer](https://github.com/Hook12aaa/auto-model-trainer) plugin for Claude Code. Given an objective YAML, it handles data validation, feature engineering, baseline creation, experiment tree exploration, ensembling, convergence detection, and the final report -- autonomously.

```
/auto-train objective.yaml
```

Plugin: https://github.com/Hook12aaa/auto-model-trainer
