import json
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, ".auto-trainer/_harness")
import data as D
import model as M

BASE = {
    "train_path": "train.csv",
    "test_path": "test.csv",
    "target_column": "Transported",
    "id_column": "PassengerId",
    "prediction_column": "Transported",
    "features_path": ".auto-trainer/features.py",
    "metric": "accuracy",
    "seed": 42,
    "val_size": 0.2,
}

CONFIGS = [
    {"model_type": "logistic_regression", "hyperparameters": {"C": 1.0}},
    {"model_type": "ridge_classifier", "hyperparameters": {"alpha": 1.0}},
    {"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 3}},
    {"model_type": "random_forest", "hyperparameters": {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 2}},
    {"model_type": "gradient_boosting", "hyperparameters": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3}},
    {"model_type": "knn", "hyperparameters": {"n_neighbors": 15}},
    {"model_type": "knn", "hyperparameters": {"n_neighbors": 30}},
    {"model_type": "mlp", "hyperparameters": {"hidden_layer_sizes": [64, 32], "max_iter": 400}},
    {"model_type": "linear_svc", "hyperparameters": {"C": 0.5}},
]

for c in CONFIGS:
    cfg = {**BASE, **c}
    t0 = time.time()
    Xtr, Xval, ytr, yval = D.load_split(cfg)
    pre = D.build_preprocessor(M.needs_scaling(cfg["model_type"]))
    est = M.build_model(cfg)
    pipe = Pipeline([("pre", pre), ("model", est)])
    pipe.fit(Xtr, ytr)
    tr_acc = accuracy_score(ytr, pipe.predict(Xtr))
    val_pred = pipe.predict(Xval)
    val_acc = accuracy_score(yval, val_pred)
    proba = M.predict_proba_pos(pipe, Xval)
    n_features = pipe.named_steps["pre"].transform(Xval[:1]).shape[1]
    params = M.count_trainable_params(pipe.named_steps["model"], n_features, len(ytr))
    dt = time.time() - t0
    print(json.dumps({
        "model": cfg["model_type"], "hp": cfg["hyperparameters"],
        "train_acc": round(float(tr_acc), 4), "val_acc": round(float(val_acc), 4),
        "gap": round(float((tr_acc - val_acc) / tr_acc), 4),
        "params": params, "proba_range": [round(float(proba.min()), 3), round(float(proba.max()), 3)],
        "secs": round(dt, 1),
    }))
