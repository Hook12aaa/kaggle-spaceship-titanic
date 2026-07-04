import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = "/Users/hook/Documents/coding/python/kaggle/spaceship-titanic"
HARNESS = os.path.join(ROOT, ".auto-trainer", "_harness")
ENSEMBLE_DIR = os.path.join(ROOT, ".auto-trainer", "worktrees", "exp_ensemble")
sys.path.insert(0, HARNESS)
import data as D
import model as M

BASE_CFG = {
    "train_path": os.path.join(ROOT, "train.csv"),
    "test_path": os.path.join(ROOT, "test.csv"),
    "target_column": "Transported",
    "id_column": "PassengerId",
    "prediction_column": "Transported",
    "features_path": os.path.join(ROOT, ".auto-trainer", "features.py"),
    "seed": 42,
    "val_size": 0.2,
}


def member_cfg(exp_id):
    with open(os.path.join(ROOT, ".auto-trainer", "worktrees", exp_id, "_cfg.json")) as f:
        return json.load(f)


def main():
    with open(os.path.join(ENSEMBLE_DIR, "ensemble_config.json")) as f:
        ens = json.load(f)
    members = ens["selected_models"]

    Xtr, Xval, ytr, yval = D.load_split(BASE_CFG)
    X_test, ids = D.load_test(BASE_CFG)

    wsum = sum(m["weight"] for m in members)
    blend_val = np.zeros(len(yval))
    blend_train = np.zeros(len(ytr))
    blend_test = np.zeros(len(ids))
    total_params = 0

    for m in members:
        eid, w = m["exp_id"], m["weight"]
        pipe = joblib.load(os.path.join(ROOT, ".auto-trainer", "worktrees", eid, "model.joblib"))
        blend_val += w * M.predict_proba_pos(pipe, Xval)
        blend_train += w * M.predict_proba_pos(pipe, Xtr)
        blend_test += w * M.predict_proba_pos(pipe, X_test)
        nf = pipe.named_steps["pre"].transform(Xval[:1]).shape[1]
        total_params += M.count_trainable_params(pipe.named_steps["model"], nf, len(ytr))

    blend_val /= wsum
    blend_train /= wsum
    blend_test /= wsum

    val_acc = float(accuracy_score(yval, (blend_val >= 0.5).astype(int)))
    train_acc = float(accuracy_score(ytr, (blend_train >= 0.5).astype(int)))

    np.save(os.path.join(ENSEMBLE_DIR, "val_predictions.npy"), blend_val.astype(float))
    np.save(os.path.join(ENSEMBLE_DIR, "val_labels.npy"), np.asarray(yval, dtype=float))

    metrics = {
        "accuracy": round(val_acc, 6),
        "val_accuracy": round(val_acc, 6),
        "train_accuracy": round(train_acc, 6),
        "trainable_params": int(total_params),
        "n_val": int(len(yval)),
        "n_train": int(len(ytr)),
        "members": members,
    }
    with open(os.path.join(ENSEMBLE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    test_pred = (blend_test >= 0.5).astype(bool)
    sub = pd.DataFrame({"PassengerId": ids, "Transported": test_pred})
    sub.to_csv(os.path.join(ENSEMBLE_DIR, "predictions.csv"), index=False)

    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
