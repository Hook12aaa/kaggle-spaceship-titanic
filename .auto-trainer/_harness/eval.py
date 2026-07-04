import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
import data as D
import model as M

cfg = C.CONFIG


def main():
    Xtr, Xval, ytr, yval = D.load_split(cfg)
    model_path = os.path.join(HERE, "model.joblib")
    pipe = joblib.load(model_path)

    val_pred = pipe.predict(Xval)
    val_acc = float(accuracy_score(yval, val_pred))
    tr_acc = float(accuracy_score(ytr, pipe.predict(Xtr)))
    val_proba = M.predict_proba_pos(pipe, Xval)

    np.save(os.path.join(HERE, "val_predictions.npy"), np.asarray(val_proba, dtype=float))
    np.save(os.path.join(HERE, "val_labels.npy"), np.asarray(yval, dtype=float))

    n_features = pipe.named_steps["pre"].transform(Xval[:1]).shape[1]
    params = M.count_trainable_params(pipe.named_steps["model"], n_features, len(ytr))

    metrics = {
        "accuracy": round(val_acc, 6),
        "val_accuracy": round(val_acc, 6),
        "train_accuracy": round(tr_acc, 6),
        "trainable_params": int(params),
        "n_val": int(len(yval)),
        "n_train": int(len(ytr)),
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    X_test, ids = D.load_test(cfg)
    test_pred = pipe.predict(X_test).astype(bool)
    sub = pd.DataFrame({cfg["id_column"]: ids, cfg["prediction_column"]: test_pred})
    sub.to_csv(os.path.join(HERE, "predictions.csv"), index=False)

    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
