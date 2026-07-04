import json
import os
import sys

import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C
import data as D
import model as M

cfg = C.CONFIG


def main():
    Xtr, Xval, ytr, yval = D.load_split(cfg)
    pre = D.build_preprocessor(M.needs_scaling(cfg["model_type"]))
    est = M.build_model(cfg)
    pipe = Pipeline([("pre", pre), ("model", est)])
    pipe.fit(Xtr, ytr)

    tr_acc = float(accuracy_score(ytr, pipe.predict(Xtr)))
    val_acc = float(accuracy_score(yval, pipe.predict(Xval)))

    joblib.dump(pipe, os.path.join(HERE, "model.joblib"))
    log = {"epoch": 1, "train_accuracy": round(tr_acc, 6), "val_accuracy": round(val_acc, 6)}
    with open(os.path.join(HERE, "train_log.jsonl"), "w") as f:
        f.write(json.dumps(log) + "\n")
    print(json.dumps(log))


if __name__ == "__main__":
    main()
