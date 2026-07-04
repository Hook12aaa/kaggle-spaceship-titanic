import hashlib
import json
import os
import sys

TREE = ".auto-trainer/experiment-tree.json"


def node_sha(config_hash, parent_sha):
    return hashlib.sha256(f"{config_hash}+{parent_sha}".encode()).hexdigest()


def main():
    spec = json.loads(sys.argv[1])
    with open(TREE) as f:
        tree = json.load(f)
    nodes = tree["nodes"]

    parent = spec.get("parent")
    if parent is None:
        parent_sha = "root"
        depth = 0
    else:
        parent_sha = nodes[parent]["sha"]
        depth = nodes[parent]["depth"] + 1

    sha = node_sha(spec["config_hash"], parent_sha)

    wt = spec["worktree_path"]
    metrics_path = os.path.join(wt, "metrics.json")
    with open(metrics_path) as f:
        m = json.load(f)

    node = {
        "exp_id": spec["exp_id"],
        "parent": parent,
        "depth": depth,
        "architecture_class": spec["architecture_class"],
        "model": spec["model"],
        "config_hash": spec["config_hash"],
        "sha": sha,
        "status": spec.get("status", "DONE"),
        "metrics": {"accuracy": m["accuracy"], "val_accuracy": m["val_accuracy"], "train_accuracy": m["train_accuracy"]},
        "trainable_params": int(m["trainable_params"]),
        "worktree_path": wt,
        "eval_verdict": spec.get("eval_verdict"),
        "review_verdict": spec.get("review_verdict"),
    }
    nodes[spec["exp_id"]] = node
    with open(TREE, "w") as f:
        json.dump(tree, f, indent=2)
    print(json.dumps({"exp_id": spec["exp_id"], "sha": sha, "depth": depth, "accuracy": m["accuracy"], "params": int(m["trainable_params"])}))


if __name__ == "__main__":
    main()
