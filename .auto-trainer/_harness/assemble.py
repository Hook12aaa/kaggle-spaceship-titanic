import hashlib
import json
import os
import shutil
import sys

ROOT = "/Users/hook/Documents/coding/python/kaggle/spaceship-titanic"
HARNESS = os.path.join(ROOT, ".auto-trainer", "_harness")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
SHARED = ["data.py", "model.py", "train.py", "eval.py", "preflight.py"]

OBJECTIVE_INTEGRITY = {
    "metric": "accuracy",
    "metric_direction": "maximize",
    "target_column": "Transported",
    "id_column": "PassengerId",
    "prediction_column": "Transported",
}
CONSTRAINTS_HASH = hashlib.sha256(
    json.dumps(OBJECTIVE_INTEGRITY, sort_keys=True).encode()
).hexdigest()


def config_py_text(config):
    payload = json.dumps(config)
    return "import json\n\nCONFIG = json.loads(r'''" + payload + "''')\n"


def assemble(exp_id, config, dest):
    os.makedirs(dest, exist_ok=True)
    for m in SHARED:
        shutil.copy(os.path.join(HARNESS, m), os.path.join(dest, m))

    with open(os.path.join(dest, "config.py"), "w") as f:
        f.write(config_py_text(config))

    manifest = {
        "metrics": [
            {
                "name": "accuracy",
                "type": "higher_better",
                "command": f"{VENV_PY} -c \"import json;print(json.load(open('metrics.json'))['accuracy'])\"",
            }
        ]
    }
    with open(os.path.join(dest, "metrics_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    constraints = {
        "max_epochs": 1,
        "timeout_seconds": 600,
        "memory_limit_mb": 4096,
        "constraints_hash": CONSTRAINTS_HASH,
        "objective_integrity": OBJECTIVE_INTEGRITY,
    }
    with open(os.path.join(dest, "constraints.lock"), "w") as f:
        json.dump(constraints, f, indent=2)

    run_sh = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        f"PY={VENV_PY}\n"
        '"$PY" preflight.py\n'
        '"$PY" train.py\n'
        '"$PY" eval.py\n'
    )
    rp = os.path.join(dest, "run.sh")
    with open(rp, "w") as f:
        f.write(run_sh)
    os.chmod(rp, 0o755)

    build_report = (
        f"# BUILD_REPORT: {exp_id}\n\n"
        f"hypothesis: {config.get('hypothesis', 'n/a')}\n\n"
        f"architecture_class: {config.get('architecture_class')}\n"
        f"model_type: {config['model_type']}\n"
        f"parent: {config.get('parent_id')}\n"
        f"resolved_config: {json.dumps(config)}\n\n"
        f"trainable_params: PENDING_AFTER_RUN\n"
    )
    with open(os.path.join(dest, "BUILD_REPORT.md"), "w") as f:
        f.write(build_report)

    return CONSTRAINTS_HASH


def config_hash(dest):
    with open(os.path.join(dest, "config.py"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if __name__ == "__main__":
    exp_id = sys.argv[1]
    config_path = sys.argv[2]
    dest = sys.argv[3]
    with open(config_path) as f:
        cfg = json.load(f)
    ch = assemble(exp_id, cfg, dest)
    print(json.dumps({"exp_id": exp_id, "dest": dest, "constraints_hash": ch, "config_hash": config_hash(dest)}))
