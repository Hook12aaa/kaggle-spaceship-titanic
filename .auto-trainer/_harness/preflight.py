import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C

cfg = C.CONFIG
errors = []

for key in ("train_path", "test_path", "target_column", "id_column", "prediction_column", "model_type", "seed", "val_size"):
    if key not in cfg:
        errors.append(f"missing config key: {key}")

for path_key in ("train_path", "test_path"):
    p = cfg.get(path_key)
    if p and not os.path.exists(p):
        errors.append(f"data file not found: {path_key}={p}")

fp = cfg.get("features_path")
if fp and not os.path.exists(fp):
    errors.append(f"features_path not found: {fp}")

lock = os.path.join(HERE, "constraints.lock")
if not os.path.exists(lock):
    errors.append("constraints.lock missing")

try:
    import data as _d
    import model as _m
except Exception as e:
    errors.append(f"import failure: {e}")

if errors:
    for e in errors:
        print(f"PREFLIGHT_FAIL: {e}", file=sys.stderr)
    sys.exit(1)

print("PREFLIGHT_OK")
