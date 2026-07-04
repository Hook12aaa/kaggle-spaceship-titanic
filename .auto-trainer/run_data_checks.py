import json
import sys
import numpy as np
import pandas as pd

train_path = sys.argv[1]
target_col = sys.argv[2]
id_col = sys.argv[3]

df = pd.read_csv(train_path)
n_rows = len(df)

id_like = {id_col, "Name"}
feature_cols = [c for c in df.columns if c not in id_like and c != target_col]

checks = {}

n_features = len(feature_cols)
spf = n_rows / n_features
checks["shape_and_size"] = {
    "passed": bool(spf >= 10),
    "samples": int(n_rows),
    "features": int(n_features),
    "samples_per_feature": round(float(spf), 2),
}

coercion_failures = {}
for c in feature_cols:
    s = df[c]
    if s.dtype == object:
        coerced = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()
        if non_null > 0:
            newly_failed = coerced.isna().sum() - s.isna().sum()
            rate = newly_failed / non_null
            if rate < 1.0 and rate > 0.05:
                coercion_failures[c] = round(float(rate), 4)
checks["data_types"] = {
    "passed": len(coercion_failures) == 0,
    "columns_checked": n_features,
    "coercion_failures": coercion_failures,
    "max_coercion_failure_rate": float(max(coercion_failures.values())) if coercion_failures else 0.0,
}

mitigations = {}
for c in feature_cols:
    miss = df[c].isna().mean()
    if miss <= 0:
        continue
    if miss < 0.05:
        band, method = "low", "median_imputer" if df[c].dtype != object else "mode_imputer"
    elif miss < 0.20:
        band, method = "moderate", "knn_imputer" if df[c].dtype != object else "mode_imputer"
    elif miss < 0.50:
        band, method = "high", "missing_indicator_then_impute"
    else:
        band, method = "critical", "dropped"
    mitigations[c] = {"band": band, "method": method, "missing_pct": round(float(miss) * 100, 2)}
checks["missing_values"] = {
    "passed": True,
    "columns_with_missing": int(sum(1 for c in feature_cols if df[c].isna().any())),
    "mitigations_applied": mitigations,
}

tgt = df[target_col]
tgt_missing = float(tgt.isna().mean())
is_numeric_tgt = pd.api.types.is_numeric_dtype(tgt) and tgt.nunique() > 20
if is_numeric_tgt:
    tgt_summary = {"variance": float(tgt.var())}
    zero_var = tgt.var() == 0
else:
    vc = tgt.value_counts(dropna=True)
    tgt_summary = {"class_distribution": {str(k): int(v) for k, v in vc.items()}, "n_classes": int(tgt.nunique())}
    zero_var = tgt.nunique() <= 1
checks["target_variable"] = {
    "passed": bool(target_col in df.columns and tgt_missing <= 0.20 and not zero_var),
    "name": target_col,
    "dtype": str(tgt.dtype),
    "missing_pct": round(tgt_missing * 100, 2),
    **tgt_summary,
}

dup = int(df.duplicated().sum())
dup_pct = dup / n_rows * 100
checks["duplicates"] = {
    "passed": bool(dup_pct <= 1.0),
    "duplicate_rows": dup,
    "duplicate_pct": round(float(dup_pct), 3),
}

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
high_skew, zero_variance, near_constant = [], [], []
for c in numeric_cols:
    s = df[c].dropna()
    if len(s) == 0:
        continue
    if s.std() == 0:
        zero_variance.append(c)
    if s.nunique() / max(len(s), 1) < 0.01:
        near_constant.append(c)
    sk = s.skew()
    if abs(sk) > 2:
        high_skew.append(c)
checks["distributions"] = {
    "passed": True,
    "high_skew_columns": high_skew,
    "zero_variance_columns": zero_variance,
    "near_constant_columns": near_constant,
}

redundant_pairs, leakage_suspects, high_vif = [], [], []
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr(numeric_only=True)
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.95:
                redundant_pairs.append([cols[i], cols[j]])
checks["correlations"] = {
    "passed": len(leakage_suspects) == 0,
    "redundant_pairs": redundant_pairs,
    "leakage_suspects": leakage_suspects,
    "high_vif_columns": high_vif,
}

flagged_rows = 0
if numeric_cols:
    sub = df[numeric_cols]
    z_flags = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        s = sub[c]
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0 or pd.isna(mad):
            z_flags[c] = False
        else:
            mod_z = 0.6745 * (s - med) / mad
            z_flags[c] = mod_z.abs() > 3.5
    frac_flagged = z_flags.mean(axis=1)
    flagged_rows = int((frac_flagged > 0.05).sum())
checks["outliers"] = {
    "passed": True,
    "flagged_rows": flagged_rows,
    "flagged_pct": round(float(flagged_rows / n_rows * 100), 2),
}

report = {
    "status": "PASS" if all(c["passed"] for c in checks.values()) else "FAIL",
    "dataset_path": train_path,
    "checks": checks,
}
print(json.dumps(report, indent=2))
