import numpy as np
import pandas as pd

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
REQUIRED_COLS = [
    "PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination",
    "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name",
]


def engineer_features(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"engineer_features missing required columns: {missing}")

    out = df.copy()

    pid = out["PassengerId"].astype("string")
    out["Group"] = pid.str.split("_").str[0]
    out["GroupPosition"] = pd.to_numeric(pid.str.split("_").str[1], errors="coerce")
    group_size = out.groupby("Group")["PassengerId"].transform("size")
    out["GroupSize"] = group_size.astype("int64")
    out["IsAlone"] = (out["GroupSize"] == 1).astype("int64")

    cabin = out["Cabin"].astype("string")
    parts = cabin.str.split("/", expand=True)
    out["Deck"] = parts[0] if parts.shape[1] > 0 else pd.NA
    out["CabinNum"] = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
    out["Side"] = parts[2] if parts.shape[1] > 2 else pd.NA
    out["CabinRegion"] = (out["CabinNum"] // 300)

    spend = out[SPEND_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out["TotalSpend"] = spend.sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype("int64")
    out["TotalSpend_log1p"] = np.log1p(out["TotalSpend"])
    for c in SPEND_COLS:
        out[f"{c}_log1p"] = np.log1p(spend[c])
    out["LuxurySpend"] = spend[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
    out["AmenitiesUsedCount"] = (spend > 0).sum(axis=1).astype("int64")
    group_spend = out.groupby("Group")["TotalSpend"].transform("sum")
    out["GroupSpendTotal"] = group_spend

    age = pd.to_numeric(out["Age"], errors="coerce")
    out["Age_missing"] = age.isna().astype("int64")
    out["IsChild"] = (age < 13).fillna(False).astype("int64")
    out["AgeBin"] = pd.cut(
        age, bins=[-1, 12, 18, 25, 40, 60, 200], labels=[0, 1, 2, 3, 4, 5]
    ).astype("float64")

    name = out["Name"].astype("string")
    out["Surname"] = name.str.split(" ").str[-1]
    out["FamilySize"] = out.groupby("Surname")["PassengerId"].transform("size")
    out.loc[out["Surname"].isna(), "FamilySize"] = 1
    out["FamilySize"] = out["FamilySize"].astype("float64")

    cryo = out["CryoSleep"]
    cryo_true = cryo.astype("string").str.lower().isin(["true", "1", "1.0"])
    out["CryoSpendInconsistent"] = (cryo_true & (out["TotalSpend"] > 0)).astype("int64")
    out["CryoImpliedBySpend"] = (out["TotalSpend"] == 0).astype("int64")

    for c in ["Age", "HomePlanet", "CryoSleep", "Destination", "VIP"] + SPEND_COLS:
        out[f"{c}_missing"] = out[c].isna().astype("int64")

    spend_log = {c: np.log1p(spend[c]) for c in SPEND_COLS}
    age_filled = age.fillna(age.median())
    out["SpendPerAge"] = out["TotalSpend"] / (age_filled + 1.0)
    out["Spa_VRDeck_interaction"] = spend_log["Spa"] * spend_log["VRDeck"]
    out["RoomService_FoodCourt_interaction"] = spend_log["RoomService"] * spend_log["FoodCourt"]

    return out


ENGINEERED_COLUMNS = [
    "Group", "GroupPosition", "GroupSize", "IsAlone",
    "Deck", "CabinNum", "Side", "CabinRegion",
    "TotalSpend", "NoSpend", "TotalSpend_log1p",
    "RoomService_log1p", "FoodCourt_log1p", "ShoppingMall_log1p", "Spa_log1p", "VRDeck_log1p",
    "LuxurySpend", "AmenitiesUsedCount", "GroupSpendTotal",
    "Age_missing", "IsChild", "AgeBin",
    "Surname", "FamilySize",
    "CryoSpendInconsistent", "CryoImpliedBySpend",
    "HomePlanet_missing", "CryoSleep_missing", "Destination_missing", "VIP_missing",
    "RoomService_missing", "FoodCourt_missing", "ShoppingMall_missing", "Spa_missing", "VRDeck_missing",
    "SpendPerAge", "Spa_VRDeck_interaction", "RoomService_FoodCourt_interaction",
]


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    d = pd.read_csv(src)
    before = len(d)
    res = engineer_features(d)
    after = len(res)
    produced = [c for c in ENGINEERED_COLUMNS if c in res.columns]
    missing_out = [c for c in ENGINEERED_COLUMNS if c not in res.columns]
    print(f"rows_before={before} rows_after={after} produced={len(produced)} missing={missing_out}")
