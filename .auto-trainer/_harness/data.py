import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Age", "GroupPosition", "GroupSize", "IsAlone", "CabinNum", "CabinRegion",
    "TotalSpend", "NoSpend", "TotalSpend_log1p",
    "RoomService_log1p", "FoodCourt_log1p", "ShoppingMall_log1p", "Spa_log1p", "VRDeck_log1p",
    "LuxurySpend", "AmenitiesUsedCount", "GroupSpendTotal",
    "Age_missing", "IsChild", "AgeBin", "FamilySize",
    "CryoSpendInconsistent", "CryoImpliedBySpend",
    "HomePlanet_missing", "CryoSleep_missing", "Destination_missing", "VIP_missing",
    "RoomService_missing", "FoodCourt_missing", "ShoppingMall_missing", "Spa_missing", "VRDeck_missing",
    "SpendPerAge", "Spa_VRDeck_interaction", "RoomService_FoodCourt_interaction",
]
CATEGORICAL_FEATURES = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]


def _load_features_module(features_path):
    spec = importlib.util.spec_from_file_location("features", features_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engineer(df, features_path):
    if features_path and os.path.exists(features_path):
        return _load_features_module(features_path).engineer_features(df)
    return df


def build_preprocessor(scale):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def _prep_categoricals(df):
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("object")
        df[c] = df[c].where(df[c].notna(), "missing").astype(str)
    return df


def load_split(config):
    train_path = config["train_path"]
    target = config["target_column"]
    features_path = config.get("features_path")

    df = pd.read_csv(train_path)
    df = _engineer(df, features_path)
    df = _prep_categoricals(df)

    y = df[target].astype(int).to_numpy()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config["val_size"], random_state=config["seed"], stratify=y
    )
    return X_train, X_val, y_train, y_val


def load_test(config):
    test_path = config["test_path"]
    id_col = config["id_column"]
    features_path = config.get("features_path")
    df = pd.read_csv(test_path)
    ids = df[id_col].to_numpy()
    df = _engineer(df, features_path)
    df = _prep_categoricals(df)
    X_test = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return X_test, ids
