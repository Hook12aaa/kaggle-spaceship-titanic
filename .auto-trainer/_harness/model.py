import numpy as np
from scipy.special import expit
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

NEEDS_SCALING = {"logistic_regression", "ridge_classifier", "knn", "mlp", "linear_svc"}


def build_model(config):
    mt = config["model_type"]
    hp = config.get("hyperparameters", {})
    seed = config["seed"]
    if mt == "logistic_regression":
        return LogisticRegression(C=hp.get("C", 1.0), max_iter=2000, random_state=seed)
    if mt == "ridge_classifier":
        return RidgeClassifier(alpha=hp.get("alpha", 1.0), random_state=seed)
    if mt == "random_forest":
        return RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            min_samples_leaf=hp.get("min_samples_leaf", 1),
            random_state=seed, n_jobs=-1,
        )
    if mt == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            random_state=seed, n_jobs=-1,
        )
    if mt == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 100),
            learning_rate=hp.get("learning_rate", 0.1),
            max_depth=hp.get("max_depth", 3),
            random_state=seed,
        )
    if mt == "knn":
        return KNeighborsClassifier(
            n_neighbors=hp.get("n_neighbors", 5),
            weights=hp.get("weights", "uniform"),
            n_jobs=-1,
        )
    if mt == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(hp.get("hidden_layer_sizes", [64])),
            alpha=hp.get("alpha", 1e-4),
            max_iter=hp.get("max_iter", 300),
            random_state=seed,
        )
    if mt == "linear_svc":
        return LinearSVC(C=hp.get("C", 1.0), random_state=seed)
    raise ValueError(f"unknown model_type: {mt}")


def needs_scaling(model_type):
    return model_type in NEEDS_SCALING


def predict_proba_pos(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return expit(estimator.decision_function(X))
    return estimator.predict(X).astype(float)


def count_trainable_params(fitted_model, n_features, n_train):
    name = type(fitted_model).__name__
    if name in ("LogisticRegression", "RidgeClassifier", "LinearSVC"):
        return int(np.asarray(fitted_model.coef_).size + np.asarray(fitted_model.intercept_).size)
    if name in ("RandomForestClassifier", "ExtraTreesClassifier"):
        return int(sum(t.tree_.node_count for t in fitted_model.estimators_))
    if name == "GradientBoostingClassifier":
        return int(sum(t.tree_.node_count for t in fitted_model.estimators_.ravel()))
    if name == "MLPClassifier":
        return int(sum(c.size for c in fitted_model.coefs_) + sum(b.size for b in fitted_model.intercepts_))
    if name == "KNeighborsClassifier":
        return int(n_train * n_features)
    return int(n_features)
