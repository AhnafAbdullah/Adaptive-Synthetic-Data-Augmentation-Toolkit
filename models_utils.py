"""
models_utils.py
---------------
Handles model training and performance evaluation for the Adaptive
Synthetic Data Augmentation Toolkit.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import joblib
import os


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
}


def build_and_train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    model_params: dict | None = None,
    random_state: int = 42,
) -> object:
    """
    Initializes and trains a scikit-learn classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training feature matrix.
    y_train : np.ndarray
        Binary target labels.
    model_type : str
        One of "random_forest", "decision_tree", "logistic_regression".
    model_params : dict, optional
        Keyword arguments forwarded to the sklearn constructor.
    random_state : int
        Reproducibility seed (applied where the model supports it).

    Returns
    -------
    model : fitted sklearn estimator
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: {list(SUPPORTED_MODELS)}"
        )

    model_params = model_params or {}

    # Inject random_state for models that support it
    if model_type in ("random_forest", "decision_tree", "logistic_regression"):
        model_params.setdefault("random_state", random_state)

    # Extra defaults for LR to avoid convergence warnings
    if model_type == "logistic_regression":
        model_params.setdefault("max_iter", 1000)
        model_params.setdefault("solver", "lbfgs")

    model = SUPPORTED_MODELS[model_type](**model_params)
    model.fit(X_train, y_train)

    print(f"[model_utils] '{model_type}' trained on {len(y_train)} samples.")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_performance(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = False,
) -> dict:
    """
    Computes a comprehensive set of classification metrics.

    Returns
    -------
    metrics : dict
        Keys: accuracy, precision, recall, f1, roc_auc
        (macro-averaged for multi-class safety; works for binary too)
    """
    y_pred = model.predict(X_test)

    # For AUC we need probability scores
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="binary", zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    if verbose:
        print("[model_utils] Evaluation results:")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
        print()
        print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: object, path: str) -> None:
    """Serialises a fitted model to disk using joblib."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"[model_utils] Model saved -> {path}")


def load_model(path: str) -> object:
    """Loads a joblib-serialised model from disk."""
    model = joblib.load(path)
    print(f"[model_utils] Model loaded <- {path}")
    return model