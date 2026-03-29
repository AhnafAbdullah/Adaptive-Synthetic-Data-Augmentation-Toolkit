"""
loop_engine.py
--------------
Implements the closed-loop adaptive augmentation engine.

Each iteration of the loop:
  1. Samples targeted synthetic data from the fitted generator.
  2. Combines (interleaves) synthetic data with real training data.
  3. Preprocesses the augmented set using the existing preprocessor.
  4. Retrains the classifier.
  5. Evaluates on the held-out test set.
  6. Logs all metrics.
  7. Checks early-stopping criteria.
"""

import pandas as pd
import numpy as np
import copy

from data_utils import apply_preprocessor, interleave_data
from models_utils import build_and_train_model, evaluate_performance
from generator_utils import sample_targeted_synthetic_data


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def execute_adaptive_loop(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    preprocessor,
    initial_model,
    generator,
    cohorts_metadata: list,
    target_col: str = "survived",
    model_type: str = "random_forest",
    max_iterations: int = 5,
    n_samples_per_iter: int = 80,
    early_stop_threshold: float = 0.001,
    random_state: int = 42,
) -> tuple:
    """
    Iteratively augments training data with targeted synthetic samples,
    retrains the model, and evaluates until performance plateaus.

    Parameters
    ----------
    X_train : pd.DataFrame
        Raw (un-scaled) training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Raw test features (for reference / potential re-use).
    y_test : pd.Series
        Test labels.
    X_train_proc : np.ndarray
        Already-preprocessed training features (used as baseline).
    X_test_proc : np.ndarray
        Already-preprocessed test features.
    preprocessor : fitted ColumnTransformer
        Used to scale/encode new (augmented) batches.
    initial_model : fitted sklearn estimator
        The baseline model (will not be mutated; a fresh model is trained
        each iteration).
    generator : fitted SDV synthesizer
        The trained synthetic data generator.
    cohorts_metadata : list[dict]
        Cohort definitions passed to `sample_targeted_synthetic_data()`.
    target_col : str
        Name of the target column.
    model_type : str
        Model architecture used in each retrain.
    max_iterations : int
        Maximum number of augmentation-and-retrain cycles.
    n_samples_per_iter : int
        Requested synthetic samples per cohort per iteration.
    early_stop_threshold : float
        Minimum AUC improvement required to continue; stops if below.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    final_model : fitted sklearn estimator
        The best model found (highest AUC).
    metrics_history : list[dict]
        Per-iteration metric snapshots.
    """

    # Accumulate augmented data across iterations (real + all synthetic so far)
    X_aug_raw = X_train.copy()
    y_aug_raw = y_train.copy()

    # Baseline evaluation
    baseline_metrics = evaluate_performance(initial_model, X_test_proc, y_test, verbose=False)
    metrics_history = [{"iteration": 0, "label": "baseline", **baseline_metrics}]

    print(f"\n{'─'*60}")
    print(f"  Baseline  |  AUC={baseline_metrics['roc_auc']:.4f}  "
          f"F1={baseline_metrics['f1']:.4f}  "
          f"Recall={baseline_metrics['recall']:.4f}")
    print(f"{'─'*60}")

    best_auc = baseline_metrics["roc_auc"]
    best_model = initial_model
    prev_auc = best_auc

    for iteration in range(1, max_iterations + 1):
        print(f"\n[loop_engine] ── Iteration {iteration}/{max_iterations} ──")

        # 1. Generate synthetic samples for the weak cohorts
        synthetic_df = sample_targeted_synthetic_data(
            generator=generator,
            real_data=pd.concat([X_aug_raw, y_aug_raw.rename(target_col)], axis=1),
            cohorts_metadata=cohorts_metadata,
            n_samples=n_samples_per_iter,
            target_col=target_col,
        )

        if synthetic_df.empty:
            print("[loop_engine] No synthetic data produced; stopping early.")
            break

        # 2. Interleave synthetic with current accumulated real+synthetic data
        real_combined = pd.concat(
            [X_aug_raw, y_aug_raw.rename(target_col)], axis=1
        )
        X_aug_df, y_aug_series = interleave_data(
            real_combined, synthetic_df, target_col=target_col, random_state=random_state + iteration
        )

        # 3. Preprocess augmented set
        try:
            X_aug_proc = apply_preprocessor(X_aug_df, preprocessor)
        except Exception as e:
            print(f"[loop_engine] Preprocessing failed: {e}. Skipping iteration.")
            continue

        # 4. Retrain model on augmented data
        iter_model = build_and_train_model(
            X_aug_proc, y_aug_series.values,
            model_type=model_type,
            random_state=random_state + iteration,
        )

        # 5. Evaluate on held-out test set
        iter_metrics = evaluate_performance(iter_model, X_test_proc, y_test, verbose=False)
        iter_metrics["iteration"] = iteration
        iter_metrics["label"] = f"iter_{iteration}"
        iter_metrics["synthetic_added"] = len(synthetic_df)
        iter_metrics["total_train_size"] = len(y_aug_series)
        metrics_history.append(iter_metrics)

        curr_auc = iter_metrics["roc_auc"]
        delta_auc = curr_auc - prev_auc

        print(f"  Iter {iteration:2d}   |  AUC={curr_auc:.4f}  "
              f"F1={iter_metrics['f1']:.4f}  "
              f"Recall={iter_metrics['recall']:.4f}  "
              f"ΔAUC={delta_auc:+.4f}  "
              f"(+{len(synthetic_df)} synthetic)")

        # Track best model
        if curr_auc > best_auc:
            best_auc = curr_auc
            best_model = iter_model

        # Update accumulated dataset (grow it with validated synthetic data)
        X_aug_raw = X_aug_df.copy()
        y_aug_raw = y_aug_series.copy()

        # 6. Early stopping
        if abs(delta_auc) < early_stop_threshold and iteration > 1:
            print(f"\n[loop_engine] Early stopping: ΔAUC ({delta_auc:+.5f}) "
                  f"< threshold ({early_stop_threshold}). Best AUC={best_auc:.4f}")
            break

        prev_auc = curr_auc

    print(f"\n{'─'*60}")
    print(f"[loop_engine] Loop complete. Best AUC: {best_auc:.4f}")
    print(f"{'─'*60}\n")

    return best_model, metrics_history


def execute_adaptive_loop_stream(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    preprocessor,
    initial_model,
    generator,
    cohorts_metadata: list,
    target_col: str = "survived",
    model_type: str = "random_forest",
    max_iterations: int = 5,
    n_samples_per_iter: int = 80,
    early_stop_threshold: float = 0.001,
    random_state: int = 42,
):
    """
    Generator version of the adaptive loop for Streamlit integration.
    Yields dicts describing the current state.
    """
    X_aug_raw = X_train.copy()
    y_aug_raw = y_train.copy()

    baseline_metrics = evaluate_performance(initial_model, X_test_proc, y_test, verbose=False)
    metrics_history = [{"iteration": 0, "label": "baseline", **baseline_metrics}]

    best_auc = baseline_metrics["roc_auc"]
    best_model = initial_model
    prev_auc = best_auc
    
    best_metrics = metrics_history[0].copy()
    best_data = pd.concat([X_aug_raw, y_aug_raw.rename(target_col)], axis=1)

    yield {"status": "baseline", "metrics": baseline_metrics, "metrics_history": metrics_history}

    for iteration in range(1, max_iterations + 1):
        yield {"status": "generating", "iteration": iteration, "max_iterations": max_iterations}
        
        synthetic_df = sample_targeted_synthetic_data(
            generator=generator,
            real_data=pd.concat([X_aug_raw, y_aug_raw.rename(target_col)], axis=1),
            cohorts_metadata=cohorts_metadata,
            n_samples=n_samples_per_iter,
            target_col=target_col,
        )

        if synthetic_df.empty:
            yield {"status": "early_stop", "reason": "No synthetic data produced"}
            break

        real_combined = pd.concat([X_aug_raw, y_aug_raw.rename(target_col)], axis=1)
        X_aug_df, y_aug_series = interleave_data(
            real_combined, synthetic_df, target_col=target_col, random_state=random_state + iteration
        )

        try:
            X_aug_proc = apply_preprocessor(X_aug_df, preprocessor)
        except Exception as e:
            yield {"status": "error", "message": f"Preprocessing failed: {e}"}
            continue

        yield {"status": "training", "iteration": iteration, "max_iterations": max_iterations}
        iter_model = build_and_train_model(
            X_aug_proc, y_aug_series.values,
            model_type=model_type,
            random_state=random_state + iteration,
        )

        iter_metrics = evaluate_performance(iter_model, X_test_proc, y_test, verbose=False)
        iter_metrics["iteration"] = iteration
        iter_metrics["label"] = f"iter_{iteration}"
        iter_metrics["synthetic_added"] = len(synthetic_df)
        iter_metrics["total_train_size"] = len(y_aug_series)
        metrics_history.append(iter_metrics)

        curr_auc = iter_metrics["roc_auc"]
        delta_auc = curr_auc - prev_auc

        if curr_auc > best_auc:
            best_auc = curr_auc
            best_model = iter_model
            best_metrics = iter_metrics.copy()
            best_data = pd.concat([X_aug_df, y_aug_series.rename(target_col)], axis=1)

        X_aug_raw = X_aug_df.copy()
        y_aug_raw = y_aug_series.copy()

        yield {"status": "iter_complete", "iteration": iteration, "metrics": iter_metrics, "metrics_history": list(metrics_history), "best_model": best_model}

        if abs(delta_auc) < early_stop_threshold and iteration > 1:
            yield {"status": "early_stop", "reason": f"ΔAUC ({delta_auc:+.5f}) < threshold. Best AUC: {best_auc:.4f}"}
            break

        prev_auc = curr_auc

    yield {"status": "complete", "best_model": best_model, "best_metrics": best_metrics, "metrics_history": metrics_history, "final_data": best_data}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def save_metrics_log(metrics_history: list, path: str = "training_loop_log.csv") -> None:
    """Saves the metrics history to a CSV file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(metrics_history)
    df.to_csv(path, index=False)
    print(f"[loop_engine] Metrics log saved → {path}")