"""
generator_utils.py
------------------
Handles synthetic data generation using SDV (Synthetic Data Vault).
Provides targeted oversampling of identified "weak cohorts" and quality
evaluation against the real data distribution.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Generator training
# ---------------------------------------------------------------------------

def train_generator(
    real_data: pd.DataFrame,
    target_col: str = "survived",
    generator_type: str = "copula",
    max_fit_samples: int = 5000,
) -> object:
    """
    Fits a generative model on the full real training dataset.

    Parameters
    ----------
    real_data : pd.DataFrame
        Raw (un-scaled) training DataFrame including the target column.
    target_col : str
        Name of the binary target column.
    generator_type : str
        "copula" → GaussianCopulaSynthesizer (default, fast)
        "tvae"   → TVAESynthesizer (slower, neural)
    max_fit_samples : int
        Caps the amount of rows passed to SDV for extreme performance limits.

    Returns
    -------
    synthesizer : fitted SDV synthesizer object
    """
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer

    if len(real_data) > max_fit_samples:
        fit_data = real_data.sample(n=max_fit_samples, random_state=42)
    else:
        fit_data = real_data

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(fit_data)

    if generator_type == "copula":
        synthesizer = GaussianCopulaSynthesizer(metadata)
    elif generator_type == "tvae":
        synthesizer = TVAESynthesizer(metadata)
    else:
        raise ValueError(f"Unknown generator_type '{generator_type}'. Use 'copula' or 'tvae'.")

    synthesizer.fit(fit_data)
    print(f"[generator_utils] '{generator_type}' synthesizer fitted on {len(fit_data)} rows.")
    return synthesizer


# ---------------------------------------------------------------------------
# Targeted sampling
# ---------------------------------------------------------------------------

def sample_targeted_synthetic_data(
    generator: object,
    real_data: pd.DataFrame,
    cohorts_metadata: list,
    n_samples: int = 100,
    target_col: str = "survived",
    conservative_limit: float = 0.30,
) -> pd.DataFrame:
    """
    Generates synthetic samples focused on the identified problem cohorts.

    For each cohort definition, a subset of the real training data matching
    those conditions is extracted, a lightweight per-cohort synthesizer is
    fitted (or the global generator is used with conditions), and samples are
    drawn.

    A *conservative generation limit* caps synthetic output at
    `conservative_limit` × |real cohort data| to avoid overwhelming the
    real signal with synthetic artifacts.

    Parameters
    ----------
    generator : fitted SDV synthesizer
        The global synthesizer returned by `train_generator()`.
    real_data : pd.DataFrame
        Raw training data (same one used to fit the generator).
    cohorts_metadata : list[dict]
        Each dict must have:
            "name"       : str  — human-readable label
            "conditions" : dict — column → value (equality) filter
            "label"      : int  — the target class to assign (0 or 1)
    n_samples : int
        Per-cohort sample request before the conservative limit is applied.
    target_col : str
        Name of the binary target column.
    conservative_limit : float
        Maximum ratio of synthetic↔real data per cohort (default 30%).

    Returns
    -------
    synthetic_df : pd.DataFrame
        All generated samples concatenated (includes target column).
    """
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    all_synthetic = []

    for cohort in cohorts_metadata:
        name = cohort.get("name", "unnamed")
        conditions = cohort.get("conditions", {})
        label = cohort.get("label", 1)

        # Filter real data to this cohort
        mask = pd.Series([True] * len(real_data), index=real_data.index)
        for col, val in conditions.items():
            if col in real_data.columns:
                mask = mask & (real_data[col] == val)

        cohort_data = real_data[mask].copy()

        if len(cohort_data) < 5:
            print(f"[generator_utils] Cohort '{name}' has < 5 samples; skipping.")
            continue

        # Conservative limit: cap requested samples
        max_allowed = max(1, int(len(cohort_data) * conservative_limit))
        actual_n = min(n_samples, max_allowed)

        print(f"[generator_utils] Cohort '{name}': {len(cohort_data)} real → "
              f"generating {actual_n} synthetic samples.")

        # Fit a lightweight per-cohort synthesizer
        try:
            if len(cohort_data) > 5000:
                fit_cohort_data = cohort_data.sample(n=5000, random_state=42)
            else:
                fit_cohort_data = cohort_data
                
            meta = SingleTableMetadata()
            meta.detect_from_dataframe(fit_cohort_data)
            cohort_synth = GaussianCopulaSynthesizer(meta)
            cohort_synth.fit(fit_cohort_data)
            samples = cohort_synth.sample(num_rows=actual_n)
            # Override target column with the intended label
            samples[target_col] = label
        except Exception as e:
            print(f"[generator_utils] Per-cohort fit failed for '{name}': {e}. "
                  f"Falling back to global generator.")
            samples = generator.sample(num_rows=actual_n)
            samples[target_col] = label

        all_synthetic.append(samples)

    if not all_synthetic:
        print("[generator_utils] Warning: no synthetic samples produced.")
        return pd.DataFrame()

    synthetic_df = pd.concat(all_synthetic, ignore_index=True)
    print(f"[generator_utils] Total synthetic samples generated: {len(synthetic_df)}")
    return synthetic_df


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

def evaluate_synthetic_quality(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_col: str = "survived",
) -> dict:
    """
    Measures statistical closeness between real and synthetic distributions.

    Metrics computed per numeric column:
    - KS-statistic (lower = more similar distributions)
    - Mean absolute difference
    - Std absolute difference

    Returns
    -------
    quality_report : dict
        {
          "per_column": {col: {"ks_stat": ..., "mean_diff": ..., "std_diff": ...}},
          "overall_ks_mean": float,
          "overall_ks_max":  float,
        }
    """
    feature_cols = [c for c in real_data.columns if c != target_col]
    numeric_cols = real_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    per_column = {}
    ks_stats = []

    for col in numeric_cols:
        if col not in synthetic_data.columns:
            continue
        real_vals = real_data[col].dropna().values
        synt_vals = synthetic_data[col].dropna().values

        ks_stat, _ = stats.ks_2samp(real_vals, synt_vals)
        mean_diff = abs(real_vals.mean() - synt_vals.mean())
        std_diff  = abs(real_vals.std()  - synt_vals.std())

        per_column[col] = {
            "ks_stat":   round(ks_stat, 4),
            "mean_diff": round(mean_diff, 4),
            "std_diff":  round(std_diff, 4),
        }
        ks_stats.append(ks_stat)

    quality_report = {
        "per_column":      per_column,
        "overall_ks_mean": round(np.mean(ks_stats), 4) if ks_stats else None,
        "overall_ks_max":  round(np.max(ks_stats),  4) if ks_stats else None,
    }

    print("[generator_utils] Synthetic quality report:")
    print(f"  Mean KS stat : {quality_report['overall_ks_mean']}")
    print(f"  Max  KS stat : {quality_report['overall_ks_max']}")
    return quality_report


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_generator(synthesizer: object, path: str) -> None:
    """Saves a fitted SDV synthesizer to disk."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    synthesizer.save(path)
    print(f"[generator_utils] Synthesizer saved -> {path}")


def load_generator(path: str) -> object:
    """Loads an SDV synthesizer from disk."""
    from sdv.single_table import GaussianCopulaSynthesizer
    synthesizer = GaussianCopulaSynthesizer.load(path)
    print(f"[generator_utils] Synthesizer loaded <- {path}")
    return synthesizer