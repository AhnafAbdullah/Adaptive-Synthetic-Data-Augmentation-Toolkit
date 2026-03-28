"""
data_utils.py
-------------
Handles all data ingestion, cleaning, feature preprocessing, and merging
for the Adaptive Synthetic Data Augmentation Toolkit.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_data(
    file_path: str = "seaborn:titanic",
    target_col: str = "survived",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Loads a tabular dataset and returns clean train/test splits.

    Parameters
    ----------
    file_path : str
        Either a path to a CSV file or a special token like "seaborn:<name>"
        to load built-in seaborn datasets (e.g. "seaborn:titanic").
    target_col : str
        Name of the binary target column.
    test_size : float
        Fraction of data to set aside for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
        Clean, raw (un-scaled) feature splits plus labels.
    """
    # ---- Load ---------------------------------------------------------------
    if file_path.startswith("seaborn:"):
        dataset_name = file_path.split(":")[1]
        df = sns.load_dataset(dataset_name)
    elif file_path == "simulated:fraud":
        try:
            df = pd.read_csv("simulated_fraud_data.csv")
        except FileNotFoundError:
            from sklearn.datasets import make_classification
            X_sim, y_sim = make_classification(n_samples=4000, n_features=15, n_informative=5,
                                       n_redundant=2, n_clusters_per_class=2, weights=[0.95, 0.05],
                                       flip_y=0.01, random_state=42)
            df = pd.DataFrame(X_sim, columns=[f'feature_{i}' for i in range(15)])
            df[target_col] = y_sim
            df.to_csv("simulated_fraud_data.csv", index=False)
    else:
        df = pd.read_csv(file_path)

    # ---- Dataset-specific cleaning (titanic) --------------------------------
    if "titanic" in file_path:
        df = _clean_titanic(df, target_col)
    elif "fraud" in file_path:
        pass # already clean
    else:
        df = _generic_clean(df, target_col)

    # ---- Split --------------------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), \
           y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def _clean_titanic(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Titanic-specific feature engineering and cleaning."""
    # Keep only useful columns
    useful_cols = [target_col, "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df = df[useful_cols].copy()

    # Fill missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # Drop any remaining NaN rows
    df = df.dropna()

    # Encode target as int
    df[target_col] = df[target_col].astype(int)

    return df


def _generic_clean(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Generic cleaning for arbitrary CSV datasets."""
    # Drop columns with > 50% missing
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Fill numeric NaNs with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    df[target_col] = df[target_col].astype(int)
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    Builds and fits a ColumnTransformer on training data, then transforms
    both training and test sets.

    Returns
    -------
    X_train_proc, X_test_proc : np.ndarray
        Transformed feature matrices.
    preprocessor : fitted ColumnTransformer
        Save this to apply the same transformation to synthetic data.
    feature_names : list[str]
        Output feature names after transformation.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ], remainder="drop")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Recover feature names
    num_names = numeric_cols
    cat_names = list(preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"]
                     .get_feature_names_out(categorical_cols)) if categorical_cols else []
    feature_names = num_names + cat_names

    return X_train_proc, X_test_proc, preprocessor, feature_names


def apply_preprocessor(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """
    Applies a pre-fitted preprocessor to a new DataFrame (e.g. synthetic data).
    """
    return preprocessor.transform(X)


# ---------------------------------------------------------------------------
# Interleaving
# ---------------------------------------------------------------------------

def interleave_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_col: str = "survived",
    random_state: int = 42,
) -> tuple:
    """
    Concatenates real and synthetic DataFrames (including the target column),
    shuffles them to prevent the retrained model from overfitting to synthetic
    quirks, and returns X, y splits.

    Parameters
    ----------
    real_data : pd.DataFrame
        Original training features (no target column).
    synthetic_data : pd.DataFrame
        Synthetic samples including the target column.
    target_col : str
        Name of the target column present in synthetic_data.

    Returns
    -------
    X_aug : pd.DataFrame
    y_aug : pd.Series
    """
    # synthetic_data already has the target; add it to real_data
    combined = pd.concat([real_data, synthetic_data], ignore_index=True)
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_aug = combined[target_col]
    X_aug = combined.drop(columns=[target_col])

    return X_aug, y_aug


def save_splits(X_train, y_train, X_test, y_test, output_dir: str = ".") -> None:
    """Saves the four data splits as CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train_clean.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train_clean.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test_clean.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test_clean.csv", index=False)
    print(f"[data_utils] Splits saved to '{output_dir}'")