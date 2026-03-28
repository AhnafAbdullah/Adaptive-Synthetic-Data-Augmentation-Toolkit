import sys, warnings, os, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from data_utils import preprocess_features
from models_utils import build_and_train_model, evaluate_performance
from generator_utils import train_generator
from loop_engine import execute_adaptive_loop

# 1. Generate Highly Imbalanced Dataset
X, y = make_classification(n_samples=4000, n_features=15, n_informative=5,
                           n_redundant=2, n_clusters_per_class=2, weights=[0.95, 0.05],
                           flip_y=0.01, random_state=42)
df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(15)])
df['target'] = y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42, stratify=df['target'])

X_train_proc, X_test_proc, preprocessor, feat_names = preprocess_features(X_train, X_test)

# 2. Baseline
baseline = build_and_train_model(X_train_proc, y_train.values, model_type='random_forest', model_params={'n_estimators': 100, 'max_depth': 4, 'class_weight': None})
baseline_metrics = evaluate_performance(baseline, X_test_proc, y_test.values, verbose=False)
print(f"Baseline -> AUC={baseline_metrics['roc_auc']:.4f}  F1={baseline_metrics['f1']:.4f}  Recall={baseline_metrics['recall']:.4f} Precision={baseline_metrics['precision']:.4f}")

# 3. Generator
train_full = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename('target')], axis=1)
generator = train_generator(train_full, target_col='target', generator_type='copula')

# Target cohorts: target the minority class broadly
target_cohorts = [
    {'name': 'minority_class', 'group_col': 'target', 'group_val': 1, 'cohort_type': 'equality', 'conditions': {'target': 1}, 'label': 1, 'recall': baseline_metrics['recall'], 'n_survivors_in_test': sum(y_test)}
]

# 4. Adaptive loop
final_model, history = execute_adaptive_loop(
    X_train=X_train.reset_index(drop=True), y_train=y_train.reset_index(drop=True),
    X_test=X_test.reset_index(drop=True), y_test=y_test.reset_index(drop=True),
    X_train_proc=X_train_proc, X_test_proc=X_test_proc,
    preprocessor=preprocessor, initial_model=baseline, generator=generator,
    cohorts_metadata=target_cohorts, target_col='target',
    model_type='random_forest', max_iterations=4, n_samples_per_iter=300,
    early_stop_threshold=0.0005, random_state=42
)

final_metrics = evaluate_performance(final_model, X_test_proc, y_test.values)
print(f"Final -> AUC={final_metrics['roc_auc']:.4f}  F1={final_metrics['f1']:.4f}  Recall={final_metrics['recall']:.4f} Precision={final_metrics['precision']:.4f}")

delta_f1 = final_metrics['f1'] - baseline_metrics['f1']
delta_rec = final_metrics['recall'] - baseline_metrics['recall']
print(f"Delta: F1 {delta_f1:+.4f}, Recall {delta_rec:+.4f}")
