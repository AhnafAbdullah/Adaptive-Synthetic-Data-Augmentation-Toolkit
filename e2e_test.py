"""
e2e_test.py — Full end-to-end validation of the pipeline.
Run: python e2e_test.py
"""
import sys, warnings, os, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import joblib

# ── Step 1: Data ────────────────────────────────────────────────────────
print("=== STEP 1: Data Ingestion & Preprocessing ===")
from data_utils import ingest_data, preprocess_features, save_splits

X_train, X_test, y_train, y_test = ingest_data('simulated:fraud', target_col='is_fraud')
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Class balance (train): {y_train.value_counts().to_dict()}")

X_train_proc, X_test_proc, preprocessor, feat_names = preprocess_features(X_train, X_test)
print(f"Preprocessed shape: {X_train_proc.shape}")

os.makedirs('artifacts', exist_ok=True)
save_splits(X_train, y_train, X_test, y_test, output_dir='artifacts')
joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')

# ── Step 2: Baseline Model ───────────────────────────────────────────────
print("\n=== STEP 2: Baseline Model ===")
from models_utils import build_and_train_model, evaluate_performance, save_model

baseline = build_and_train_model(
    X_train_proc, y_train.values,
    model_type='random_forest',
    model_params={'n_estimators': 100, 'max_depth': 4, 'class_weight': None},
)
baseline_metrics = evaluate_performance(baseline, X_test_proc, y_test.values, verbose=False)
print(f"Baseline -> AUC={baseline_metrics['roc_auc']:.4f}  "
      f"F1={baseline_metrics['f1']:.4f}  Recall={baseline_metrics['recall']:.4f}")
save_model(baseline, 'artifacts/baseline_rf_model.pkl')

# ── Step 0: Generator ─────────────────────────────────────────────────────
print("\n=== STEP 0: Synthetic Generator ===")
from generator_utils import train_generator, evaluate_synthetic_quality, save_generator

train_full = pd.concat([X_train, y_train.rename('is_fraud')], axis=1)
generator = train_generator(train_full, target_col='is_fraud', generator_type='copula')

synthetic_raw = generator.sample(num_rows=00)
synthetic_raw['is_fraud'] = 1
print(f"Quick sample: {synthetic_raw.shape}")

quality = evaluate_synthetic_quality(train_full, synthetic_raw, target_col='is_fraud')
print(f"Quality - Mean KS: {quality['overall_ks_mean']}  Max KS: {quality['overall_ks_max']}")

save_generator(generator, 'artifacts/sdv_copula_model.pkl')

# Minimal target cohorts
target_cohorts = [
    {
        'name': 'minority_class_boost',
        'group_col': 'is_fraud',
        'group_val': '1',
        'cohort_type': 'equality',
        'conditions': {'is_fraud': 1},
        'label': 1,
        'recall': 0.20,
        'n_survivors_in_test': 20,
    }
]
with open('artifacts/target_cohorts.json', 'w') as f:
    json.dump(target_cohorts, f, indent=2)
print(f"target_cohorts.json written ({len(target_cohorts)} cohorts)")

# ── Step 4: Adaptive Loop ─────────────────────────────────────────────────
print("\n=== STEP 4: Adaptive Loop (0 iterations) ===")
from loop_engine import execute_adaptive_loop, save_metrics_log

final_model, history = execute_adaptive_loop(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_train_proc=X_train_proc,
    X_test_proc=X_test_proc,
    preprocessor=preprocessor,
    initial_model=baseline,
    generator=generator,
    cohorts_metadata=target_cohorts,
    target_col='is_fraud',
    model_type='random_forest',
    max_iterations=0,
    n_samples_per_iter=300,
    early_stop_threshold=0.001,
    random_state=42,
)

final_metrics = evaluate_performance(final_model, X_test_proc, y_test.values)
print(f"Final -> AUC={final_metrics['roc_auc']:.4f}  "
      f"F1={final_metrics['f1']:.4f}  Recall={final_metrics['recall']:.4f}")

save_model(final_model, 'artifacts/final_adapted_model.pkl')
save_metrics_log(history, 'artifacts/training_loop_log.csv')

print("\n=== Metrics History ===")
log = pd.DataFrame(history)
print(log[['label', 'roc_auc', 'f1', 'recall']].to_string(index=False))

print("\n=== All artifacts ===")
for fname in sorted(os.listdir('artifacts')):
    print(f"  {fname}")

print("\nFULL END-TO-END TEST PASSED!")
