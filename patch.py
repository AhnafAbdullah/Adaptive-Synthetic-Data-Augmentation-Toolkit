import sys, json

# Update e2e_test.py
with open('e2e_test.py', 'r', encoding='utf-8') as f:
    e2e = f.read()

e2e = e2e.replace("'seaborn:titanic'", "'simulated:fraud'")
e2e = e2e.replace("target_col='survived'", "target_col='is_fraud'")
e2e = e2e.replace("'survived'", "'is_fraud'")
e2e = e2e.replace("'pclass'", "'feature_1'")
e2e = e2e.replace("'sex'", "'feature_2'")
e2e = e2e.replace("3", "0")
e2e = e2e.replace("'male'", "0")
e2e = e2e.replace("max_iterations=3", "max_iterations=4")
e2e = e2e.replace("n_samples_per_iter=40", "n_samples_per_iter=300")
e2e = e2e.replace("max_depth': 8", "max_depth': 4")
e2e = e2e.replace("'class_weight': 'balanced'", "'class_weight': None")
with open('e2e_test.py', 'w', encoding='utf-8') as f:
    f.write(e2e)

# Update Notebooks
def patch_notebook(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        src = cell['source']
        for i in range(len(src)):
            src[i] = src[i].replace("seaborn:titanic", "simulated:fraud")
            src[i] = src[i].replace("survived", "is_fraud")
            # src[i] = src[i].replace("Survived", "Is_Fraud") # Avoid changing labels aggressively
            src[i] = src[i].replace("survivor", "fraud_case")
            src[i] = src[i].replace("survivors", "fraud_cases")
            src[i] = src[i].replace("titanic", "fraud")
            src[i] = src[i].replace("Titanic", "Fraud")
            src[i] = src[i].replace("['age', 'fare', 'sibsp', 'parch', 'pclass']", "['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']")
            src[i] = src[i].replace("['pclass', 'sex', 'age_band', 'fare_band', 'embarked']", "['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']")
            src[i] = src[i].replace("'sex', 'embarked'", "'feature_5', 'feature_6'")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

patch_notebook('01_Baseline_and_EDA.ipynb')
patch_notebook('02_Error_Analysis.ipynb')
patch_notebook('03_Static_Synthetic_Generation.ipynb')
patch_notebook('04_Closed_Loop_Pipeline.ipynb')
print('Patching complete!')
