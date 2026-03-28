import json

def patch_nb(target_col, file_path):
    with open('01_Baseline_and_EDA.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                if 'file_path=' in line:
                    cell['source'][i] = line.replace("'simulated:fraud'", f"'{file_path}'")
                if 'target_col=' in line:
                    cell['source'][i] = line.replace("'is_fraud'", f"'{target_col}'")
    with open('01_Baseline_and_EDA.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

patch_nb('default', 'generic_customer_data.csv')
print("Notebook parms updated.")
