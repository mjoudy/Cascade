
import json
import sys

nb_path = '/home/mjoudy/Documents/codes/Cascade/nature_review_datasets.ipynb'

try:
    with open(nb_path, 'r') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error reading notebook: {e}")
    sys.exit(1)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'tau' in source or 'df_final' in source:
            print(f"--- Cell {i} (Execution Count: {cell.get('execution_count')}) ---")
            print(source)
            print("-" * 40)
