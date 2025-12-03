import numpy as np
from pathlib import Path

base_path = Path('clustering_evaluation_project/phase1_dataset_unification/arrays_20250822_004929')

print('=== ESTRUCTURA DE ARRAYS ===')
for array_file in base_path.glob('*.npy'):
    try:
        arr = np.load(array_file, allow_pickle=True)
        print(f'{array_file.name}: {arr.shape} - dtype: {arr.dtype}')
        if len(arr.shape) == 2:
            print(f'  Rango valores: [{arr.min():.6f}, {arr.max():.6f}]')
            print(f'  Media: {arr.mean():.6f}, Std: {arr.std():.6f}')
        elif len(arr.shape) == 1:
            if arr.dtype.kind in ['U', 'S', 'O']:
                print(f'  Primeros 3 elementos: {arr[:3]}')
            else:
                print(f'  Rango valores: [{arr.min():.6f}, {arr.max():.6f}]')
                print(f'  Media: {arr.mean():.6f}, Std: {arr.std():.6f}')
        print()
    except Exception as e:
        print(f'Error cargando {array_file.name}: {e}')
        print()