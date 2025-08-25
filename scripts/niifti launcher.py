import subprocess
import os

# Get project root directory (one level up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env = os.environ.copy()
env['nnUNet_raw_data_base'] = os.path.join(PROJECT_ROOT, 'output', 'nnunet_raw_data_base')
env['nnUNet_preprocessed'] = os.path.join(PROJECT_ROOT, 'output', 'nnunet_preprocessed')
env['RESULTS_FOLDER'] = os.path.join(PROJECT_ROOT, 'output', 'nnunet_results')

# План и препроцессинг
subprocess.run(['nnUNet_plan_and_preprocess', '-t', '001', '--verify_dataset_integrity'], env=env, check=True)

# Инференс
result = subprocess.run([
    'nnUNet_predict',
    '-i', os.path.join(PROJECT_ROOT, 'output', 'nnunet_raw_data_base', 'Task001_BrainTumour', 'imagesTr'),
    '-o', os.path.join(PROJECT_ROOT, 'output', 'nnunet_results'),
    '-t', '001',
    '-m', '3d_fullres',
    '-f', '0'
], env=env, check=True)

print(f"Command finished with return code {result.returncode}")

# Определи нужные переменные окружения
env = os.environ.copy()
env['nnUNet_raw_data_base'] = os.path.join(PROJECT_ROOT, 'output', 'nnunet_raw_data_base')
env['nnUNet_preprocessed'] = os.path.join(PROJECT_ROOT, 'output', 'nnunet_preprocessed')
env['RESULTS_FOLDER'] = os.path.join(PROJECT_ROOT, 'output', 'nnunet_results')

cmd = [
    'nnUNet_predict',
    '-i', os.path.join(PROJECT_ROOT, 'dataset', 'nifti_files'),  # Update path as needed
    '-o', os.path.join(PROJECT_ROOT, 'output', 'resultant_Masks'),
    '-t', '001',
    '-m', '3d_fullres',
    '-f', '0'
]

# Запуск команды с обновлённым окружением
result = subprocess.run(cmd, env=env)

print(f"Command finished with return code {result.returncode}")
