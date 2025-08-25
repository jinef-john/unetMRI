import subprocess
import os


env = os.environ.copy()
env['nnUNet_raw_data_base'] = r'E:\MRI_LOWMEM\nnunet_raw_data_base'
env['nnUNet_preprocessed'] = r'E:\MRI_LOWMEM\nnunet_preprocessed'
env['RESULTS_FOLDER'] = r'E:\MRI_LOWMEM\nnunet_results'

# План и препроцессинг
subprocess.run(['nnUNet_plan_and_preprocess', '-t', '001', '--verify_dataset_integrity'], env=env, check=True)

# Инференс
result = subprocess.run([
    'nnUNet_predict',
    '-i', r'E:\MRI_LOWMEM\nnunet_raw_data_base\Task001_BrainTumour\imagesTr',
    '-o', r'E:\MRI_LOWMEM\nnunet_results',
    '-t', '001',
    '-m', '3d_fullres',
    '-f', '0'
], env=env, check=True)

print(f"Command finished with return code {result.returncode}")

# Определи нужные переменные окружения
env = os.environ.copy()
env['nnUNet_raw_data_base'] = r'E:\MRI_LOWMEM\nnunet_raw_data_base'
env['nnUNet_preprocessed'] = r'E:\MRI_LOWMEM\nnunet_preprocessed'
env['RESULTS_FOLDER'] = r'E:\MRI_LOWMEM\nnunet_results'

cmd = [
    'nnUNet_predict',
    '-i', r'E:/MRI_LOWMEM/train_nifti',
    '-o', r'E:/MRI_LOWMEM/resultant_Masks',
    '-t', '001',
    '-m', '3d_fullres',
    '-f', '0'
]

# Запуск команды с обновлённым окружением
result = subprocess.run(cmd, env=env)

print(f"Command finished with return code {result.returncode}")
