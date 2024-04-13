import os
import pandas as pd
import subprocess
import shutil

# Get the current working directory
current_directory = os.getcwd()

# Get the base name of the current working directory
folder = os.path.basename(current_directory)

current_dir = os.getcwd()
files = os.listdir(current_dir)
exec_files = [f for f in files if f.endswith('exec')]

num_run = 10
subprocess.run(["./batch_report_run.sh", str(num_run)] + exec_files)

df = pd.read_csv('execution_times.csv')

df['Compiler_Optimization'] = df['Executable'].apply(lambda x: '_'.join(x.split('_')[-3:-1]))
df['Executable'] = df['Executable'].apply(lambda x: '_'.join(x.split('_')[1:4]))

df.sort_values(by=['Compiler_Optimization', 'Executable'], inplace=True)

df.to_csv(f'Execution_times_{folder}.csv', index=False)

# Source file path
source_file = f"Execution_times_{folder}.csv"

# Destination directory path
destination_dir = "../Results/"

# Copy the file to the destination directory
shutil.copy(source_file, destination_dir)

print("CSV file generated successfully.")
