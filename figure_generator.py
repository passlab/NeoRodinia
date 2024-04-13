import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'agg' before importing pyplot
import matplotlib.pyplot as plt
import os
import shutil

# Get the current working directory
current_directory = os.getcwd()

# Get the base name of the current working directory
folder = os.path.basename(current_directory)

# Load the CSV file into a DataFrame
try:
    df = pd.read_csv(f'Execution_times_{folder}.csv')
except FileNotFoundError:
    print("Error: File 'Execution_times.csv' not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: File 'Execution_times.csv' is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: Unable to parse 'Execution_times.csv'.")
    exit(1)

# Filter out rows where Compiler_Optimization is 'nvcc_O1' or 'nvcc_O3'
df = df[~df['Compiler_Optimization'].isin(['nvcc_O1', 'nvcc_O3'])]

# Group the data by compiler and executable, and calculate the average execution time
grouped_df = df.groupby(['Compiler_Optimization', 'Executable']).mean().unstack()

# Reorder the columns to place 'serial_CPU_P0' in front of 'omp_CPU_P1'
columns = grouped_df.columns.tolist()
columns.remove(('AvgExecutionTime', 'serial_CPU_P0'))
columns.insert(columns.index(('AvgExecutionTime', 'omp_CPU_P1')), ('AvgExecutionTime', 'serial_CPU_P0'))
grouped_df = grouped_df[columns]

# Calculate standard deviation and quartiles
std_dev = grouped_df.stack().std().iloc[0]
q1 = grouped_df.stack().quantile(0.25).iloc[0]
q3 = grouped_df.stack().quantile(0.75).iloc[0]


# Set y-axis range based on standard deviation and quartiles
y_min = max(0, q1 - 1.5 * std_dev)
y_max = q3 + 1.5 * std_dev

# Plot the grouped bar chart with fixed bar width
ax = grouped_df.plot(kind='bar', figsize=(16, 6), width=0.9, cmap='tab20b')

plt.ylim(y_min, y_max)  # Set y-axis limits
plt.ylabel('Time(ms)')
plt.xlabel('Parallelization Levels')
plt.title(f'Average Execution Times Report for {folder}')
legend_labels = [col[1] for col in grouped_df.columns]
plt.legend(labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.xticks(rotation=0, ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for col in grouped_df.columns:
    col_label = col[1]  # Extract the second part of the tuple (Executable name)
    for i, p in enumerate(ax.patches):
        if i % len(grouped_df.columns) == grouped_df.columns.get_loc(col):
            height = p.get_height()
            if height != 0:  # Display label only if the height is non-zero
                label_y = min(height + 0.5, y_max - 90)  # Ensure label is within y-axis limits
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., label_y),
                        ha='center', va='center', fontsize=8, color='black', xytext=(0, 16),
                        textcoords='offset points', rotation=90)  # Rotate the label by 90 degrees and adjust y position

plt.tight_layout()
plt.savefig(f"Execution_Time_Report_{folder}.svg", format="svg")  # Save the figure as an SVG file
plt.savefig(f'Execution_Time_Report_{folder}.png')  # Save the figure as an image file

# Source file path
source_file = f"Execution_Time_Report_{folder}.svg"

# Destination directory path
destination_dir = "../Results/"

# Copy the file to the destination directory
shutil.copy(source_file, destination_dir)

plt.show()  # Display the plot
print("Execution_Time_Report.png and Execution_Time_Report.svg are generated. Done!")
