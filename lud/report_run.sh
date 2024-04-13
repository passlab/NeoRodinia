#!/bin/bash

####################################################################################################################################
# This Bash script is designed to automate the process of running a specified executable file multiple times with varying inputs and to calculate the average execution time across all runs.
# To use this script, you need to provide two arguments when running it from the command line.
# The first argument should be the number of times you want to run the executable (`<num_runs>`), and the second argument should be the name of the executable file (`<executable_name>`).
# The script will then execute the specified executable for the specified number of runs.
# By default, it generates an input matrix internally with a specific size, but you can also specify a custom input file by modifying the `OPTION` and `INPUT_FILE` variables within the script.
# After running the executable multiple times, the script calculates the average execution time and prints it to the console.
# |------------------------------------------------------------------------------------------------------------------------|
# | Variable       | Description                                                                                           |
# |----------------|-------------------------------------------------------------------------------------------------------|
# | OPTION         | - Default: "-s"                                                                                       |
# |                |   Indicates that the script generates an input matrix internally with a specific size (MATRIX_SIZE).  |
# |                |                                                                                                       |
# |                | - Option: "-i"                                                                                        |
# |                |   Allows users to specify a custom input file.                                                        |
# |                |   When selected, the script uses the input file specified in the INPUT_FILE variable.                 |
# |----------------|-------------------------------------------------------------------------------------------------------|
# | INPUT_FILE     | Specifies the path to the custom input file.                                                          |
# |                | Used only when OPTION is set to "-i".                                                                 |
# |                | Default: "../data/lud/512.dat". Users can modify it to point to their desired input file.             |
# |------------------------------------------------------------------------------------------------------------------------|

# Check if the user provided the correct number of arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <num_runs> <executable_name>"
    echo "If input_file is not provided, the script will use the default option to generate an input matrix internally with a specific size."
    echo "To specify the input file, please adjust the OPTION variable to '-i' and update the INPUT_FILE variable to match your desired input file. "
    exit 1
fi

# Extract user input
NUM_RUNS=$1
EXE_NAME=$2

#These options are set as defaults. If the user wishes to specify input files, please make changes here.
OPTION="-s"  # Default to "-s", if you wish to specify the input file, change it to "-i".
MATRIX_SIZE=800    # Default matrix size
INPUT_FILE="../data/lud/512.dat"  # Default input file


# Check if the executable exists
if [ ! -x "$EXE_NAME" ]; then
    echo "Error: Executable '$EXE_NAME' not found or not executable."
    exit 1
fi

# Run the executable multiple times
total_time=0
if [ "$OPTION" == "-s" ]; then
    for ((i=1; i<=$NUM_RUNS; i++)); do
        echo "Run $i of $NUM_RUNS..."
        elapsed_time=$(NR_FULL_REPORT=0 ./$EXE_NAME $OPTION $MATRIX_SIZE)
        total_time=$(echo "$total_time + $elapsed_time" | bc)
        echo "Iteration $i: $elapsed_time"
    done
elif [ "$OPTION" == "-i" ]; then
    for ((i=1; i<=$NUM_RUNS; i++)); do
        echo "Run $i of $NUM_RUNS..."
        elapsed_time=$(NR_FULL_REPORT=0 ./$EXE_NAME $OPTION $INPUT_FILE)
        total_time=$(echo "$total_time + $elapsed_time" | bc)
        echo "Iteration $i: $elapsed_time"
    done
else
    echo "Invalid option: $OPTION"
    exit 1
fi

# Calculate average execution time
average_time=$(echo "scale=4; $total_time / $NUM_RUNS" | bc)

# Output the result (only the average execution time)
echo "$average_time"
