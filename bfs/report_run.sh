#!/bin/bash

# User can specify the input file and the number of threads here.
# If the user comment these 2 variable, the program will use the default values defined in the main function.
# INPUT_FILE=../data/bfs/graph4096.txt

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_runs> <exename>"
    exit 1
fi

# Extract user inputs
NUM_RUNS=$1
EXE_NAME=$2

# Check if the executable exists
if [ ! -x "$EXE_NAME" ]; then
    echo "Error: Executable '$EXE_NAME' not found or not executable."
    exit 1
fi

# Run the executable multiple times
total_time=0
for ((i=1; i<=$NUM_RUNS; i++)); do
    echo "Run $i of $NUM_RUNS..."
    elapsed_time=$(NR_FULL_REPORT=0 ./$EXE_NAME $INPUT_FILE)
    total_time=$(echo "$total_time + $elapsed_time" | bc)
    echo "Iteration $i: $elapsed_time"
done

# Calculate average execution time
average_time=$(echo "scale=4; $total_time / $NUM_RUNS" | bc)

# Output the result (only the average execution time)
echo "$average_time"
