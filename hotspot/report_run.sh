#!/bin/bash

# User can specify the inputs here.
GRID_ROWS=1024
GRID_COLS=1024
SIM_TIME=2
TEMP_FILE=../data/hotspot/temp_1024
POWER_FILE=../data/hotspot/power_1024
OUTPUT_FILE=output.out

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
    elapsed_time=$(NR_FULL_REPORT=0 ./$EXE_NAME $GRID_ROWS $GRID_COLS $SIM_TIME $TEMP_FILE $POWER_FILE $OUTPUT_FILE)
    total_time=$(echo "$total_time + $elapsed_time" | bc)
    echo "Iteration $i: $elapsed_time"
done

# Calculate average execution time
average_time=$(echo "scale=4; $total_time / $NUM_RUNS" | bc)

# Output the result (only the average execution time)
echo "$average_time"
