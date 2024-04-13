#!/bin/bash
# User can specify the inputs here.
GRID_ROWS=1024
GRID_COLS=1024
SIM_TIME=2
TEMP_FILE=../data/hotspot/temp_1024
POWER_FILE=../data/hotspot/power_1024
OUTPUT_FILE=output.out
# Check if the user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable_name>"
    exit 1
fi

# Extract user input
USER_INPUT=$1

# Run the specified executable
NR_FULL_REPORT=1 ./$USER_INPUT $GRID_ROWS $GRID_COLS $SIM_TIME $TEMP_FILE $POWER_FILE $OUTPUT_FILE
