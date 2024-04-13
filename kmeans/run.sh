#!/bin/bash

# User can specify the input file here.
INPUT_FILE=../data/kmeans/819200.txt

# Check if the user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable_name>"
    exit 1
fi

# Extract user input
EXE_NAME=$1
OPTION="-i"

# Run the specified executable
NR_FULL_REPORT=1 ./$EXE_NAME $OPTION $INPUT_FILE
