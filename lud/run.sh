#!/bin/bash

# Check if the user provided an argument
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <executable_name> [<input_file>]"
    echo "If input_file is not provided, the script will use the default option to generate an input matrix internally with a specific size."
    exit 1
fi

# Extract user input
USER_INPUT=$1
OPTION=${2:-"-s"}  # Default to "-s" if no option is provided
MATRIX_SIZE=800    # Default matrix size
INPUT_FILE=${3:-"../data/lud/512.dat"}  # Default input file

# Run the specified executable with the provided option
if [ "$OPTION" == "-s" ]; then
    NR_FULL_REPORT=1 ./$USER_INPUT $OPTION $MATRIX_SIZE
elif [ "$OPTION" == "-i" ]; then
    NR_FULL_REPORT=1 ./$USER_INPUT $OPTION $INPUT_FILE
else
    echo "Invalid option: $OPTION"
    exit 1
fi
