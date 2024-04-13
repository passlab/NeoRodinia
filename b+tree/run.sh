#!/bin/bash

# User can specify the input file and the command file here.
# To test 'k' option, comment out line 7 and uncomment line 8.

INPUT_FILE=../data/b+tree/mil.txt
COMMAND_FILE=../data/b+tree/command_j.txt
#COMMAND_FILE=../data/b+tree/command_k.txt

# Check if the user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable_name>"
    exit 1
fi

# Extract user input
USER_INPUT=$1

# Run the specified executable
NR_FULL_REPORT=1 ./$USER_INPUT $INPUT_FILE $COMMAND_FILE
