#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable_name>"
    exit 1
fi

# User can uncomment this line and change the input size.
#INPUT_SIZE=65536

# Extract user input
USER_INPUT=$1

# Run the specified executable
NR_FULL_REPORT=1 ./$USER_INPUT $INPUT_SIZE
