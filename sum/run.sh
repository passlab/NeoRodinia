#!/bin/bash

# User can specify the input_size and the number of threads here.
# If the user comment these 2 variable, the program will use the default values defined in the main function.
#INPUT_SIZE=51200

# Check if the user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable_name>"
    exit 1
fi

# Extract user input
USER_INPUT=$1

# Run the specified executable
NR_FULL_REPORT=1 ./$USER_INPUT $INPUT_SIZE
