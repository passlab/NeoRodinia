#!/bin/bash

####################################################################################################################################
# This Bash script is designed to automate the process of running a specified executable file.
# By default, it generates an input matrix internally with a specific size, but you can also specify a custom input file by modifying the `OPTION` and `INPUT_FILE` variables within the script.
# After running the executable multiple times, the script calculates the average execution time and prints it to the console.
# |------------------------------------------------------------------------------------------------------------------------|
# | Variable       | Description                                                                                           |
# |----------------|-------------------------------------------------------------------------------------------------------|
# | OPTION         | - Default: "-s"                                                                                       |
# |                |   Indicates that the script generates an input matrix internally with a specific size (MATRIX_SIZE).  |
# |                |                                                                                                       |
# |                | - Option: "-f"                                                                                        |
# |                |   Allows users to specify a custom input file.                                                        |
# |                |   When selected, the script uses the input file specified in the INPUT_FILE variable.                 |
# |----------------|-------------------------------------------------------------------------------------------------------|
# | INPUT_FILE     | Specifies the path to the custom input file.                                                          |
# |                | Used only when OPTION is set to "-f".                                                                 |
# |                | Default: "../data/gaussian/matrix16.txt". Users can modify it to point to their desired input file.   |
# |------------------------------------------------------------------------------------------------------------------------|

# Check if the user provided an argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <executable_name>"
    echo "To specify the input file, please adjust the OPTION variable to '-f' and update the INPUT_FILE variable to match your desired input file. "
    echo "If input_file is not provided, the script will use the default option to generate an input matrix internally with a specific size."
    exit 1
fi

# Extract user input
USER_INPUT=$1
OPTION="-s"  # Default to "-s", if you wish to specify the input file, change it to "-f".
MATRIX_SIZE=16    # Default matrix size
INPUT_FILE="../data/gaussian/matrix16.txt"  # Default input file

# Run the specified executable with the provided option
if [ "$OPTION" == "-s" ]; then
    NR_FULL_REPORT=1 ./$USER_INPUT $OPTION $MATRIX_SIZE
elif [ "$OPTION" == "-f" ]; then
    NR_FULL_REPORT=1 ./$USER_INPUT $OPTION $INPUT_FILE
else
    echo "Invalid option: $OPTION"
    exit 1
fi
