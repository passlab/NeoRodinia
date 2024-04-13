#!/bin/bash

K1=10
K2=20
D=256
N=65536
CHUNKSIZE=65536
CLUSTERSIZE=1000
INFILE="none"
OUTFILE="output.log"
NPROC=1

# Check if the user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable_name>"
    exit 1
fi

# Extract user input
USER_INPUT=$1

# Run the specified executable
NR_FULL_REPORT=1 ./$USER_INPUT $K1 $K2 $D $N $CHUNKSIZE $CLUSTERSIZE $INFILE $OUTFILE $NPROC
