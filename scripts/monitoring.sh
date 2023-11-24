#!/bin/bash

counter=0
start_time=$(date +%s)

while true; do
    sleep 10
    # Run the make command and save the output
    make backend_test | python scripts/create_dataframe_monitoring.py
    rm scripts/test_results.csv
    ((counter++))
    # Calculate the elapsed time
    elapsed_time=$(($(date +%s) - start_time))
    echo "Loop: $counter, Elapsed time: $elapsed_time seconds"
done