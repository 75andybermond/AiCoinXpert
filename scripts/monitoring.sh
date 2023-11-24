#!/bin/bash

while true; do
    # Run the make command and save the output
    make backend_test | python scripts/create_dataframe_monitoring.py
    rm test_results.csv
    sleep 30
done
