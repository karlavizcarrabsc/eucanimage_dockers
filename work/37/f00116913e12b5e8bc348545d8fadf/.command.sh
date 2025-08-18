#!/bin/bash -ue
python3 /app/compute_metrics.py -i predictions.csv -c ECI_UC7_Class -e PCR_Prediction -g goldstandard_dir -p tool1 -com EuCanImage -o "assessment_results.json"
