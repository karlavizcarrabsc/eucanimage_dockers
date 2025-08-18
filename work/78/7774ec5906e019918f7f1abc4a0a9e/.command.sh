#!/bin/bash -ue
python3 /app/compute_metrics.py -i predictions.csv -c ECI_UC0_Seg -e Brain_Cancer_Diagnosis -g gt.csv -p tool1 -com EuCanImage -o "assessment_results.json"
