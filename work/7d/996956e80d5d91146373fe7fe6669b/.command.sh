#!/bin/bash -ue
python3 /app/validation.py -i predictions.csv -com EuCanImage -c ECI_UC0_Seg -e Brain_Cancer_Diagnosis -p tool1 -g goldstandard_dir
