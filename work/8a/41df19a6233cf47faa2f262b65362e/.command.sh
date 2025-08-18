#!/bin/bash -ue
python3 /app/validation.py -i predictions.csv -com EuCanImage -c ECI_UC7_Class -e PCR_Prediction -p tool1 -g gt.csv
