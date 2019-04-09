#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python reproduce_explanation_results.py -d data/beer/ -l explanation/beer/hyper_parameters.csv -s explanation/beer/final_explanation3.csv
