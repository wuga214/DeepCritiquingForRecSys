#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python reproduce_general_results.py -d data/beer/ -p beer -s final_general_results10.csv
