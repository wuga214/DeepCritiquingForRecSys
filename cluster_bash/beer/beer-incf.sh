#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python tune_parameters.py -d data/beer/ -n beer/incf.csv -y config/beer/incf.yml
