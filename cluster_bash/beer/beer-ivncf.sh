#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python tune_parameters.py -d data/beer/ -n beer/ivncf.csv -y config/beer/ivncf.yml
