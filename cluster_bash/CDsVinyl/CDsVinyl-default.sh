#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python tune_parameters.py -d data/CDsVinyl/ -n CDsVinyl/default.csv -y config/default.yml
