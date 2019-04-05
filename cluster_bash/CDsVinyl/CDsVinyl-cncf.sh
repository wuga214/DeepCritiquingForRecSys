#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python tune_parameters.py -d data/CDsVinyl/ -n CDsVinyl/cncf.csv -y config/cncf.yml
