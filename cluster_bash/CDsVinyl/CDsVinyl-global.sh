#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/InterpretableAutoRec
python tune_parameters.py -d data/CDsVinyl/ -n CDsVinyl/global.csv -y config/global.yml
