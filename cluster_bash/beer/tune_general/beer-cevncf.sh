#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python tune_parameters.py --data_dir data/beer/ --save_path beer/cevncf.csv --parameters config/beer/cevncf.yml
