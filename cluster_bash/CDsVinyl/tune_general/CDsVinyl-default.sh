#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/default.csv --parameters config/default.yml
