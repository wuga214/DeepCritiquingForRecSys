#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CD_explanation_tuning/encf.csv --parameters config/CDsVinyl/encf.yml --explanation
