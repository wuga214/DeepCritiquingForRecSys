#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CD_explanation_tuning/ncf.csv --parameters config/CDsVinyl/ncf.yml --explanation
