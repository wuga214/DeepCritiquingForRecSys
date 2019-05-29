#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CD_explanation_tuning/cevncf.csv --parameters config/CDsVinyl/cevncf.yml --explanation
