#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python reproduce_explanation_results.py --data_dir data/CDsVinyl/ --load_path explanation/CDsVinyl/hyper_parameters.csv --save_path CD_final_explanation/CD_final_explanation1.csv
