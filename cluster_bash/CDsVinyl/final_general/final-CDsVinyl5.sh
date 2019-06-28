#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl --save_path CD_final/CD_final_result5.csv
