#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python reproduce_general_results.py --data_dir data/beer/ --tuning_result_path beer --save_path beer_final/beer_final_result1.csv
