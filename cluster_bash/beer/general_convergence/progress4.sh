#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python progress_analysis.py --data_dir data/ --dataset_name beer --epoch 300 --save_path beer_convergence/beer_convergence_analysis4.csv --tuning_result_path beer/
