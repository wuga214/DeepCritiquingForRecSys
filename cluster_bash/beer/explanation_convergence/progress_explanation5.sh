#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python progress_analysis.py --data_dir data/ --dataset_name beer --epoch 300 --save_path beer_explanation_convergence/beer_convergence_analysis5.csv --explanation --tuning_result_path beer_explanation_tuning/
