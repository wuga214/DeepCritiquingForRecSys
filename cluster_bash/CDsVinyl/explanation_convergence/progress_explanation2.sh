#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python progress_analysis.py --data_dir data/ --dataset_name CDsVinyl --epoch 300 --save_path CD_explanation_convergence/CD_convergence_analysis2.csv --explanation --tuning_result_path CD_explanation_tuning/
