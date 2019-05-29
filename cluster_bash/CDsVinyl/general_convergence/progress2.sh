#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python progress_analysis.py --data_dir data/ --dataset_name CDsVinyl --epoch 300 --save_path CD_convergence/CD_convergence_analysis2.csv --tuning_result_path CDsVinyl/
