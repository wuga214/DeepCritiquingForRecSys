#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python reproduce_critiquing.py --data_dir data/beer/ --model_saved_path beer --load_path explanation/beer/hyper_parameters.csv --num_users_sampled 1000 --save_path beer_fmap/beer_Critiquing_3
