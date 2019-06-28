#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python reproduce_explanation_results.py --data_dir data/beer/ --load_path explanation/beer/hyper_parameters.csv --save_path beer_final_explanation/beer_final_explanation2.csv
