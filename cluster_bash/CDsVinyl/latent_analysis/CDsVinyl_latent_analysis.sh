#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/DeepCritiquingForRecSys
python reproduce_latent_analysis.py --data_dir data/CDsVinyl/ --model_saved_path CDsVinyl --load_path explanation/CDsVinyl/hyper_parameters.csv --num_users_sampled 6000 --save_path CD_density/CDsVinyl_Critiquing
