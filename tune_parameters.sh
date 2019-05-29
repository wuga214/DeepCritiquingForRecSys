#!/usr/bin/env bash
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/cncf.csv --parameters config/CDsVinyl/cncf.yml -gpu
python tune_parameters.py --data_dir data/beer/ --save_path beer/cncf.csv --parameters config/beer/cncf.yml -gpu
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/cvncf.csv --parameters config/CDsVinyl/cvncf.yml -gpu
python tune_parameters.py --data_dir data/beer/ --save_path beer/cvncf.csv --parameters config/beer/cvncf.yml -gpu
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/incf.csv --parameters config/CDsVinyl/incf.yml -gpu
python tune_parameters.py --data_dir data/beer/ --save_path beer/incf.csv --parameters config/beer/incf.yml -gpu
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/ivncf.csv --parameters config/CDsVinyl/ivncf.yml -gpu
python tune_parameters.py --data_dir data/beer/ --save_path beer/ivncf.csv --parameters config/beer/ivncf.yml -gpu
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/ncf.csv --parameters config/CDsVinyl/ncf.yml -gpu
python tune_parameters.py --data_dir data/beer/ --save_path beer/ncf.csv --parameters config/beer/ncf.yml -gpu
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/vncf.csv --parameters config/CDsVinyl/vncf.yml -gpu
python tune_parameters.py --data_dir data/beer/ --save_path beer/vncf.csv --parameters config/beer/vncf.yml -gpu


#python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/ncf.csv --parameters config/CDsVinyl/ncf.yml -gpu
#python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/incf.csv --parameters config/CDsVinyl/incf.yml -gpu
#python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/cncf.csv --parameters config/CDsVinyl/cncf.yml -gpu
#python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/vncf.csv --parameters config/CDsVinyl/vncf.yml -gpu
#python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/ivncf.csv --parameters config/CDsVinyl/ivncf.yml -gpu
#python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/cvncf.csv --parameters config/CDsVinyl/cvncf.yml -gpu
#python tune_parameters.py --data_dir data/beer/ --save_path beer/ncf.csv --parameters config/beer/ncf.yml -gpu
#python tune_parameters.py --data_dir data/beer/ --save_path beer/incf.csv --parameters config/beer/incf.yml -gpu
#python tune_parameters.py --data_dir data/beer/ --save_path beer/cncf.csv --parameters config/beer/cncf.yml -gpu
#python tune_parameters.py --data_dir data/beer/ --save_path beer/vncf.csv --parameters config/beer/vncf.yml -gpu
#python tune_parameters.py --data_dir data/beer/ --save_path beer/ivncf.csv --parameters config/beer/ivncf.yml -gpu
#python tune_parameters.py --data_dir data/beer/ --save_path beer/cvncf.csv --parameters config/beer/cvncf.yml -gpu

