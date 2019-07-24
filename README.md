Deep Language-based Critiquing for Recommender Systems
====================================================================
![](https://img.shields.io/badge/linux-ubuntu-red.svg)
![](https://img.shields.io/badge/Mac-OS-red.svg)

![](https://img.shields.io/badge/cuda-10.0-green.svg)
![](https://img.shields.io/badge/python-2.7-green.svg)
![](https://img.shields.io/badge/python-3.6-green.svg)

![](https://img.shields.io/badge/cython-0.29-blue.svg)
![](https://img.shields.io/badge/fbpca-1.0-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.0.0-blue.svg)
![](https://img.shields.io/badge/numpy-1.15.2-blue.svg)
![](https://img.shields.io/badge/pandas-0.23.3-blue.svg)
![](https://img.shields.io/badge/pyyaml-4.1-blue.svg)
![](https://img.shields.io/badge/scipy-1.1.0-blue.svg)
![](https://img.shields.io/badge/seaborn-0.9.0-blue.svg)
![](https://img.shields.io/badge/sklearn-0.20.1-blue.svg)
![](https://img.shields.io/badge/tensorflow-1.12.0-blue.svg)
![](https://img.shields.io/badge/tqdm-4.28.1-blue.svg)


If you are interested in building up your research on this work, please cite:
```
@inproceedings{recsys19,
  author    = {Ga Wu and Kai Luo and Scott Sanner and Harold Soh},
  title     = {Deep Language-based Critiquing for Recommender Systems},
  booktitle = {Proceedings of the 13th International {ACM} Conference on Recommender Systems {(RecSys-19)}},
  address   = {Copenhagen, Denmark},
  year      = {2019}
}
```

# Author Affiliate
<p align="center">
<a href="https://www.utoronto.ca//"><img src="https://github.com/wuga214/NCE_Projected_LRec/blob/master/logos/U-of-T-logo.svg" height="80"></a> | 
<a href="https://vectorinstitute.ai/"><img src="https://github.com/wuga214/NCE_Projected_LRec/blob/master/logos/vectorlogo.svg" height="80"></a> | 
<a href="http://nus.edu.sg/"><img src="https://github.com/wuga214/DeepCritiquingForRecSys/blob/refactor/logos/NUS_Logo.svg"  height="80"></a>
</p>

# Algorithm Implemented
1. Critiquable and Explainable Variational Neural Collaborative Filtering (CE-VNCF)
2. Critiquable and Explainable Neural Collaborative Filtering (CE-NCF)
3. Explainable Variational Neural Collaborative Filtering (E-VNCF)
4. Explainable Neural Collaborative Filtering (E-NCF)
5. Variational Neural Collaborative Filtering (VNCF)
6. Neural Collaborative Filtering (NCF)

# Dataset
1. Amazon CDs&Vinyl,
2. Beer Advocate,

We don't have rights to release the datasets. Please ask permission from Professor [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/).

Please refer to the `preprocess` folder for preprocessing raw datasets steps.

# Keyphrase
Keyphrases we used are not necessarily the best. If you are interested in how we extracted those keyphrases, please refer to the `preprocess` folder. If you are interested in what keyphrases we extracted, please refer to the `data` folder.

# Critiquing Demo
Please refer to IPython Notebook `Critiquing Demo.ipynb` for critiquing demo.

# Example Commands

### General Recommendation and Explanation Single Run
```
python general_main.py --data_dir data/CDsVinyl/ --epoch 300 --rank 200 --lambda 0.0001 --learning_rate 0.0001 --model CE-VNCF --topk 50 --disable_validation
```

### Dataset Resplit
Resplit data into three datasets: one for train, one for validation, one for test.
```
python dataset_split.py --data_dir data/CDsVinyl/
```

Please check out the `cluster_bash` folder for all commands details. Below are only example commands.

### General Recommendation Hyper-parameter Tuning
```
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl/cevncf.csv --parameters config/CDsVinyl/cevncf.yml
```

### Reproduce Final General Recommendation Performance
```
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl --save_path CD_final/CD_final_result.csv
```

### Explanation Prediction Performance Hyper-parameter Tuning
```
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CD_explanation_tuning/cevncf.csv --parameters config/CDsVinyl/cevncf.yml --explanation
```

### Reproduce Final Explanation Prediction Performance
Find the best hyperparameter set for each model from tuning results and put them in folder `tables/explanation/beer/hyperparameters.csv` and `tables/explanation/CDsVinyl/hyperparameters.csv`. Then run the following command.
```
python reproduce_explanation_results.py --data_dir data/CDsVinyl/ --load_path explanation/CDsVinyl/hyper_parameters.csv --save_path explanation/CDsVinyl/CD_final_explanation.csv
```

### Reproduce Critiquing
```
python reproduce_critiquing.py --data_dir data/beer/ --model_saved_path beer --load_path explanation/beer/hyper_parameters.csv --num_users_sampled 1000 --save_path beer_fmap/beer_Critiquing
```

### Note
We expect the reproduced results will have negligible difference due to values used in hyper-parameter sets.

For baselines we used, please refer to [Noise Contrastive Estimation Projected Linear Recommender(NCE-PLRec)](https://github.com/wuga214/NCE_Projected_LRec).
