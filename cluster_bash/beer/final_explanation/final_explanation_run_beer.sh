#!/usr/bin/env bash
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-explanation-beer1.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-explanation-beer2.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-explanation-beer3.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-explanation-beer4.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-explanation-beer5.sh
