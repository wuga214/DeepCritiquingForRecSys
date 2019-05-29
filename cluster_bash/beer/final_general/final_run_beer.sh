#!/usr/bin/env bash
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-beer1.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-beer2.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-beer3.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-beer4.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-beer5.sh
