#!/usr/bin/env bash
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-CDsVinyl1.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-CDsVinyl2.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-CDsVinyl3.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-CDsVinyl4.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 final-CDsVinyl5.sh
