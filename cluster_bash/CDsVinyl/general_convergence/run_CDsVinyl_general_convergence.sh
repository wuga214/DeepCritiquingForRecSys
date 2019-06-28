#!/usr/bin/env bash
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress1.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress2.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress3.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress4.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress5.sh
