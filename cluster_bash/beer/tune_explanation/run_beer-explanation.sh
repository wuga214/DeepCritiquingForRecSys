#!/usr/bin/env bash
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 beer-ncf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 beer-vncf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 beer-encf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 beer-evncf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 beer-cencf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 beer-cevncf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 beer-default-explanation.sh
