#!/usr/bin/env bash
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-ncf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-vncf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-encf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-evncf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-cencf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-cevncf-explanation.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-default-explanation.sh
