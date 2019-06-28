#!/usr/bin/env bash
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-ncf.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-vncf.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-encf.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-evncf.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-cencf.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-cevncf.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 CDsVinyl-default.sh
