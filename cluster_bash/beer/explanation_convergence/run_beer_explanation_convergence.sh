#!/usr/bin/env bash
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress_explanation1.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress_explanation2.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress_explanation3.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress_explanation4.sh
sbatch --nodes=1 --time=36:00:00 --mem=32G --cpus=4 --gres=gpu:1 progress_explanation5.sh
