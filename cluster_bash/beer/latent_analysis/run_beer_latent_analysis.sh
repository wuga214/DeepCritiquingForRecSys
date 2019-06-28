#!/usr/bin/env bash
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 beer_latent_analysis.sh
