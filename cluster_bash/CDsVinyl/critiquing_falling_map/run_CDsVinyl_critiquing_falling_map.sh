#!/usr/bin/env bash
sbatch --nodes=1 --time=4:00:00 --mem=32G --cpus=4 --gres=gpu:1 critiquing_falling_map1.sh
sbatch --nodes=1 --time=4:00:00 --mem=32G --cpus=4 --gres=gpu:1 critiquing_falling_map2.sh
sbatch --nodes=1 --time=4:00:00 --mem=32G --cpus=4 --gres=gpu:1 critiquing_falling_map3.sh
sbatch --nodes=1 --time=4:00:00 --mem=32G --cpus=4 --gres=gpu:1 critiquing_falling_map4.sh
sbatch --nodes=1 --time=4:00:00 --mem=32G --cpus=4 --gres=gpu:1 critiquing_falling_map5.sh
