#!/usr/bin/env bash
sbatch --nodes=1 --time=72:00:00 --mem=32G --cpus=4 CDsVinyl-cncf.sh
sbatch --nodes=1 --time=72:00:00 --mem=32G --cpus=4 CDsVinyl-cvncf.sh
sbatch --nodes=1 --time=72:00:00 --mem=32G --cpus=4 CDsVinyl-default.sh
sbatch --nodes=1 --time=72:00:00 --mem=32G --cpus=4 CDsVinyl-global.sh
sbatch --nodes=1 --time=72:00:00 --mem=32G --cpus=4 CDsVinyl-incf.sh
sbatch --nodes=1 --time=72:00:00 --mem=32G --cpus=4 CDsVinyl-ivncf.sh
sbatch --nodes=1 --time=72:00:00 --mem=32G --cpus=4 CDsVinyl-ncf.sh
sbatch --nodes=1 --time=72:00:00 --mem=32G --cpus=4 CDsVinyl-vncf.sh