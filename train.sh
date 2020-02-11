#!/bin/bash
#SBATCH -o train.%j.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J danmuSentiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

python train.py