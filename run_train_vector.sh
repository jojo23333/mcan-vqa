#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=vqa
#SBATCH --output=pytorch_job_%j.out

# Operations
echo "Job start at $(date)"
export PATH=/pkgs/anaconda3/bin:$PATH
source activate vqa 
python3 run.py --RUN='train' --VERSION='abs_small'
echo "Job end at $(date)"
