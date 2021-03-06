#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=slurm_logs/qclassifier.vqa_%j.out
 
# a file for errors
#SBATCH --error=slurm_logs/vqa_%j.err
 
# gpus per node
#SBATCH --gpus=geforce:2
 
# number of requested nodes
#SBATCH --nodes=1
 
# memory per node
#SBATCH --mem=32GB
#SBATCH --job-name=qclassifier
#SBATCH --partition=edith
#SBATCH --time=36:00:00

# email setting
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=muchenli@cs.ubc.ca    # Where to send mail	

conda activate pytorch
python3.8 run.py --RUN='train' --VERSION='Qclassifier' --GPU="0,1" --SPLIT='train' --MODEL=q_small
