#!/bin/bash
# The interpreter used to execute the script


#SBATCH --job-name=FakeNewsDetection
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --account=eecs487f23_class
#SBATCH --partition=gpu
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=6000m
#SBATCH --gpus=a100

module load python3.10-anaconda/2023.03
module load cuda/12.3.0
eval "$(conda shell.bash hook)"
conda activate fake-news-detection

python3 training.py