#!/bin/bash
#SBATCH --job-name=
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=
#SBATCH --mem=32G
#SBATCH --time==24:00:00
#SBATCH --output=log/train_%j.out
#SBATCH --error=logs/train_%j.err

module load python/3.11.11
module load cuda
module load pytorch

source ~/anaconda3/bin/activate env

cd ~/ICIS-Cancer-Detection 

python src/train.py --config config.yml --epochs 50 --batch-size 32 --learning-rate 0.001

conda deactivate