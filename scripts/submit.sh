#!/bin/bash
#SBATCH --job-name=isic-training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-user=petep@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log

# Load modules
module python3.9.12
module load cuda/11.6.2
module load cudnn

# Initialize conda
source /home/petep/isic/bin/activate

export PYTHONPATH=$PYTHONPATH:/home/petep/ICIS-Cancer-Detection

cd /home/petep/ICIS-Cancer-Detection

python src/main.py --config configs/config.yaml

deactivate
