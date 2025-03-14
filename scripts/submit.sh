#!/bin/bash
#SBATCH --job-name=islic-training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --account=petep
#SBATCH --mail-user=petep@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=%x-%j.log

# Load modules
module python3.9
module load cuda/11.7
module load cudnn

# Initialize conda
source /home/petep/icis/bin/activate

export PYTHONPATH=$PYTHONPATH:/home/petep/ISIC-Cancer-Detection

cd /home/petep/ISIC-Cancer-Detection

python main.py --config configs/config.yaml