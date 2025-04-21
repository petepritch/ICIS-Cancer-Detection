#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --partition=gpu
#SBATCH --account=eecs545w25_class
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:05:00
#SBATCH --output=gpu-test-%j.log

nvidia-smi
