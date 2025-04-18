#!/bin/bash -l
# FILENAME: run.sh

#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=20:00
#SBATCH -A standby

#SBATCH --error=%x-%J-train.err
#SBATCH --output=%x-%J-train.out

module load conda
conda activate /scratch/gilbreth/lstonest/envs/inn-paper

python ../inn_mnist.py