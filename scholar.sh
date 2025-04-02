#!/bin/bash
# FILENAME:  scholar

#SBATCH --export=ALL          # Export your current environment settings to the job environment
#SBATCH -A gpu                # Account name
#SBATCH --ntasks=1            # Number of MPI ranks per node (one rank per GPU)
#SBATCH --cpus-per-task=4     # Number of CPU cores per MPI rank (change this if needed)
#SBATCH --gres=gpu:1          # Use one GPU
#SBATCH --mem-per-cpu=2G      # Required memory per GPU (specify how many GB)
#SBATCH --time=1:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J hmc                # Job name
#SBATCH -o slurm_logs/%j      # Name of stdout output file

module load conda
conda activate /home/sohn31/.conda/envs/cent7/2024.02-py311/CS587

python model/setup.py