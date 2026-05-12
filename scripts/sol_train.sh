#!/bin/bash
#SBATCH --job-name=AFC_DRL
#SBATCH --output=logs/afc_%j.out
#SBATCH --error=logs/afc_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8        
#SBATCH --mem=64G
#SBATCH --partition=general        


"""MODIFY THIS .SH TO LAUNCH YOUR TRAINING JOB ON HPC CLUSTERS, NOTE THAT IT IS CURRENTLY MADE FOR ASU'S SOL SUPERCOMPUTER."""

module load mamba                  
conda activate afc-drl             

cd $SLURM_SUBMIT_DIR
mkdir -p logs results/training results/tensorboard

# DOLFINx distributes FEM matrix assembly across all MPI ranks automatically.
# The RL loop runs on rank 0 and CFD solves benefit from the parallel assembly.
mpirun -n 8 python scripts/train_headless.py \
    --airfoil    airfoils/naca001234.dat       \
    --alpha      25                            \
    --checkpoint checkpoints/naca0012_alpha25  \
    --save_path  results/training/learned_policy \
    --tb_log     results/tensorboard           \
    --n_episodes 500


