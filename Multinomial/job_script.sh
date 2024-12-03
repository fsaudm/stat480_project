#!/bin/bash

# Set the wallclock time
#SBATCH --time=48:00:00

# Set the number of nodes, tasks per node, and CPUs per task for the job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G

# Set the job name
#SBATCH --job-name="Multinomial_Log"

# Set the partition
#SBATCH --partition=interlagos

# Set the output and error file paths
#SBATCH --output=Multinomial_log_%j.out
#SBATCH --error=Multinomial_log_%j.err

# Set email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Notifications for job start, end, and failure
#SBATCH --mail-user=gfs3@illinois.edu

# Load modules
module load modtree/gpu
module load gcc anaconda3_gpu
module load cuda/12.2.1

# Activate the appropriate conda environment
source activate /u/gfs3/.conda/envs/BigData

# Run the Python script
python /u/gfs3/Project/DSRS/gfs3/stat480_project/Multinomial/multinomial_logreg.py