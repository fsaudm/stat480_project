#!/bin/bash

# Set the wallclock time
#SBATCH --time=48:00:00

# Set the number of nodes, tasks per node, and CPUs per task for the job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G

# Set the job name
#SBATCH --job-name="Parallel Binomial Double Lasso"

# Set the partition
#SBATCH --partition=interlagos

# Set the output and error file paths
#SBATCH --output=out_log_%j.out
#SBATCH --error=err_log_%j.err
# Set email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Notifications for job start, end, and failure
#SBATCH --mail-user=gfs3@illinois.edu

# Load modules
module load modtree/gpu
module load gcc anaconda3_gpu
module load cuda/12.2.1


# Activate the appropriate conda environment
conda init 
bash
conda activate /u/gfs3/.conda/envs/Rfarid



# Run the Python script
#Rscript /u/gfs3/Project/DSRS/gfs3/stat480_project/parallel_binomial_double_lassoV2.r
Rscript /u/gfs3/Project/DSRS/gfs3/stat480_project/parallel_binomial_double_lassoV3.r


#sbatch Rjob_script.sh

