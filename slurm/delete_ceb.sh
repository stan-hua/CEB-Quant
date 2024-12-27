#!/bin/bash -l
#SBATCH --job-name=ceb_delete                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=2                 # Number of CPU cores per TASK
#SBATCH --mem=4GB
#SBATCH --tmp=1GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=10:00:00


################################################################################
#                                 Environment                                  #
################################################################################
# Load CUDA libraries
module load gcc/12.1.0
module load cuda/12.4.1

# Load any necessary modules or activate your virtual environment here
micromamba activate fairbench

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn


################################################################################
#                                 Delete Data                                  #
################################################################################
python -m ceb_benchmark delete --dataset_regex "CEB-Continuation-T" --file_regex "age.json" --inference --evaluation
python -m ceb_benchmark delete --dataset_regex "CEB-Continuation-T" --file_regex "gender.json" --inference --evaluation
python -m ceb_benchmark delete --dataset_regex "CEB-Continuation-T" --file_regex "race.json" --inference --evaluation
python -m ceb_benchmark delete --dataset_regex "CEB-Continuation-T" --file_regex "religion.json" --inference --evaluation
