#!/bin/bash -l
#SBATCH --job-name=paper                    # Job name
# --gres=gpu:NVIDIA_L40S:1
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=12                 # Number of CPU cores per TASK
#SBATCH --mem=16GB
#SBATCH -o slurm/logs/slurm-paper-%j.out
#SBATCH --time=24:00:00
# --begin=now+10minutes

# If you want to do it in the terminal,
# salloc --nodes=1 --cpus-per-task=8 --mem=16G
# srun (command)

################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
micromamba activate fairbench

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

################################################################################
#                                Set Constants                                 #
################################################################################
port=$(shuf -i 6000-9000 -n 1)
echo $port

# System prompt type ("no_sys_prompt", "really_1x", "really_2x", "really_3x", "really_4x")
export SYSTEM_PROMPT_TYPE="no_sys_prompt"

################################################################################
#                                  Evaluation                                  #
################################################################################
DATASET_NAMES=(
    "BBQ"
    # "BiasLens-Choices"
    # "BiasLens-YesNo"
    # "IAT"
    # "SocialStigmaQA"
    # "StereoSet-Intersentence"
    # "StereoSet-Intrasentence"
)

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    srun python -m scripts.sup_evaluation analyze_discrim_dataset $DATASET_NAME;
done

# DiscrimEval
# srun python -m scripts.sup_evaluation analyze_de
