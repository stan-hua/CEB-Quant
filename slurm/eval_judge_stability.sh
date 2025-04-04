#!/bin/bash -l
#SBATCH --job-name=judge_stability                    # Job name
#SBATCH --gres=gpu:NVIDIA_L40S:1
# --nodelist=cn532                         # Number of nodes
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per TASK
#SBATCH --mem=24GB
#SBATCH -o slurm/logs/slurm-judge_stability-%j.out
#SBATCH --time=24:00:00
# --begin=now+10minutes

# If you want to do it in the terminal,
# --begin=now+2hours
# NOTE: cn515, cn519-526
# salloc --gpus=1 --nodelist=cn527 --mem=32G --tmp 8GB
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_L40S:1 --cpus-per-task=4 --mem=32G --tmp 8GB
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_H100_NVL:1 --cpus-per-task=4 --mem=32G
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_H100_80GB_HBM3:1 --mem=32G --tmp 8GB
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

# Evaluator Choice
export EVALUATOR="prometheus"     # chatgpt or prometheus or atla
export JUDGE_PROMPT_VER="4"
export SYSTEM_PROMPT_TYPE="no_sys_prompt"


################################################################################
#                                  Evaluation                                  #
################################################################################
python -m scripts.sup_evaluation test_judge_stability --evaluator_choice ${EVALUATOR};