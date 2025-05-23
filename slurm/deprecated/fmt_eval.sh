#!/bin/bash -l
#SBATCH --job-name=chatgpt_eval                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=2                 # Number of CPU cores per TASK
#SBATCH --mem=8GB
#SBATCH --tmp=1GB
#SBATCH -o slurm/logs/slurm-chatgpt_eval-%j.out
#SBATCH --time=12:00:00

################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
micromamba activate fairbench

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn


################################################################################
#                                Run Evaluation                                #
################################################################################
# Use ChatGPT to annotate 300 samples in format similar to Prometheus/Atla
# srun python -m scripts.analysis add_chatgpt_annotations


MODEL_NAMES=(
    qwen2.5-14b-instruct
    qwen2.5-14b-instruct-awq-w4a16
    qwen2.5-14b-instruct-gptq-w4a16
    qwen2.5-14b-instruct-gptq-w8a16
    qwen2.5-14b-instruct-lc-rtn-w4a16
    qwen2.5-14b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-14b-instruct-lc-rtn-w8a8
    qwen2.5-14b-instruct-lc-smooth-rtn-w8a8
    qwen2.5-14b-instruct-lc-rtn-w8a16
    qwen2.5-14b-instruct-lc-smooth-rtn-w8a16
)
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    python -m scripts.analysis fmt_bias_eval --model_name ${MODEL_NAME};
done