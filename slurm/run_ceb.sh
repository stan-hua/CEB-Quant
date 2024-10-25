#!/bin/bash -l
#SBATCH --job-name=ceb_generate                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:NVIDIA_H100_80GB_HBM3:1                      # Request one GPU
#SBATCH --cpus-per-task=2                 # Number of CPU cores per TASK
#SBATCH --mem=32GB
#SBATCH --tmp=8GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=8:00:00

# If you want to do it in the terminal,
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_L40S:1 --cpus-per-task=2 --mem=16G --tmp 8GB
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_H100_80GB_HBM3:1 --cpus-per-task=2 --mem=16G --tmp 8GB
# srun (command)


################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
micromamba activate fairbench

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn


################################################################################
#                                 Choose Model                                 #
################################################################################
# HuggingFace ID
HF_ID="stan-hua"

# Unquantized models to evaluate
# UNQ_MODELS=(
#     "meta-llama/Meta-Llama-3-8B-Instruct"
#     "mistralai/Mistral-7B-Instruct-v0.3"
#     "google/gemma-7b-it"
# )

# # Quantized models to evaluate
# Q_MODELS=(
#     "Meta-Llama-3-8B-Instruct"
#     "Mistral-7B-Instruct-v0.3"
#     "Gemma-7B-Instruct"
# )

# Chosen model
# MODEL="Meta-Llama-3-8B-Instruct"
# MODEL="Mistral-7B-Instruct-v0.3"

# Choose model
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME="${HF_ID}/${MODEL}-GPTQ-4bit"

# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# MODEL_NAME="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

MODEL_NAMES=(
    # "stan-hua/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit"
    # "stan-hua/Meta-Llama-3.1-8B-Instruct-GPTQ-4bit"
    # "stan-hua/Meta-Llama-3.1-8B-Instruct-GPTQ-3bit"
    # "stan-hua/Meta-Llama-3.1-8B-Instruct-GPTQ-2bit"
    # "stan-hua/Meta-Llama-3.1-8B-Instruct-AWQ-4bit"
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

    # "stan-hua/Meta-Llama-3.1-8B-GPTQ-8bit"
    # "stan-hua/Meta-Llama-3.1-8B-GPTQ-4bit"
    # "stan-hua/Meta-Llama-3.1-8B-GPTQ-3bit"
    # "stan-hua/Meta-Llama-3.1-8B-GPTQ-2bit"
    # "stan-hua/Meta-Llama-3.1-8B-AWQ-4bit"
    # "Xu-Ouyang/Llama-3.1-8B-int2-GPTQ-wikitext2"
    # "Xu-Ouyang/Meta-Llama-3.1-8B-int3-GPTQ-wikitext2"

    # "meta-llama/Llama-2-7b-chat-hf"
    # "TheBloke/Llama-2-7B-Chat-GPTQ"

    # "meta-llama/Llama-2-70b-chat-hf"

    # "meta-llama/Llama-3.1-70B-Instruct"
    # "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    "ISTA-DASLab/Meta-Llama-3.1-70B-Instruct-AQLM-PV-2Bit-1x16"
)


################################################################################
#                              Perform Benchmark                               #
################################################################################
# Assign port
port=$(shuf -i 6000-9000 -n 1)
echo $port

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    python -m ceb_benchmark generate --model_path ${MODEL_NAME};
done
