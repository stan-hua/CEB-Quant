#!/bin/bash -l
#SBATCH --job-name=ceb_generate                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:NVIDIA_L40S:1
#SBATCH --cpus-per-task=2                 # Number of CPU cores per TASK
#SBATCH --mem=32GB
#SBATCH --tmp=8GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=10:00:00

# If you want to do it in the terminal,
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_L40S:1 --cpus-per-task=2 --mem=16G --tmp 8GB
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_H100_80GB_HBM3:1 --cpus-per-task=1 --mem=16G --tmp 8GB
# srun (command)


################################################################################
#                                 Environment                                  #
################################################################################
# Load CUDA libraries
module load gcc/12.1.0
module load cuda/12.4.1

# Load any necessary modules or activate your virtual environment here
micromamba activate fairbench
# micromamba activate quip

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
    # "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
    # "Llama-3.1-8B-Instruct-LC-RTN-W4A16"
    # "Llama-3.1-8B-Instruct-LC-RTN-W8A8"
    # "Llama-3.1-8B-Instruct-LC-RTN-W8A16"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-GPTQ-W4A16"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-GPTQ-W8A8"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-GPTQ-W8A16"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-RTN-W4A16"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-RTN-W8A8"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-RTN-W8A16"

    # "Xu-Ouyang/Llama-3.1-8B-int2-GPTQ-wikitext2"
    # "Xu-Ouyang/Meta-Llama-3.1-8B-int3-GPTQ-wikitext2"

    # "meta-llama/Llama-2-7b-chat-hf"
    # "TheBloke/Llama-2-7B-Chat-GPTQ"

    # "meta-llama/Llama-2-70b-chat-hf"

    # "meta-llama/Llama-3.1-70B"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
    # "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    # "ISTA-DASLab/Meta-Llama-3.1-70B-Instruct-AQLM-PV-2Bit-1x16"
    # "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"
    # "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8"
    # "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a16"
    # "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"
    # "Meta-Llama-3.1-70B-Instruct-LC-RTN-W4A16"
    # "Meta-Llama-3.1-70B-Instruct-LC-RTN-W4A16-KV8"
    # "Meta-Llama-3.1-70B-Instruct-LC-RTN-W8A8"
    # "Meta-Llama-3.1-70B-Instruct-LC-RTN-W8A16"
    # "Meta-Llama-3.1-70B-Instruct-LC-SmoothQuant-RTN-W4A16"
    # "Meta-Llama-3.1-70B-Instruct-LC-SmoothQuant-RTN-W8A8"
    # "Meta-Llama-3.1-70B-Instruct-LC-SmoothQuant-RTN-W8A16"

    # "mistralai/Mistral-7B-v0.3"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Ministral-8B-Instruct-2410"
    # "mistralai/Mistral-Small-Instruct-2409"
)

QUIP_MODELS=(
    # "relaxml/Llama-2-70b-chat-E8P-2Bit"
)

# List of VPTQ models to infer
VPTQ_MODELS=(
    # "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-65536-woft"
    # "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft"
    # "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k16384-0-woft"
)

# Flag to use chat template
CHAT_FLAG=True


################################################################################
#                              Perform Benchmark                               #
################################################################################
# Assign port
port=$(shuf -i 6000-9000 -n 1)
echo $port

# 1. Regular models
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    python -m ceb_benchmark generate --model_path ${MODEL_NAME} --use_chat_template $CHAT_FLAG;
done

# # 2. QuIP# models
# for MODEL_NAME in "${QUIP_MODELS[@]}"; do
#     python -m ceb_benchmark generate --model_path ${MODEL_NAME};
# done

# # 3. VPTQ models
# for MODEL_NAME in "${VPTQ_MODELS[@]}"; do
#     python -m ceb_benchmark generate --model_path ${MODEL_NAME} --model_provider "vptq";
# done