#!/bin/bash -l
#SBATCH --job-name=eval_bias                    # Job name
#SBATCH --gres=gpu:NVIDIA_H100_NVL:1
# --nodelist=cn532                         # Number of nodes
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per TASK
#SBATCH --mem=24GB
#SBATCH -o slurm/logs/slurm-eval_bias-%j.out
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

module load java/17


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

ALL_MODELS=(
    # # # 2.0. LLaMA 3.2 1B model
    llama3.2-1b
    hf-llama3.2-1b-aqlm-pv-2bit-2x8
    llama3.2-1b-instruct
    llama3.2-1b-instruct-lc-rtn-w4a16
    llama3.2-1b-instruct-lc-smooth-rtn-w4a16
    llama3.2-1b-instruct-lc-rtn-w8a8
    llama3.2-1b-instruct-lc-smooth-rtn-w8a8
    llama3.2-1b-instruct-lc-rtn-w8a16
    llama3.2-1b-instruct-lc-gptq-w4a16
    llama3.2-1b-instruct-lc-smooth-gptq-w4a16
    llama3.2-1b-instruct-awq-w4a16
    hf-llama3.2-1b-instruct-aqlm-pv-2bit-2x8

    # # # 2.1. LLaMA 3.2 3B model
    llama3.2-3b
    hf-llama3.2-3b-aqlm-pv-2bit-2x8
    llama3.2-3b-instruct
    llama3.2-3b-instruct-lc-rtn-w4a16
    llama3.2-3b-instruct-lc-smooth-rtn-w4a16
    llama3.2-3b-instruct-lc-rtn-w8a8
    llama3.2-3b-instruct-lc-smooth-rtn-w8a8
    llama3.2-3b-instruct-lc-rtn-w8a16
    llama3.2-3b-instruct-lc-gptq-w4a16
    llama3.2-3b-instruct-lc-smooth-gptq-w4a16
    llama3.2-3b-instruct-awq-w4a16
    hf-llama3.2-3b-instruct-aqlm-pv-2bit-2x8

    # # # 2.2. LLaMA 3.1 8B model
    llama3.1-8b
    llama3.1-8b-instruct
    llama3.1-8b-instruct-lc-rtn-w4a16
    llama3.1-8b-instruct-lc-smooth-rtn-w4a16
    llama3.1-8b-instruct-lc-rtn-w8a8
    llama3.1-8b-instruct-lc-smooth-rtn-w8a8
    llama3.1-8b-instruct-lc-rtn-w8a16
    nm-llama3.1-8b-instruct-gptq-w4a16
    llama3.1-8b-instruct-lc-smooth-gptq-w4a16
    hf-llama3.1-8b-instruct-awq-4bit
    hf-llama3.1-8b-instruct-aqlm-pv-2bit-2x8
    hf-llama3.1-8b-instruct-aqlm-pv-1bit-1x16

    # # 2.3. LLaMA 3.1 70B model
    llama3.1-70b
    llama3.1-70b-instruct
    llama3.1-70b-instruct-lc-rtn-w4a16
    llama3.1-70b-instruct-lc-smooth-rtn-w4a16
    llama3.1-70b-instruct-lc-rtn-w8a8
    llama3.1-70b-instruct-lc-smooth-rtn-w8a8
    llama3.1-70b-instruct-lc-rtn-w8a16
    nm-llama3.1-70b-instruct-gptq-w4a16
    hf-llama3.1-70b-instruct-awq-int4
    
    # # # Mistral 7B
    # mistral-v0.3-7b
    # mistral-v0.3-7b-instruct

    # # # 2.4 Ministral 8B
    ministral-8b-instruct
    ministral-8b-instruct-lc-rtn-w4a16
    ministral-8b-instruct-lc-smooth-rtn-w4a16
    ministral-8b-instruct-lc-rtn-w8a8
    ministral-8b-instruct-lc-smooth-rtn-w8a8
    ministral-8b-instruct-lc-rtn-w8a16
    ministral-8b-instruct-lc-gptq-w4a16
    ministral-8b-instruct-lc-smooth-gptq-w4a16
    ministral-8b-instruct-awq-w4a16

    # # # 2.5 Mistral Small 22B
    mistral-small-22b-instruct
    mistral-small-22b-instruct-lc-rtn-w4a16
    mistral-small-22b-instruct-lc-smooth-rtn-w4a16
    mistral-small-22b-instruct-lc-rtn-w8a8
    mistral-small-22b-instruct-lc-smooth-rtn-w8a8
    mistral-small-22b-instruct-lc-rtn-w8a16
    mistral-small-22b-instruct-lc-gptq-w4a16
    mistral-small-22b-instruct-lc-smooth-gptq-w4a16
    mistral-small-22b-instruct-awq-w4a16

    # # # 2.6. Qwen2 7B
    qwen2-7b
    qwen2-7b-instruct
    qwen2-7b-instruct-lc-rtn-w4a16
    qwen2-7b-instruct-lc-smooth-rtn-w4a16
    qwen2-7b-instruct-lc-rtn-w8a8
    qwen2-7b-instruct-lc-smooth-rtn-w8a8
    qwen2-7b-instruct-lc-rtn-w8a16
    hf-qwen2-7b-instruct-awq-w4a16
    hf-qwen2-7b-instruct-gptq-w4a16
    hf-qwen2-7b-instruct-gptq-w8a16

    # 2.7. Qwen2 72B
    qwen2-72b
    qwen2-72b-instruct
    qwen2-72b-instruct-lc-rtn-w4a16
    qwen2-72b-instruct-lc-smooth-rtn-w4a16
    qwen2-72b-instruct-lc-rtn-w8a8
    qwen2-72b-instruct-lc-smooth-rtn-w8a8
    qwen2-72b-instruct-lc-rtn-w8a16
    hf-qwen2-72b-instruct-gptq-w4a16
    hf-qwen2-72b-instruct-awq-w4a16
    hf-qwen2-72b-instruct-aqlm-pv-2bit-1x16
    hf-qwen2-72b-instruct-aqlm-pv-1bit-1x16

    # 2.8. Qwen2.5 0.5B
    qwen2.5-0.5b
    qwen2.5-0.5b-instruct
    qwen2.5-0.5b-instruct-awq-w4a16
    qwen2.5-0.5b-instruct-gptq-w4a16
    qwen2.5-0.5b-instruct-gptq-w8a16
    qwen2.5-0.5b-instruct-lc-rtn-w4a16
    qwen2.5-0.5b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-0.5b-instruct-lc-rtn-w8a8
    qwen2.5-0.5b-instruct-lc-smooth-rtn-w8a8
    qwen2.5-0.5b-instruct-lc-rtn-w8a16
    qwen2.5-0.5b-instruct-lc-smooth-rtn-w8a16

    # # # 2.9. Qwen2.5 1.5B
    qwen2.5-1.5b
    qwen2.5-1.5b-instruct
    qwen2.5-1.5b-instruct-awq-w4a16
    qwen2.5-1.5b-instruct-gptq-w4a16
    qwen2.5-1.5b-instruct-gptq-w8a16
    qwen2.5-1.5b-instruct-lc-rtn-w4a16
    qwen2.5-1.5b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-1.5b-instruct-lc-rtn-w8a8
    qwen2.5-1.5b-instruct-lc-smooth-rtn-w8a8
    qwen2.5-1.5b-instruct-lc-rtn-w8a16
    qwen2.5-1.5b-instruct-lc-smooth-rtn-w8a16

    # # Qwen2.5 3B
    qwen2.5-3b
    qwen2.5-3b-instruct
    qwen2.5-3b-instruct-awq-w4a16
    qwen2.5-3b-instruct-gptq-w4a16
    qwen2.5-3b-instruct-gptq-w8a16
    qwen2.5-3b-instruct-lc-rtn-w4a16
    qwen2.5-3b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-3b-instruct-lc-rtn-w8a8
    qwen2.5-3b-instruct-lc-smooth-rtn-w8a8
    qwen2.5-3b-instruct-lc-rtn-w8a16
    qwen2.5-3b-instruct-lc-smooth-rtn-w8a16

    # Qwen2.5 7B
    qwen2.5-7b
    qwen2.5-7b-instruct
    qwen2.5-7b-instruct-awq-w4a16
    qwen2.5-7b-instruct-gptq-w4a16
    qwen2.5-7b-instruct-gptq-w8a16
    qwen2.5-7b-instruct-lc-rtn-w4a16
    qwen2.5-7b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-7b-instruct-lc-rtn-w8a8
    qwen2.5-7b-instruct-lc-smooth-rtn-w8a8
    qwen2.5-7b-instruct-lc-rtn-w8a16
    qwen2.5-7b-instruct-lc-smooth-rtn-w8a16

    # # # Qwen2.5 14B
    qwen2.5-14b
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

    # # Qwen2.5 32B
    qwen2.5-32b
    qwen2.5-32b-instruct
    qwen2.5-32b-instruct-awq-w4a16
    qwen2.5-32b-instruct-gptq-w4a16
    qwen2.5-32b-instruct-gptq-w8a16
    qwen2.5-32b-instruct-lc-rtn-w4a16
    qwen2.5-32b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-32b-instruct-lc-rtn-w8a8
    qwen2.5-32b-instruct-lc-smooth-rtn-w8a8
    qwen2.5-32b-instruct-lc-rtn-w8a16
    qwen2.5-32b-instruct-lc-smooth-rtn-w8a16

    # # Qwen2.5 72B
    qwen2.5-72b
    qwen2.5-72b-instruct
    qwen2.5-72b-instruct-awq-w4a16
    qwen2.5-72b-instruct-gptq-w4a16
    qwen2.5-72b-instruct-gptq-w8a16
    qwen2.5-72b-instruct-lc-rtn-w4a16
    qwen2.5-72b-instruct-lc-smooth-rtn-w4a16
    qwen2.5-72b-instruct-lc-rtn-w8a8
    qwen2.5-72b-instruct-lc-smooth-rtn-w8a8
    qwen2.5-72b-instruct-lc-rtn-w8a16
    qwen2.5-72b-instruct-lc-smooth-rtn-w8a16

    # # # Phi3 8B
    # phi3-3.8b-instruct
    # phi3-3.8b-instruct-lc-rtn-w4a16
    # phi3-3.8b-instruct-lc-rtn-w8a16
    # phi3-3.8b-instruct-lc-rtn-w8a8
    # phi3-3.8b-instruct-lc-smooth-rtn-w4a16
    # phi3-3.8b-instruct-lc-smooth-rtn-w8a16
    # phi3-3.8b-instruct-lc-smooth-rtn-w8a8

    # # # Phi3 7B
    # phi3-7b-instruct
    # phi3-7b-instruct-lc-rtn-w4a16
    # phi3-7b-instruct-lc-rtn-w8a16
    # phi3-7b-instruct-lc-rtn-w8a8
    # phi3-7b-instruct-lc-smooth-rtn-w4a16
    # phi3-7b-instruct-lc-smooth-rtn-w8a16
    # phi3-7b-instruct-lc-smooth-rtn-w8a8

    # # # Phi3 14B
    # phi3-14b-instruct
    # phi3-14b-instruct-lc-rtn-w4a16
    # phi3-14b-instruct-lc-rtn-w8a16
    # phi3-14b-instruct-lc-rtn-w8a8
    # phi3-14b-instruct-lc-smooth-rtn-w4a16
    # phi3-14b-instruct-lc-smooth-rtn-w8a16
    # phi3-14b-instruct-lc-smooth-rtn-w8a8

    # # Gemma 2B
    # gemma2-2b-instruct
    # gemma2-2b-instruct-lc-rtn-w4a16
    # gemma2-2b-instruct-lc-rtn-w8a16
    # gemma2-2b-instruct-lc-rtn-w8a8
    # gemma2-2b-instruct-lc-smooth-rtn-w4a16
    # gemma2-2b-instruct-lc-smooth-rtn-w8a16
    # gemma2-2b-instruct-lc-smooth-rtn-w8a8

    # # # Gemma 9B
    # gemma2-9b-instruct
    # gemma2-9b-instruct-lc-rtn-w4a16
    # gemma2-9b-instruct-lc-rtn-w8a16
    # gemma2-9b-instruct-lc-rtn-w8a8
    # gemma2-9b-instruct-lc-smooth-rtn-w4a16
    # gemma2-9b-instruct-lc-smooth-rtn-w8a16
    # gemma2-9b-instruct-lc-smooth-rtn-w8a8

    # # # Gemma 27B
    # gemma2-27b-instruct
    # gemma2-27b-instruct-lc-rtn-w4a16
    # gemma2-27b-instruct-lc-rtn-w8a16
    # gemma2-27b-instruct-lc-rtn-w8a8
    # gemma2-27b-instruct-lc-smooth-rtn-w4a16
    # gemma2-27b-instruct-lc-smooth-rtn-w8a16
    # gemma2-27b-instruct-lc-smooth-rtn-w8a8
)


################################################################################
#                         Generalized Text Evaluation                          #
################################################################################
# Evaluate model
for MODEL_NAME in "${ALL_MODELS[@]}"; do
    python -m scripts.benchmark bias_evaluate ${MODEL_NAME};
done
