#!/bin/bash -l
#SBATCH --job-name=ceb_eval                    # Job name
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:NVIDIA_L40S:1
# --nodelist=cn532                         # Number of nodes
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per TASK
#SBATCH --mem=24GB
#SBATCH -o slurm/logs/slurm-eval_ceb-%j.out
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

# 0. Impact of Base vs. Instruct model
Q0_RESULTS_DIRS=(
    "llama3.2-1b"
    "llama3.2-1b-instruct"
    "llama3.2-3b"
    "llama3.2-3b-instruct"
    llama3.1-8b
    llama3.1-8b-instruct
    llama3.1-70b
    llama3.1-70b-instruct
    mistral-v0.3-7b
    mistral-v0.3-7b-instruct
    "qwen2-7b"
    "qwen2-7b-instruct"
    # TODO: Generate this
    # "qwen2-72b"
    # "qwen2-72b-instruct"
)

# 1. Impact of Chat Template
Q1_RESULTS_DIRS=(
    "llama3.2-1b-instruct"
    "llama3.2-1b-instruct-chat"
    "llama3.2-3b-instruct"
    "llama3.2-3b-instruct-chat"
    "llama3.1-8b-instruct"
    "llama3.1-8b-instruct-chat"
    "llama3.1-70b-instruct"
    "llama3.1-70b-instruct-chat"
    "mistral-v0.3-7b-instruct"
    "mistral-v0.3-7b-instruct-chat"
    "ministral-8b-instruct"
    "ministral-8b-instruct-chat"
    "mistral-small-22b-instruct"
    "mistral-small-22b-instruct-chat"
    "qwen2-7b-instruct"
    "qwen2-7b-instruct-chat"
    # "qwen2-72b-instruct"
    # "qwen2-72b-instruct-chat"

    # Quantized models
    "hf-llama3.1-8b-instruct-aqlm-pv-2bit-2x8"
    "hf-llama3.1-8b-instruct-aqlm-pv-2bit-2x8-chat"
    "hf-llama3.1-8b-instruct-aqlm-pv-1bit-1x16"
    "hf-llama3.1-8b-instruct-aqlm-pv-1bit-1x16-chat"
    "hf-llama3.1-70b-instruct-aqlm-pv-2bit-1x16"
    "hf-llama3.1-70b-instruct-aqlm-pv-2bit-1x16-chat"
)

# 2. Impact of RTN at different bit lengths
Q2_RESULTS_DIRS=(
    # 2.0. LLaMA 3.2 1B model
    llama3.2-1b-instruct
    llama3.2-1b-instruct-lc-rtn-w4a16
    llama3.2-1b-instruct-lc-rtn-w8a8
    llama3.2-1b-instruct-lc-rtn-w8a16
    # 2.1. LLaMA 3.2 3B model
    llama3.2-3b-instruct
    llama3.2-3b-instruct-lc-rtn-w4a16
    llama3.2-3b-instruct-lc-rtn-w8a8
    llama3.2-3b-instruct-lc-rtn-w8a16
    # 2.2. LLaMA 3.1 8B model
    llama3.1-8b-instruct
    llama3.1-8b-instruct-lc-rtn-w4a16
    llama3.1-8b-instruct-lc-rtn-w8a8
    llama3.1-8b-instruct-lc-rtn-w8a16
    # 2.3. LLaMA 3.1 70B model
    llama3.1-70b-instruct
    llama3.1-70b-instruct-lc-rtn-w4a16
    llama3.1-70b-instruct-lc-rtn-w8a8
    llama3.1-70b-instruct-lc-rtn-w8a16
    # 2.4 Ministral 8B
    "ministral-8b-instruct"
    "ministral-8b-instruct-lc-rtn-w4a16"
    "ministral-8b-instruct-lc-rtn-w8a8"
    "ministral-8b-instruct-lc-rtn-w8a16"
    # 2.5 Mistral Small 22B
    "mistral-small-22b-instruct"
    "mistral-small-22b-instruct-lc-rtn-w4a16"
    "mistral-small-22b-instruct-lc-rtn-w8a8"
    "mistral-small-22b-instruct-lc-rtn-w8a16"
    # 2.6. Qwen2 7B
    "qwen2-7b-instruct"
    "qwen2-7b-instruct-lc-rtn-w4a16"
    "qwen2-7b-instruct-lc-rtn-w8a16"
    "qwen2-7b-instruct-lc-rtn-w8a8"
    # # 2.7. Qwen2 72B
    # # "qwen2-72b-instruct"
    # # "qwen2-72b-instruct-lc-rtn-w4a16"
    # # "qwen2-72b-instruct-lc-rtn-w8a16"
    # # "qwen2-72b-instruct-lc-rtn-w8a8"
)

# 3. Comparison of RTN vs. GPTQ vs. AWQ W4A16
Q3_RESULTS_DIRS=(
    # 3.0. LLaMA 3.2 1B model
    llama3.2-1b-instruct
    llama3.2-1b-instruct-lc-rtn-w4a16
    llama3.2-1b-instruct-lc-gptq-w4a16
    llama3.2-1b-instruct-awq-w4a16
    # 3.1. LLaMA 3.2 3B model
    llama3.2-3b-instruct
    llama3.2-3b-instruct-lc-rtn-w4a16
    llama3.2-3b-instruct-lc-gptq-w4a16
    llama3.2-3b-instruct-awq-w4a16

    # 3.1. LLaMA 3.1 8B
    llama3.1-8b-instruct
    llama3.1-8b-instruct-lc-rtn-w4a16
    nm-llama3.1-8b-instruct-gptq-w4a16
    hf-llama3.1-8b-instruct-awq-4bit
    # 3.2. LLaMA 3.1 70B
    llama3.1-70b-instruct
    llama3.1-70b-instruct-lc-rtn-w4a16
    nm-llama3.1-70b-instruct-gptq-w4a16
    hf-llama3.1-70b-instruct-awq-int4

    # 3.3. Ministral 8B
    ministral-8b-instruct
    ministral-8b-instruct-lc-rtn-w4a16
    ministral-8b-instruct-lc-gptq-w4a16
    ministral-8b-instruct-awq-w4a16

    # 3.4. Mistral Small 22B
    mistral-small-22b-instruct
    mistral-small-22b-instruct-lc-rtn-w4a16
    mistral-small-22b-instruct-lc-gptq-w4a16
    mistral-small-22b-instruct-awq-w4a16

    # 3.5. Qwen2 7B
    "qwen2-7b-instruct"
    "qwen2-7b-instruct-lc-rtn-w4a16"
    "hf-qwen2-7b-instruct-gptq-w4a16"
    "hf-qwen2-7b-instruct-awq-w4a16"
    # 3.6. Qwen2 72B
    # TODO: Uncomment after generating
    # "qwen2-72b-instruct"
    # "qwen2-72b-instruct-lc-rtn-w4a16"
    # "hf-qwen2-72b-instruct-gptq-w4a16"
    # "hf-qwen2-72b-instruct-awq-w4a16"
)

# 4. Comparison against sub-4 bit Quantization
Q4_RESULTS_DIRS=(
    # 4.0. LLaMA 3.2 1B model
    llama3.2-1b
    hf-llama3.2-1b-aqlm-pv-2bit-2x8
    # 4.1. LLaMA 3.2 1B (Instruct) model
    llama3.2-1b-instruct
    llama3.2-1b-instruct-lc-gptq-w4a16
    hf-llama3.2-1b-instruct-aqlm-pv-2bit-2x8
    # 4.2. LLaMA 3.2 3B model
    llama3.2-3b
    hf-llama3.2-3b-aqlm-pv-2bit-2x8
    # 4.3. LLaMA 3.2 3B (Instruct) model
    llama3.2-3b-instruct
    llama3.2-3b-instruct-lc-gptq-w4a16
    hf-llama3.2-3b-instruct-aqlm-pv-2bit-2x8

    # 4.4. LLaMA 3.1 8B model
    "llama3.1-8b-instruct"
    nm-llama3.1-8b-instruct-gptq-w4a16
    "hf-llama3.1-8b-instruct-aqlm-pv-2bit-2x8"
    "hf-llama3.1-8b-instruct-aqlm-pv-1bit-1x16"

    # 4.5. LLaMA 3.1 70B model
    llama3.1-70b-instruct
    nm-llama3.1-70b-instruct-gptq-w4a16
    hf-llama3.1-70b-instruct-aqlm-pv-2bit-1x16

    # 4.6. Qwen2 72B
    # "qwen2-72b-instruct"
    # "hf-qwen2-72b-instruct-gptq-w4a16"
    # "hf-qwen2-72b-instruct-aqlm-pv-2bit-1x16",
    # "hf-qwen2-72b-instruct-aqlm-pv-1bit-1x16",

    # # TODO: Consider adding QuIP#
    # # hf-llama3.1-70b-instruct-vptq-2bit
    # # hf-llama3.1-70b-instruct-vptq-1.75bit
)

# 5. Impact of Outlier Smoothening
Q5_RESULTS_DIRS=(
    # 5.0. LLaMA 3.2 1B model (RTN W4A16)
    llama3.2-1b-instruct-lc-rtn-w4a16
    llama3.2-1b-instruct-lc-smooth-rtn-w4a16
    # 5.1. LLaMA 3.2 1B model (RTN W8A8)
    llama3.2-1b-instruct-lc-rtn-w8a8
    llama3.2-1b-instruct-lc-smooth-rtn-w8a8

    # 5.2. LLaMA 3.2 3B model (RTN W4A16)
    llama3.2-3b-instruct-lc-rtn-w4a16
    llama3.2-3b-instruct-lc-smooth-rtn-w4a16
    # 5.3. LLaMA 3.2 3B model (RTN W8A8)
    llama3.2-3b-instruct-lc-rtn-w8a8
    llama3.2-3b-instruct-lc-smooth-rtn-w8a8

    # 5.4. LLaMA 8B model (RTN W4A16)
    llama3.1-8b-instruct-lc-rtn-w4a16
    llama3.1-8b-instruct-lc-smooth-rtn-w4a16
    # 5.5. LLaMA 8B model (RTN W8A8)
    llama3.1-8b-instruct-lc-rtn-w8a8
    llama3.1-8b-instruct-lc-smooth-rtn-w8a8
    # TODO: Consider uncommenting
    # LLaMA 8B model (RTN W8A16)
    # llama3.1-8b-instruct-lc-rtn-w8a16
    # llama3.1-8b-instruct-lc-smooth-rtn-w8a16

    # 5.6. LLaMA 70B model (RTN W4A16)
    llama3.1-70b-instruct-lc-rtn-w4a16
    llama3.1-70b-instruct-lc-smooth-rtn-w4a16
    # 5.7. LLaMA 70B model (RTN W8A8)
    llama3.1-70b-instruct-lc-rtn-w8a8
    llama3.1-70b-instruct-lc-smooth-rtn-w8a8
    # TODO: Consider uncommenting
    # LLaMA 70B model (RTN W8A16)
    # llama3.1-70b-instruct-lc-rtn-w8a16
    # llama3.1-70b-instruct-lc-smooth-rtn-w8a16

    # 5.8. Ministral 8B model (RTN W4A16)
    ministral-8b-instruct-lc-rtn-w4a16
    ministral-8b-instruct-lc-smooth-rtn-w4a16
    # 5.9. Ministral 8B model (RTN W8A8)
    ministral-8b-instruct-lc-rtn-w8a8
    ministral-8b-instruct-lc-smooth-rtn-w8a8

    # Mistral Small 22B model (RTN W4A16)
    mistral-small-22b-instruct-lc-rtn-w4a16
    mistral-small-22b-instruct-lc-smooth-rtn-w4a16
    # Mistral Small 22B model (RTN W8A8)
    mistral-small-22b-instruct-lc-rtn-w8a8
    mistral-small-22b-instruct-lc-smooth-rtn-w8a8

    # Qwen2 7B model (RTN W4A16)
    "qwen2-7b-instruct-lc-rtn-w4a16"
    "qwen2-7b-instruct-lc-smooth-rtn-w4a16"
    # Qwen2 7B model (RTN W8A8)
    "qwen2-7b-instruct-lc-rtn-w8a8"
    "qwen2-7b-instruct-lc-smooth-rtn-w8a8"

    # # Qwen2 72B model (RTN W8A8)
    # TODO: Uncomment after generating
    # "qwen2-72b-instruct-lc-rtn-w8a8"
    # "qwen2-72b-instruct-lc-smooth-rtn-w8a8"

    # GPTQ W4A16 Models
    # LLaMA 3.2 1B
    "llama3.2-1b-instruct-lc-gptq-w4a16"
    "llama3.2-1b-instruct-lc-smooth-gptq-w4a16"
    # LLaMA 3.2 3B
    "llama3.2-3b-instruct-lc-gptq-w4a16"
    "llama3.2-3b-instruct-lc-smooth-gptq-w4a16"
    # LLaMA 3.1 8B
    "nm-llama3.1-8b-instruct-gptq-w4a16"
    "llama3.1-8b-instruct-lc-smooth-gptq-w4a16"
    # Ministral 8B
    "ministral-8b-instruct-lc-gptq-w4a16"
    "ministral-8b-instruct-lc-smooth-gptq-w4a16"
    # Mistral Small 22B
    "mistral-small-22b-instruct-lc-gptq-w4a16"
    "mistral-small-22b-instruct-lc-smooth-gptq-w4a16"
)

# 6. Impact of Quantizing KV Cache
Q6_RESULTS_DIRS=(
    # # 6.1. 70B model
    # llama3.1-70b-instruct
    # llama3.1-70b-instruct-lc-rtn-w4a16
    # llama3.1-70b-instruct-lc-rtn-w4a16kv8
)

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
    
    # # Mistral 7B
    mistral-v0.3-7b
    mistral-v0.3-7b-instruct

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

    # # 2.9. Qwen2.5 1.5B
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

    # # Qwen2.5 7B
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

    # # Qwen2.5 14B
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

    # # Phi3 8B
    phi3-3.8b-instruct
    phi3-3.8b-instruct-lc-rtn-w4a16
    phi3-3.8b-instruct-lc-rtn-w8a16
    phi3-3.8b-instruct-lc-rtn-w8a8
    phi3-3.8b-instruct-lc-smooth-rtn-w4a16
    phi3-3.8b-instruct-lc-smooth-rtn-w8a16
    phi3-3.8b-instruct-lc-smooth-rtn-w8a8

    # # Phi3 7B
    phi3-7b-instruct
    phi3-7b-instruct-lc-rtn-w4a16
    phi3-7b-instruct-lc-rtn-w8a16
    phi3-7b-instruct-lc-rtn-w8a8
    phi3-7b-instruct-lc-smooth-rtn-w4a16
    phi3-7b-instruct-lc-smooth-rtn-w8a16
    phi3-7b-instruct-lc-smooth-rtn-w8a8

    # # Phi3 14B
    phi3-14b-instruct
    phi3-14b-instruct-lc-rtn-w4a16
    phi3-14b-instruct-lc-rtn-w8a16
    phi3-14b-instruct-lc-rtn-w8a8
    phi3-14b-instruct-lc-smooth-rtn-w4a16
    phi3-14b-instruct-lc-smooth-rtn-w8a16
    phi3-14b-instruct-lc-smooth-rtn-w8a8

    # Gemma 2B
    gemma2-2b-instruct
    gemma2-2b-instruct-lc-rtn-w4a16
    gemma2-2b-instruct-lc-rtn-w8a16
    gemma2-2b-instruct-lc-rtn-w8a8
    gemma2-2b-instruct-lc-smooth-rtn-w4a16
    gemma2-2b-instruct-lc-smooth-rtn-w8a16
    gemma2-2b-instruct-lc-smooth-rtn-w8a8

    # # Gemma 9B
    gemma2-9b-instruct
    gemma2-9b-instruct-lc-rtn-w4a16
    gemma2-9b-instruct-lc-rtn-w8a16
    gemma2-9b-instruct-lc-rtn-w8a8
    gemma2-9b-instruct-lc-smooth-rtn-w4a16
    gemma2-9b-instruct-lc-smooth-rtn-w8a16
    gemma2-9b-instruct-lc-smooth-rtn-w8a8

    # # Gemma 27B
    gemma2-27b-instruct
    gemma2-27b-instruct-lc-rtn-w4a16
    gemma2-27b-instruct-lc-rtn-w8a16
    gemma2-27b-instruct-lc-rtn-w8a8
    gemma2-27b-instruct-lc-smooth-rtn-w4a16
    gemma2-27b-instruct-lc-smooth-rtn-w8a16
    gemma2-27b-instruct-lc-smooth-rtn-w8a8
)

# Questions to Evaluate
RESULTS_DIRS=(
    "${Q0_RESULTS_DIRS[@]}"
    "${Q1_RESULTS_DIRS[@]}"
    "${Q2_RESULTS_DIRS[@]}"
    "${Q3_RESULTS_DIRS[@]}"
    "${Q4_RESULTS_DIRS[@]}"
    "${Q5_RESULTS_DIRS[@]}"
    # "${Q6_RESULTS_DIRS[@]}"
)

# Evaluator Choice
EVALUATOR="prometheus"     # chatgpt or prometheus or atla
export JUDGE_PROMPT_VER="4"
# If ChatGPT evaluator, OpenAI model to use as a judge
OPENAI_MODEL='gpt-4o-2024-08-06'

# NOTE: Force parallel=1 is useful if Prometheus-Eval was already called on
#       all required queries, and now you only need to compute metrics. Turning
#       on allows you to do multi-proc on metric computation.
export FORCE_PARALLEL=0

# Bias Type to Evaluate
BIAS_TYPE="all"
TASK_TYPE="indirect"

# Directory to store comparisons
DIR_COMPARISONS="save_data/metrics_comparisons/$EVALUATOR"

# Flag to filter out harmful prompts
FILTER_KWARGS=""
# TODO: Uncomment below when filter harmful questions
# FILTER_KWARGS="{is_harmful:True}"
# DIR_COMPARISONS="$DIR_COMPARISONS/harmful"
# TODO: Uncomment below when filter not harmful questions
# FILTER_KWARGS="{is_harmful:False}"
# DIR_COMPARISONS="$DIR_COMPARISONS/not_harmful"

################################################################################
#                                  Evaluation                                  #
################################################################################
# Print models missing evaluations
# python -m ceb_benchmark find_unfinished --generation --pattern "*" --filter_models "${Q0_RESULTS_DIRS[@]}"
# python -m ceb_benchmark find_unfinished --evaluation --pattern $EVALUATOR --filter_models "${Q0_RESULTS_DIRS[@]}"

# Evaluate model generations
for RESULT_DIR in "${ALL_MODELS[@]}"; do
    python -m ceb_benchmark evaluate --results_dir ${RESULT_DIR} --evaluator_choice ${EVALUATOR} --openai_model ${OPENAI_MODEL} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --overwrite;
done

# Evaluate model generations
# for RESULT_DIR in "${RESULTS_DIRS[@]}"; do
#     python -m ceb_benchmark evaluate --results_dir ${RESULT_DIR} --evaluator_choice ${EVALUATOR} --openai_model ${OPENAI_MODEL} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --overwrite;
# done

# Format list of result directories in format expected by Fire
# TODO: Model comparisons after adding Qwen 72B = [7, 12, 24]
# python -m ceb_benchmark compare ${Q0_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/base_vs_instruct" --model_comparisons 6 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
# python -m ceb_benchmark compare ${Q1_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/nonchat_vs_chat" --model_comparisons 11 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
# python -m ceb_benchmark compare ${Q2_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/rtn_at_different_bits" --model_comparisons 21 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
# python -m ceb_benchmark compare ${Q3_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/w4a16_quantizers" --model_comparisons 21 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
# python -m ceb_benchmark compare ${Q4_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/sub_w4_quantizers" --model_comparisons 14 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
# python -m ceb_benchmark compare ${Q5_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/outlier_smoothing" --model_comparisons 19 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
# # python -m ceb_benchmark compare ${Q6_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/kv_cache_quantizer" --model_comparisons 2 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS

# Accumulate all results
# COMPARISONS=(
#     "base_vs_instruct" "nonchat_vs_chat" "rtn_at_different_bits" "w4a16_quantizers"
#     "sub_w4_quantizers" "outlier_smoothing"
#     # "kv_cache_quantizer"
# )
# python -m ceb_benchmark format_comparisons ${COMPARISONS[@]} --save_dir $DIR_COMPARISONS