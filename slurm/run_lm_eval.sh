#!/bin/bash -l
#SBATCH --job-name=ceb_lm_eval                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:2
#SBATCH --reservation=stan_gpu
# --nodelist=cn532                         # Number of nodes
# --gres=gpu:NVIDIA_L40S:1
# --gres=gpu:NVIDIA_H100_80GB_HBM3:2
#SBATCH --cpus-per-task=2                 # Number of CPU cores per TASK
#SBATCH --mem=32GB
#SBATCH --tmp=8GB
#SBATCH -o slurm/logs/slurm-lm-eval-%j.out
#SBATCH --time=28:00:00
# --begin=now+4hours

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

# Configures vLLM to avoid multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn


################################################################################
#                                  Constants                                   #
################################################################################
# Local directory containing locally saved models to run
LOCAL_MODEL_DIR="save_data/models"

# Local directory to save LM-eval results
LM_EVAL_OUTPUT_DIR="save_data/lm-eval"

# Number of GPUS to Load a Single Model (NOTE: Splits weights onto different GPUs)
NUM_GPUS_SPLIT=2

# Number of GPUs to Divide and Conquer  (NOTE: Only works if a model can run on 1 GPU)
NUM_GPUS_DISTRIBUTE=1

# HuggingFace ID
HF_ID="stan-hua"

# Models to Evaluate
MODEL_NAMES=(
    # # # # 2.0. LLaMA 3.2 1B model
    # llama3.2-1b
    # hf-llama3.2-1b-aqlm-pv-2bit-2x8
    # llama3.2-1b-instruct
    # llama3.2-1b-instruct-lc-rtn-w4a16
    # llama3.2-1b-instruct-lc-smooth-rtn-w4a16
    # llama3.2-1b-instruct-lc-rtn-w8a8
    # llama3.2-1b-instruct-lc-smooth-rtn-w8a8
    # llama3.2-1b-instruct-lc-rtn-w8a16
    # llama3.2-1b-instruct-lc-gptq-w4a16
    # llama3.2-1b-instruct-lc-smooth-gptq-w4a16
    # llama3.2-1b-instruct-awq-w4a16
    # hf-llama3.2-1b-instruct-aqlm-pv-2bit-2x8

    # # # # 2.1. LLaMA 3.2 3B model
    # llama3.2-3b
    # hf-llama3.2-3b-aqlm-pv-2bit-2x8
    # llama3.2-3b-instruct
    # llama3.2-3b-instruct-lc-rtn-w4a16
    # llama3.2-3b-instruct-lc-smooth-rtn-w4a16
    # llama3.2-3b-instruct-lc-rtn-w8a8
    # llama3.2-3b-instruct-lc-smooth-rtn-w8a8
    # llama3.2-3b-instruct-lc-rtn-w8a16
    # llama3.2-3b-instruct-lc-gptq-w4a16
    # llama3.2-3b-instruct-lc-smooth-gptq-w4a16
    # llama3.2-3b-instruct-awq-w4a16
    # hf-llama3.2-3b-instruct-aqlm-pv-2bit-2x8

    # # # 2.2. LLaMA 3.1 8B model
    # llama3.1-8b-instruct
    # llama3.1-8b-instruct-lc-rtn-w4a16
    # llama3.1-8b-instruct-lc-smooth-rtn-w4a16
    # llama3.1-8b-instruct-lc-rtn-w8a8
    # llama3.1-8b-instruct-lc-smooth-rtn-w8a8
    # llama3.1-8b-instruct-lc-rtn-w8a16
    # nm-llama3.1-8b-instruct-gptq-w4a16
    # llama3.1-8b-instruct-lc-smooth-gptq-w4a16
    # hf-llama3.1-8b-instruct-awq-4bit
    # hf-llama3.1-8b-instruct-aqlm-pv-2bit-2x8
    # hf-llama3.1-8b-instruct-aqlm-pv-1bit-1x16

    # # # 2.3. LLaMA 3.1 70B model
    llama3.1-70b
    llama3.1-70b-instruct
    llama3.1-70b-instruct-lc-rtn-w4a16
    llama3.1-70b-instruct-lc-smooth-rtn-w4a16
    llama3.1-70b-instruct-lc-rtn-w8a8
    llama3.1-70b-instruct-lc-smooth-rtn-w8a8
    llama3.1-70b-instruct-lc-rtn-w8a16
    # nm-llama3.1-70b-instruct-gptq-w4a16
    # hf-llama3.1-70b-instruct-awq-int4
    
    # # # Mistral 7B
    # mistral-v0.3-7b
    # mistral-v0.3-7b-instruct

    # # # # 2.4 Ministral 8B
    # ministral-8b-instruct
    # ministral-8b-instruct-lc-rtn-w4a16
    # ministral-8b-instruct-lc-smooth-rtn-w4a16
    # ministral-8b-instruct-lc-rtn-w8a8
    # ministral-8b-instruct-lc-smooth-rtn-w8a8
    # ministral-8b-instruct-lc-rtn-w8a16
    # ministral-8b-instruct-lc-gptq-w4a16
    # ministral-8b-instruct-lc-smooth-gptq-w4a16
    # ministral-8b-instruct-awq-w4a16

    # # # # 2.5 Mistral Small 22B
    # mistral-small-22b-instruct
    # mistral-small-22b-instruct-lc-rtn-w4a16
    # mistral-small-22b-instruct-lc-smooth-rtn-w4a16
    # mistral-small-22b-instruct-lc-rtn-w8a8
    # mistral-small-22b-instruct-lc-smooth-rtn-w8a8
    # mistral-small-22b-instruct-lc-rtn-w8a16
    # mistral-small-22b-instruct-lc-gptq-w4a16
    # mistral-small-22b-instruct-lc-smooth-gptq-w4a16
    # mistral-small-22b-instruct-awq-w4a16

    # # # # 2.6. Qwen2 7B
    # qwen2-7b
    # qwen2-7b-instruct
    # qwen2-7b-instruct-lc-rtn-w4a16
    # qwen2-7b-instruct-lc-smooth-rtn-w4a16
    # qwen2-7b-instruct-lc-rtn-w8a8
    # qwen2-7b-instruct-lc-smooth-rtn-w8a8
    # qwen2-7b-instruct-lc-rtn-w8a16
    # hf-qwen2-7b-instruct-awq-w4a16
    # hf-qwen2-7b-instruct-gptq-w4a16
    # hf-qwen2-7b-instruct-gptq-w8a16

    # 2.7. Qwen2 72B
    # qwen2-72b
    # qwen2-72b-instruct
    # qwen2-72b-instruct-lc-rtn-w4a16
    # qwen2-72b-instruct-lc-smooth-rtn-w4a16
    # qwen2-72b-instruct-lc-rtn-w8a8
    # qwen2-72b-instruct-lc-smooth-rtn-w8a8
    # qwen2-72b-instruct-lc-rtn-w8a16
    # hf-qwen2-72b-instruct-gptq-w4a16
    # hf-qwen2-72b-instruct-awq-w4a16
    # hf-qwen2-72b-instruct-aqlm-pv-2bit-1x16
    # hf-qwen2-72b-instruct-aqlm-pv-1bit-1x16

    # # 2.8. Qwen2.5 0.5B
    # qwen2.5-0.5b
    # qwen2.5-0.5b-instruct
    # qwen2.5-0.5b-instruct-awq-w4a16
    # qwen2.5-0.5b-instruct-gptq-w4a16
    # qwen2.5-0.5b-instruct-gptq-w8a16
    # qwen2.5-0.5b-instruct-lc-rtn-w4a16
    # qwen2.5-0.5b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-0.5b-instruct-lc-rtn-w8a8
    # qwen2.5-0.5b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-0.5b-instruct-lc-rtn-w8a16
    # qwen2.5-0.5b-instruct-lc-smooth-rtn-w8a16

    # # # 2.9. Qwen2.5 1.5B
    # qwen2.5-1.5b
    # qwen2.5-1.5b-instruct
    # qwen2.5-1.5b-instruct-awq-w4a16
    # qwen2.5-1.5b-instruct-gptq-w4a16
    # qwen2.5-1.5b-instruct-gptq-w8a16
    # qwen2.5-1.5b-instruct-lc-rtn-w4a16
    # qwen2.5-1.5b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-1.5b-instruct-lc-rtn-w8a8
    # qwen2.5-1.5b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-1.5b-instruct-lc-rtn-w8a16
    # qwen2.5-1.5b-instruct-lc-smooth-rtn-w8a16

    # # # Qwen2.5 3B
    # qwen2.5-3b
    # qwen2.5-3b-instruct
    # qwen2.5-3b-instruct-awq-w4a16
    # qwen2.5-3b-instruct-gptq-w4a16
    # qwen2.5-3b-instruct-gptq-w8a16
    # qwen2.5-3b-instruct-lc-rtn-w4a16
    # qwen2.5-3b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-3b-instruct-lc-rtn-w8a8
    # qwen2.5-3b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-3b-instruct-lc-rtn-w8a16
    # qwen2.5-3b-instruct-lc-smooth-rtn-w8a16

    # # # Qwen2.5 7B
    # qwen2.5-7b
    # qwen2.5-7b-instruct
    # qwen2.5-7b-instruct-awq-w4a16
    # qwen2.5-7b-instruct-gptq-w4a16
    # qwen2.5-7b-instruct-gptq-w8a16
    # qwen2.5-7b-instruct-lc-rtn-w4a16
    # qwen2.5-7b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-7b-instruct-lc-rtn-w8a8
    # qwen2.5-7b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-7b-instruct-lc-rtn-w8a16
    # qwen2.5-7b-instruct-lc-smooth-rtn-w8a16

    # # # Qwen2.5 14B
    # qwen2.5-14b
    # qwen2.5-14b-instruct
    # qwen2.5-14b-instruct-awq-w4a16
    # qwen2.5-14b-instruct-gptq-w4a16
    # qwen2.5-14b-instruct-gptq-w8a16
    # qwen2.5-14b-instruct-lc-rtn-w4a16
    # qwen2.5-14b-instruct-lc-smooth-rtn-w4a16
    # qwen2.5-14b-instruct-lc-rtn-w8a8
    # qwen2.5-14b-instruct-lc-smooth-rtn-w8a8
    # qwen2.5-14b-instruct-lc-rtn-w8a16
    # qwen2.5-14b-instruct-lc-smooth-rtn-w8a16

    # # # Qwen2.5 32B
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

    # # # # Qwen2.5 72B
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

    # # # Gemma 2B
    # gemma2-2b-instruct
    # gemma2-2b-instruct-lc-rtn-w4a16
    # gemma2-2b-instruct-lc-rtn-w8a16
    # gemma2-2b-instruct-lc-rtn-w8a8
    # gemma2-2b-instruct-lc-smooth-rtn-w4a16
    # gemma2-2b-instruct-lc-smooth-rtn-w8a16
    # gemma2-2b-instruct-lc-smooth-rtn-w8a8

    # # # # Gemma 9B
    # gemma2-9b-instruct
    # gemma2-9b-instruct-lc-rtn-w4a16
    # gemma2-9b-instruct-lc-rtn-w8a16
    # gemma2-9b-instruct-lc-rtn-w8a8
    # gemma2-9b-instruct-lc-smooth-rtn-w4a16
    # gemma2-9b-instruct-lc-smooth-rtn-w8a16
    # gemma2-9b-instruct-lc-smooth-rtn-w8a8

    # Gemma 27B
    gemma2-27b-instruct
    gemma2-27b-instruct-lc-rtn-w4a16
    gemma2-27b-instruct-lc-rtn-w8a16
    gemma2-27b-instruct-lc-rtn-w8a8
    gemma2-27b-instruct-lc-smooth-rtn-w4a16
    gemma2-27b-instruct-lc-smooth-rtn-w8a16
    gemma2-27b-instruct-lc-smooth-rtn-w8a8
)


################################################################################
#                              Perform Benchmark                               #
################################################################################
# Assign port
port=$(shuf -i 6000-9000 -n 1)
echo $port

# Specify tasks
TASKS="arc_challenge,mmlu_pro,hellaswag,piqa,lambada_openai,truthfulqa_mc1"
# Removed tasks = bigbench_multiple_choice, truthfulqa_gen

# List out valid tasks
# lm-eval --tasks list

# Run LM-Eval for each model
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    # Get model path from nickname
    MODEL_PATH=`python -m src.bin.rename_model $MODEL_NAME --reverse`
    echo "[LM-Eval] Evaluating Model: $MODEL_NAME"

    # Create model path
    # CASE 1: HuggingFace directory provided
    if echo $MODEL_PATH | grep -q "/"; then
        MODEL_PATH=$MODEL_PATH
    # CASE 2: Local model
    else
        MODEL_PATH=$LOCAL_MODEL_DIR/$MODEL_PATH
    fi

    # Create model-specific LM-eval output directory
    OUTPUT_DIR=$LM_EVAL_OUTPUT_DIR/$MODEL_NAME

    # Perform LM evaluation
    VLLM_KWARGS="pretrained=$MODEL_PATH,tensor_parallel_size=$NUM_GPUS_SPLIT,dtype=auto,gpu_memory_utilization=0.95,enforce_eager=True,data_parallel_size=$NUM_GPUS_DISTRIBUTE,max_model_len=4096,max_num_seqs=16,trust_remote_code=True"
    lm_eval --model vllm \
        --tasks $TASKS \
        --model_args $VLLM_KWARGS \
        --batch_size 1 \
        --output_path $OUTPUT_DIR \
        --use_cache $OUTPUT_DIR
done
