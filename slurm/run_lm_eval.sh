#!/bin/bash -l
#SBATCH --job-name=ceb_lm_eval                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn528                         # Number of nodes
# --gres=gpu:NVIDIA_L40S:1
# --gres=gpu:NVIDIA_H100_80GB_HBM3:2
#SBATCH --cpus-per-task=2                 # Number of CPU cores per TASK
#SBATCH --mem=32GB
#SBATCH --tmp=8GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=12:00:00
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
NUM_GPUS_SPLIT=1

# Number of GPUs to Divide and Conquer  (NOTE: Only works if a model can run on 1 GPU)
NUM_GPUS_DISTRIBUTE=1

# HuggingFace ID
HF_ID="stan-hua"

# Models to Evaluate
MODEL_NAMES=(
    ############################################################################
    #                             LLaMA 3.1 8B                                 #
    ############################################################################
    # "meta-llama/Llama-3.1-8B"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    # "Llama-3.1-8B-Instruct-LC-RTN-W4A16"
    # "Llama-3.1-8B-Instruct-LC-RTN-W8A8"
    # "Llama-3.1-8B-Instruct-LC-RTN-W8A16"
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-GPTQ-W4A16"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-GPTQ-W8A8"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-GPTQ-W8A16"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-RTN-W4A16"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-RTN-W8A8"
    # "Llama-3.1-8B-Instruct-LC-SmoothQuant-RTN-W8A16"
    # "ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-2x8-hf"
    # "ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-1Bit-1x16-hf"

    ############################################################################
    #                            LLaMA 3.1 70B                                 #
    ############################################################################
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

    ############################################################################
    #                             LLaMA 3.2 1B                                 #
    ############################################################################
    # "meta-llama/Llama-3.2-1B"
    # "ISTA-DASLab/Llama-3.2-1B-AQLM-PV-2Bit-2x8"

    # "meta-llama/Llama-3.2-1B-Instruct"
    # "Llama-3.2-1B-Instruct-LC-RTN-W4A16"
    # "Llama-3.2-1B-Instruct-LC-RTN-W8A8"
    # "Llama-3.2-1B-Instruct-LC-RTN-W8A16"
    # "Llama-3.2-1B-Instruct-LC-GPTQ-W4A16"
    # "Llama-3.2-1B-Instruct-LC-SmoothQuant-GPTQ-W4A16"
    # "Llama-3.2-1B-Instruct-LC-SmoothQuant-RTN-W4A16"
    # "Llama-3.2-1B-Instruct-LC-SmoothQuant-RTN-W8A8"
    # "Llama-3.2-1B-Instruct-LC-SmoothQuant-RTN-W8A16"
    # "ISTA-DASLab/Llama-3.2-1B-Instruct-AQLM-PV-2Bit-2x8"

    ############################################################################
    #                             LLaMA 3.2 3B                                 #
    ############################################################################
    # "meta-llama/Llama-3.2-3B"
    # "ISTA-DASLab/Llama-3.2-3B-AQLM-PV-2Bit-2x8"

    # "meta-llama/Llama-3.2-3B-Instruct"
    # "Meta-Llama-3.2-3B-Instruct-LC-RTN-W4A16"
    # "Meta-Llama-3.2-3B-Instruct-LC-RTN-W8A8"
    # "Meta-Llama-3.2-3B-Instruct-LC-RTN-W8A16"
    # "Meta-Llama-3.2-3B-Instruct-LC-GPTQ-W4A16"
    # "Meta-Llama-3.2-3B-Instruct-LC-SmoothQuant-GPTQ-W4A16"
    # "Meta-Llama-3.2-3B-Instruct-LC-SmoothQuant-RTN-W4A16"
    # "Meta-Llama-3.2-3B-Instruct-LC-SmoothQuant-RTN-W8A8"
    # "Meta-Llama-3.2-3B-Instruct-LC-SmoothQuant-RTN-W8A16"
    # "ISTA-DASLab/Llama-3.2-3B-Instruct-AQLM-PV-2Bit-2x8"

    ############################################################################
    #                               Mistral                                    #
    ############################################################################
    # "mistralai/Mistral-7B-v0.3"
    # "mistralai/Mistral-7B-Instruct-v0.3"

    # 2. Ministral 8B
    # "mistralai/Ministral-8B-Instruct-2410"
    # "Ministral-8B-Instruct-2410-LC-RTN-W4A16"
    # "Ministral-8B-Instruct-2410-LC-RTN-W8A16"
    # "Ministral-8B-Instruct-2410-LC-RTN-W8A8"
    # "Ministral-8B-Instruct-2410-LC-SmoothQuant-RTN-W8A8"
    # TODO: Add GPTQ models here

    # 3. Mistral Small 22B
    # "mistralai/Mistral-Small-Instruct-2409"
    # "Mistral-Small-Instruct-2409-LC-RTN-W4A16"
    # "Mistral-Small-Instruct-2409-LC-RTN-W8A16"
    # "Mistral-Small-Instruct-2409-LC-RTN-W8A8"
    # "Mistral-Small-Instruct-2409-LC-SmoothQuant-RTN-W8A8"
    # TODO: Add GPTQ models here

    ############################################################################
    #                               Qwen2 7B                                   #
    ############################################################################
    # 1. Qwen2 7B Instruct
    # "Qwen/Qwen2-7B"
    # "Qwen/Qwen2-7B-Instruct"
    # "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
    # "Qwen/Qwen2-7B-Instruct-GPTQ-Int8"
    # "Qwen/Qwen2-7B-Instruct-AWQ"
    # "Qwen2-7B-Instruct-LC-RTN-W4A16"
    # "Qwen2-7B-Instruct-LC-RTN-W8A16"
    # "Qwen2-7B-Instruct-LC-RTN-W8A8"
    # "Qwen2-7B-Instruct-LC-SmoothQuant-RTN-W8A8"

    ############################################################################
    #                              Qwen2 72B                                   #
    ############################################################################
    # "Qwen/Qwen2-72B"
    # "Qwen/Qwen2-72B-Instruct" 
    # "Qwen/Qwen2-72B-Instruct-GPTQ-Int4"
    # "Qwen/Qwen2-72B-Instruct-GPTQ-Int8"
    # "Qwen/Qwen2-72B-Instruct-AWQ"
    # "ISTA-DASLab/Qwen2-72B-Instruct-AQLM-PV-2bit-1x16"
    # "ISTA-DASLab/Qwen2-72B-Instruct-AQLM-PV-1bit-1x16"
    # "Qwen2-72B-Instruct-LC-RTN-W4A16"
    # "Qwen2-72B-Instruct-LC-RTN-W8A16"
    # "Qwen2-72B-Instruct-LC-RTN-W8A8"
    # "Qwen2-72B-Instruct-LC-SmoothQuant-RTN-W8A8"
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
    # Create model path
    # CASE 1: HuggingFace directory provided
    if echo $MODEL_NAME | grep -q "/"; then
        MODEL_PATH=$MODEL_NAME
    # CASE 2: Local model
    else
        MODEL_PATH=$LOCAL_MODEL_DIR/$MODEL_NAME
    fi

    # Rename model to simpler nickname
    MODEL_NICKNAME=`python -m ceb_benchmark rename_model $MODEL_NAME`
    echo "[LM-Eval] Evaluating Model: $MODEL_NICKNAME"

    # Create model-specific LM-eval output directory
    OUTPUT_DIR=$LM_EVAL_OUTPUT_DIR/$MODEL_NICKNAME

    # Perform LM evaluation
    VLLM_KWARGS="pretrained='$MODEL_PATH',tensor_parallel_size=$NUM_GPUS_SPLIT,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS_DISTRIBUTE,max_model_len=4096,max_num_seqs=32"
    lm_eval --model vllm \
        --tasks $TASKS \
        --model_args $VLLM_KWARGS \
        --batch_size auto \
        --output_path $OUTPUT_DIR \
        --use_cache $OUTPUT_DIR
done
