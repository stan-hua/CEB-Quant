#!/bin/bash -l
#SBATCH --job-name=ceb_eval                    # Job name
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn527                         # Number of nodes
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per TASK
#SBATCH --mem=16GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=12:00:00

# If you want to do it in the terminal,
# --begin=now+2hours
# NOTE: cn515, cn519-526
# salloc --gpus=1 --nodelist=cn527 --mem=32G --tmp 8GB
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_L40S:1 --cpus-per-task=2 --mem=16G --tmp 8GB
# salloc --job-name=ceb --nodes=1 --gres=gpu:NVIDIA_H100_80GB_HBM3:1 --cpus-per-task=1 --mem=16G --tmp 8GB
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
    # 0.1. 8B model
    llama3.1-8b
    llama3.1-8b-instruct
    # 0.2. 70B model
    llama3.1-70b
    llama3.1-70b-instruct
    # 0.3. Mistral 7B
    mistral-v0.3-7b
    mistral-v0.3-7b-instruct
    # 0.4. Qwen2 7B
    "qwen2-7b"
    "qwen2-7b-instruct"
    # 0.5. Qwen2 72B
    "qwen2-72b"
    "qwen2-72b-instruct"
)

# 1. Impact of Chat Template
Q1_RESULTS_DIRS=(
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

    # TODO: Run Qwen models with chat template
)

# 2. Impact of RTN at different bit lengths
Q2_RESULTS_DIRS=(
    # 2.1. LLaMA 3.1 8B model
    llama3.1-8b-instruct
    llama3.1-8b-instruct-lc-rtn-w4a16
    llama3.1-8b-instruct-lc-rtn-w8a8
    llama3.1-8b-instruct-lc-rtn-w8a16
    # 2.2. LLaMA 3.1 70B model
    llama3.1-70b-instruct
    llama3.1-70b-instruct-lc-rtn-w4a16
    llama3.1-70b-instruct-lc-rtn-w8a8
    llama3.1-70b-instruct-lc-rtn-w8a16
    # 2.2 Ministral 8B
    "ministral-8b-instruct-lc-rtn-w4a16"
    "ministral-8b-instruct-lc-rtn-w8a16"
    "ministral-8b-instruct-lc-rtn-w8a8"
    # 2.3. Qwen2 7B
    "qwen2-7b-instruct-lc-rtn-w4a16"
    "qwen2-7b-instruct-lc-rtn-w8a16"
    "qwen2-7b-instruct-lc-rtn-w8a8"
    # 2.4. Qwen2 72B
    "qwen2-72b-instruct-lc-rtn-w4a16"
    "qwen2-72b-instruct-lc-rtn-w8a16"
    "qwen2-72b-instruct-lc-rtn-w8a8"
)

# 3. Comparison of RTN vs. GPTQ vs. AWQ W4A16
Q3_RESULTS_DIRS=(
    # 3.1. LLaMA 3.1 8B model
    llama3.1-8b-instruct
    llama3.1-8b-instruct-lc-rtn-w4a16
    nm-llama3.1-8b-instruct-gptq-w4a16
    hf-llama3.1-8b-instruct-awq-4bit
    # 3.2. LLaMA 3.1 70B model
    llama3.1-70b-instruct
    llama3.1-70b-instruct-lc-rtn-w4a16
    nm-llama3.1-70b-instruct-gptq-w4a16
    hf-llama3.1-70b-instruct-awq-int4
    # TODO: Add Mistral here after GPTQ quantizing Mistral models


    # 3.3. Qwen2 7B model
    "qwen2-7b-instruct"
    "qwen2-7b-instruct-lc-rtn-w4a16"
    "hf-qwen2-7b-instruct-gptq-w4a16"
    "hf-qwen2-7b-instruct-awq-w4a16"
    # 3.4. Qwen2 72B model
    "qwen2-72b-instruct"
    "qwen2-72b-instruct-lc-rtn-w4a16"
    "hf-qwen2-72b-instruct-gptq-w4a16"
    "hf-qwen2-72b-instruct-awq-w4a16"
)

# 4. Comparison against sub-4 bit Quantization
Q4_RESULTS_DIRS=(
    # 4.1. LLaMA 3.1 8B model
    "llama3.1-8b-instruct"
    nm-llama3.1-8b-instruct-gptq-w4a16
    "hf-llama3.1-8b-instruct-aqlm-pv-2bit-2x8"
    "hf-llama3.1-8b-instruct-aqlm-pv-1bit-1x16"

    # 4.2. LLaMA 3.1 70B model
    llama3.1-70b-instruct
    nm-llama3.1-70b-instruct-gptq-w4a16
    hf-llama3.1-70b-instruct-aqlm-pv-2bit-1x16

    # 4.3. Qwen2 72B
    "qwen2-7b-instruct"
    "hf-qwen2-7b-instruct-gptq-w4a16"
    "hf-qwen2-72b-instruct-aqlm-pv-2bit-1x16",
    "hf-qwen2-72b-instruct-aqlm-pv-1bit-1x16",

    # TODO: Consider adding QuIP#
    # hf-llama3.1-70b-instruct-vptq-2bit
    # hf-llama3.1-70b-instruct-vptq-1.75bit
)

# 5. Impact of Outlier Smoothening
Q5_RESULTS_DIRS=(
    # 5.1. 8B model (RTN W4A16)
    llama3.1-8b-instruct-lc-rtn-w4a16
    llama3.1-8b-instruct-lc-smooth-rtn-w4a16
    # 5.2. 8B model (RTN W8A8)
    llama3.1-8b-instruct-lc-rtn-w8a8
    llama3.1-8b-instruct-lc-smooth-rtn-w8a8
    # TODO: Consider uncommenting
    # 5.3. 8B model (RTN W8A16)
    # llama3.1-8b-instruct-lc-rtn-w8a16
    # llama3.1-8b-instruct-lc-smooth-rtn-w8a16

    # 5.4. 70B model (RTN W4A16)
    llama3.1-70b-instruct-lc-rtn-w4a16
    llama3.1-70b-instruct-lc-smooth-rtn-w4a16
    # 5.5. 70B model (RTN W8A8)
    llama3.1-70b-instruct-lc-rtn-w8a8
    llama3.1-70b-instruct-lc-smooth-rtn-w8a8
    # TODO: Consider uncommenting
    # 5.6. 70B model (RTN W8A16)
    # llama3.1-70b-instruct-lc-rtn-w8a16
    # llama3.1-70b-instruct-lc-smooth-rtn-w8a16

    # Qwen2 7B model (RTN W8A8)
    "qwen2-7b-instruct-lc-rtn-w8a8"
    "qwen2-7b-instruct-lc-smooth-rtn-w8a8"
    # Qwen2 72B model (RTN W8A8)
    "qwen2-72b-instruct-lc-rtn-w8a8"
    "qwen2-72b-instruct-lc-smooth-rtn-w8a8"

    # 5.7. 8B model (GPTQ W4A16)
    nm-llama3.1-8b-instruct-gptq-w4a16
    llama3.1-8b-instruct-lc-smooth-gptq-w4a16
    # 5.8. 8B model (GPTQ W8A8)
    nm-llama3.1-8b-instruct-gptq-w8a8
    llama3.1-8b-instruct-lc-smooth-gptq-w8a8
    # TODO: Consider uncommenting
    # 5.9. 8B model (GPTQ W8A16)
    # nm-llama3.1-8b-instruct-gptq-w8a16
    # llama3.1-8b-instruct-lc-smooth-gptq-w8a16
)

# 6. Impact of Quantizing KV Cache
Q6_RESULTS_DIRS=(
    # 6.1. 70B model
    llama3.1-70b-instruct
    llama3.1-70b-instruct-lc-rtn-w4a16
    llama3.1-70b-instruct-lc-rtn-w4a16kv8
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
EVALUATOR="prometheus"     # chatgpt or prometheus
# If ChatGPT evaluator, OpenAI model to use as a judge
OPENAI_MODEL='gpt-4o-2024-08-06'

# Bias Type to Evaluate
BIAS_TYPE="all"
TASK_TYPE="indirect"

# Directory to store comparisons
# DIR_COMPARISONS="metrics_comparisons"
DIR_COMPARISONS="metrics_comparisons/prometheus"

# Flag to filter out harmful prompts
FILTER_KWARGS=""
# TODO: Uncomment below when filter harmful is true
# DIR_COMPARISONS="metrics_comparisons/harmful"

################################################################################
#                                  Evaluation                                  #
################################################################################
for RESULT_DIR in "${RESULTS_DIRS[@]}"; do
    python -m ceb_benchmark evaluate --results_dir ${RESULT_DIR} --evaluator_choice ${EVALUATOR} --openai_model ${OPENAI_MODEL} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --overwrite;
done

# Format list of result directories in format expected by Fire
python -m ceb_benchmark compare ${Q0_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/base_vs_instruct" --model_comparisons 5 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
python -m ceb_benchmark compare ${Q1_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/nonchat_vs_chat" --model_comparisons 5 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
python -m ceb_benchmark compare ${Q2_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/rtn_at_different_bits" --model_comparisons 6 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
python -m ceb_benchmark compare ${Q3_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/w4a16_quantizers" --model_comparisons 24 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
python -m ceb_benchmark compare ${Q4_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/sub_w4_quantizers" --model_comparisons 5 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
python -m ceb_benchmark compare ${Q5_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/outlier_smoothing" --model_comparisons 9 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS
python -m ceb_benchmark compare ${Q6_RESULTS_DIRS[@]} --save_dir "$DIR_COMPARISONS/kv_cache_quantizer" --model_comparisons 2 --evaluator_choice ${EVALUATOR} --bias_type $BIAS_TYPE --task_type $TASK_TYPE --filter_kwargs=$FILTER_KWARGS

# Accumulate all results
COMPARISONS=(
    "base_vs_instruct" "nonchat_vs_chat" "rtn_at_different_bits" "w4a16_quantizers"
    "sub_w4_quantizers" "outlier_smoothing" "kv_cache_quantizer"
)
python -m ceb_benchmark format_comparisons ${COMPARISONS[@]} --save_dir $DIR_COMPARISONS