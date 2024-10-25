#!/bin/bash -l
#SBATCH --job-name=ceb_eval                    # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --cpus-per-task=6                 # Number of CPU cores per TASK
#SBATCH --mem=8GB
#SBATCH -o slurm/logs/slurm-%j.out
#SBATCH --time=3:00:00

# If you want to do it in the terminal,
# salloc --job-name=ceb --nodes=1 --cpus-per-task=6 --mem=6G
# srun (command)


################################################################################
#                                 Environment                                  #
################################################################################
# Load any necessary modules or activate your virtual environment here
micromamba activate fairbench


################################################################################
#                                Set Constants                                 #
################################################################################
port=$(shuf -i 6000-9000 -n 1)
echo $port

RESULTS_DIRS=(
    # "./generation_results/hf-llama3.1-8b-instruct-gptq-4bit"
    # "./generation_results/llama3.1-8b-instruct"
    # "./generation_results/llama3.1-8b-instruct-gptq-8bit"
    # "./generation_results/llama3.1-8b-instruct-gptq-4bit"
    # "./generation_results/llama3.1-8b-instruct-gptq-2bit"
    # "./generation_results/llama3.1-8b-gptq-8bit"
    # "./generation_results/llama3.1-8b-gptq-4bit"
    # "./generation_results/llama3.1-8b-gptq-2bit"

    # "./generation_results/llama3.1-70b-instruct"
    # "./generation_results/hf-llama3.1-70b-instruct-gptq-int4"
    # "./generation_results/hf-llama3.1-70b-instruct-awq-int4"
    "./generation_results/hf-llama3.1-70b-instruct-aqlm-pv-2bit-1x16"
    # "./generation_results/"
)
OPENAI_MODEL='gpt-4o-2024-08-06'


################################################################################
#                                  Evaluation                                  #
################################################################################
for RESULT_DIR in "${RESULTS_DIRS[@]}"; do
    python -m ceb_benchmark evaluate --results_dir ${RESULT_DIR} --openai_model ${OPENAI_MODEL};
done
