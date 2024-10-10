# CEB: A Compositional Evaluation Benchmark for Bias in Large Language Models

![The framework of CEB.](framework.png)

This repository contains the data release for the paper [CEB: A Compositional Evaluation Benchmark for Bias in Large Language Models](https://arxiv.org/pdf/2407.02408).

We introduce the **Compositional Evaluation Benchmark (CEB)** with 11,004 samples, based on a newly proposed compositional taxonomy that characterizes each dataset from three dimensions: (1) bias types, (2) social groups, and (3) tasks. Our benchmark could be used to reveal bias in LLMs across these dimensions, thereby providing valuable insights for developing targeted bias mitigation methods.

## Dataset

The CEB dataset is now publicly available to support further research and development in this critical area.

**[Dataset Files]**: ```./data```

**[HuggingFace Dataset Link]**: [CEB Dataset](https://huggingface.co/datasets/Song-SW/CEB)

**[Dataset Statistics]**:

| **Dataset**            | **Task Type**   | **Bias Type**   | **Age** | **Gender** | **Race** | **Religion** | **Size** |
|------------------------|-----------------|-----------------|---------|------------|----------|--------------|----------|
| CEB-Recognition-S      | Recognition     | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Selection-S        | Selection       | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Continuation-S     | Continuation    | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Conversation-S     | Conversation    | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Recognition-T      | Recognition     | Toxicity        | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Selection-T        | Selection       | Toxicity        | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Continuation-T     | Continuation    | Toxicity        | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Conversation-T     | Conversation    | Toxicity        | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Adult              | Classification  | Stereotyping    | No      | Yes        | Yes      | No           | 500      |
| CEB-Credit             | Classification  | Stereotyping    | Yes     | Yes        | No       | No           | 500      |
| CEB-Jigsaw             | Classification  | Toxicity        | No      | Yes        | Yes      | Yes          | 500      |
| CEB-WB-Recognition     | Recognition     | Stereotyping    | No      | Yes        | No       | No           | 792      |
| CEB-WB-Selection       | Selection       | Stereotyping    | No      | Yes        | No       | No           | 792      |
| CEB-SS-Recognition     | Recognition     | Stereotyping    | No      | Yes        | Yes      | Yes          | 960      |
| CEB-SS-Selection       | Selection       | Stereotyping    | No      | Yes        | Yes      | Yes          | 960      |
| CEB-RB-Recognition     | Recognition     | Stereotyping    | No      | Yes        | Yes      | Yes          | 1000     |
| CEB-RB-Selection       | Selection       | Stereotyping    | No      | Yes        | Yes      | Yes          | 1000     |
| CEB-CP-Recognition     | Recognition     | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-CP-Selection       | Selection       | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |


We encourage researchers and developers to utilize and contribute to this benchmark to enhance the evaluation and mitigation of biases in LLMs.


## Setup
```
# Install all the necessary packages
pip install envs/requirements.txt
```

## Configuration

Please add your OpenAI API key and Perspective API key to your shell config file
```
# Add to ~/.bashrc file (or to your .envrc)
echo 'export OPENAI_KEY="[ENTER HERE]"' >> ~/.bashrc
echo 'export PERSPECTIVE_KEY="[ENTER HERE]"' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

For modifying any models or changing any prompts, please refer to `./src/config/config.py`

## CEB-Benchmark Evaluation

0. If using custom HuggingFace model, create shorthand in `./src/config/config.py`
```
# Modify MODEL_INFO["model_mapping"] with shorthand for directory
```

1. Perform generations on ALL datasets (e.g., for a LLaMA 3.1 8B model on HuggingFace)
```
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
python run_gen.py --model_path ${MODEL_NAME};
```

2. Perform evaluation
```
RESULTS_DIR="./generation_results/llama3.1-8b"
OPENAI_MODEL='gpt-4o-2024-08-06'    # OpenAI model for assessing stereotype bias
python -m src.task.ceb_benchmark --results_dir ${RESULTS_DIR} --openai_model ${OPENAI_MODEL}
```

## Citation

Consider citing the original benchmark paper.
```
@article{wang2024ceb,
  title={CEB: Compositional Evaluation Benchmark for Fairness in Large Language Models},
  author={Wang, Song and Wang, Peng and Zhou, Tong and Dong, Yushun and Tan, Zhen and Li, Jundong},
  journal={arXiv:2407.02408},
  year={2024}
}
```
