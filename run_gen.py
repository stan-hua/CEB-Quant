# Standard libraries
import logging
import sys

# Non-standard libraries
from fire import Fire

# Custom libraries
from src.generation.generation import LLMGeneration


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


################################################################################
#                                  Functions                                   #
################################################################################
def generate(model_path: str, dataset_name: str, online_model: bool = False):
    """
    Run the generation task for the specified dataset(s).

    Parameters
    ----------
    model_path : str
        Path to the model.
    dataset_name : str
        Name of the dataset.
    online_model : bool
        Whether to use the online model or not (vLLM).
    """
    # CASE 1: Online APIs (e.g., ChatGPT)
    if online_model:
        print("Using online model")
        llm_gen = LLMGeneration(
            model_path=model_path,
            data_path="./data/",
            dataset_name=dataset_name,          # run on all datasets in the folder
            online_model=True,
            use_deepinfra=False,
            use_replicate=False,
            use_vllm=False,
            repetition_penalty=1.0,
            num_gpus=1,
            max_new_tokens=512,
            debug=False
        )
    # CASE 2: Offline models (vLLM)
    else:
        print("Using vLLM model")
        llm_gen = LLMGeneration(
            model_path=model_path,
            data_path="./data/",
            dataset_name=dataset_name,          # run on all datasets in the folder
            online_model=False, 
            use_deepinfra=False,
            use_replicate=False,
            use_vllm=True,
            repetition_penalty=1.0,
            num_gpus=1, 
            max_new_tokens=512, 
            debug=False
        )

    llm_gen.infer_dataset()


if __name__ == "__main__":
    Fire(generate)