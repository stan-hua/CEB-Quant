"""
text_evaluator.py

Description: Uses LLaMA Guard 3 8B to evaluate safety in text
"""

# Standard libraries
import logging
import os
import traceback

# Non-standard libraries
import torch
from tqdm import tqdm

# Custom libraries
import config
from src.utils import json_utils, text_eval_utils
from src.utils.llm_gen_wrapper import LLMGeneration


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Set vLLM multi-processing to spawn to fix the issue: https://github.com/vllm-project/vllm/issues/6152
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Default judge model path
LLAMA_MODEL = "meta-llama/Llama-Guard-3-8B"

# Default save name
DEFAULT_SAVE_FNAME = "{judge}_autoeval.json"

# Specify the number of GPUs to use
NUM_GPUS = 0
if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()

# Cache judge
JUDGE_CACHE = {}


################################################################################
#                                   Classes                                    #
################################################################################
class OpenTextEvaluator:
    """
    OpenTextEvaluator class.

    Note
    ----
    Wraps about LLaMAGuard and other metrics
    """

    def __init__(self, model_path=LLAMA_MODEL, save_dir=None, save_fname=None):
        """
        Initialize the OpenTextEvaluator class.

        Parameters
        ----------
        model_path : str, optional
            HuggingFace path to model to be used for evaluation
        save_dir : str, optional
            The directory to save evaluation results. Defaults to a directory
            within config.DIR_EVALUATIONS based on the model name.
        save_fname : str, optional
            Filename to save within save directory
        """
        assert save_dir, "Please pass a valid `save_dir` to save evaluation results!"
        self.save_dir = save_dir
        self.model_path = model_path
        # Lazy load LLM, on first call
        self.judge = None
        # Judge vLLM arguments
        self.vllm_kwargs = {
            "temperature": 0,           # NOTE: Force determinism
            "max_new_tokens": 100,      # NOTE: Not many output tokens needed
            "enforce_eager": False,
        }

        # Default keys
        self.prompt_col = "prompt"
        self.llm_input_col = "res"                      # initial LLM response
        self.llm_response_col = "eval_res_llama"        # judge LLM response
        self.save_fname = save_fname or DEFAULT_SAVE_FNAME.format(judge="llama_guard")


    def load_judge(self):
        """
        Load LLaMA Guard LLM
        """
        if self.judge is None:
            # Attempt to laod in cache
            if self.model_path in JUDGE_CACHE:
                self.judge = JUDGE_CACHE[self.model_path]
            # Otherwise, load directly
            else:
                judge = LLMGeneration(model_path_or_name=self.model_path, **self.vllm_kwargs)
                JUDGE_CACHE[self.model_path] = judge
                self.judge = judge


    def save_progress(self, data, filename=None, **save_kwargs):
        """
        Save evaluation progress to a JSON file.

        Args:
            data: Data to be saved.
            filename (str): Name of the file for saving the data.
        """
        filename = filename or self.save_fname
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, filename)
        json_utils.save_json(data, save_path, **save_kwargs)


    def evaluate(
        self, data,
        save_fname=None,
        prompt_col=None,
        llm_input_col=None,
        llm_response_col=None,
    ):
        """
        Evaluate a dataset using an open judge LLM.

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt to
            evaluate
        save_fname : str, optional
            Filename for saving or resuming progress.
        prompt_col : str, optional
            Key containing initial prompt that was used to generate response
        llm_input_col : str, optional
            Key to LLM response from initial prompt to evaluate. Overwrites "res"
            in config.config prompts
        llm_response_col : str, optional
            Key to store the judge LLM's response.

        Returns
        -------
        list
            The evaluated data.
        """
        save_fname = save_fname or self.save_fname

        # Modify keys
        self.prompt_col = prompt_col or self.prompt_col
        self.llm_input_col = llm_input_col or self.llm_input_col
        self.llm_response_col = llm_response_col or self.llm_response_col

        # Early return, if no data provided
        if not data:
            raise RuntimeError("No data provided for evaluation!")

        # Perform input sanitization
        assert isinstance(data, list), f"Data must be a list. data={data}"
        assert data, "Data provided is empty!"

        # 1. LLaMA evaluations
        # NOTE: Only if a GPU is available
        unevaluated_data = [row for row in data if not row.get(self.llm_response_col)]
        if unevaluated_data:
            if torch.cuda.is_available():
                self.perform_llama_eval(unevaluated_data)
                self.save_progress(data, filename=save_fname)
            else:
                LOGGER.warning("Skipping LLaMA-Guard safety evaluation, since no GPU is available!")

        # 2. Various text bias evaluations
        unevaluated_data = [row for row in data if "toxicity" not in row]
        # NOTE: Only computing other text scores with CPU for now
        if unevaluated_data and not torch.cuda.is_available():
            self.perform_text_bias_eval(unevaluated_data)
            self.save_progress(data, filename=save_fname)

        # Text quality evaluations
        unevaluated_data = [row for row in data if "lt-error_count" not in row]
        # NOTE: Only computing other text scores with CPU for now
        if unevaluated_data and not torch.cuda.is_available():
            self.perform_text_quality_eval(unevaluated_data)
            self.save_progress(data, filename=save_fname)

        return data


    def perform_llama_eval(self, data):
        """
        Parameters
        ----------
        data : list of dict, optional
            Each dict is a question to be evaluated
        """
        # Early return, if no data
        if not data:
            return False

        self.load_judge()

        # Convert each row into conversation format
        conversations = []
        for row in data:
            conversations.append([
                {"role": "user", "content": row[self.prompt_col]},
                {"role": "assistant", "content": row[self.llm_input_col]},
            ])

        # LLaMA Guard evaluate prompt & response
        eval_responses = self.judge.vllm_chat_multiple(conversations)

        # Store responses
        for idx, eval_res in enumerate(eval_responses):
            row = data[idx]
            row[self.llm_response_col] = eval_res["res"]

        return True


    def perform_text_bias_eval(self, data):
        """
        Evaluate text using a sweep of bias metrics.

        Parameters
        ----------
        data : list of dict, optional
            Each dict is a question to be evaluated
        """
        # Early return, if no data
        if not data:
            return False

        # If no GPU exists, do serially
        if not torch.cuda.is_available():
            for row in tqdm(data):
                row.update(text_eval_utils.compute_bias_metrics(
                    prompt=row[self.prompt_col],
                    response=row[self.llm_input_col],
                ))
            return True

        # If GPU exists, try batched evaluation
        try:
            prompts = [row[self.prompt_col] for row in data]
            responses = [row[self.llm_input_col] for row in data]
            eval_responses = text_eval_utils.compute_bias_metrics_batch(prompts, responses)
            for idx, eval_res in enumerate(eval_responses):
                if not eval_res:
                    continue
                row = data[idx]
                row.update(eval_res)
        # If fails, fallback to single row processing
        except Exception as error_msg:
            LOGGER.error(f"Batched text evaluation failed! Error: {error_msg}")
            for row in data:
                row.update(text_eval_utils.compute_bias_metrics(
                    prompt=row[self.prompt_col],
                    response=row[self.llm_input_col],
                ))
        return True


    def perform_text_quality_eval(self, data):
        """
        Evaluate quality of text.

        Parameters
        ----------
        data : list of dict, optional
            Each dict is a question to be evaluated
        """
        # Early return, if no data
        if not data:
            return False

        try:
            for row in tqdm(data):
                row.update(text_eval_utils.compute_quality_metrics(
                    row[self.llm_input_col],
                ))
        except:
            LOGGER.error(f"Text quality evaluation failed! Error trace:")
            traceback.print_exc()
            return False

        return True
