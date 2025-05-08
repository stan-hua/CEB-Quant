"""
judge_evaluator.py

Description: Uses Prometheus-Eval 2.0 / Atla as a judge to evaluate
             stereotype/toxicity in LLM-generated text.
"""

# Standard libraries
import logging
import os
import traceback

# Non-standard libraries
import torch
from atla_eval import AtlaEval
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE
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

# Default judge model paths
PROMETHEUS_MODEL = "prometheus-eval/prometheus-7b-v2.0"
# PROMETHEUS_MODEL = "prometheus-eval/prometheus-bgb-8x7b-v2.0"
ATLA_MODEL = "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
LLAMA_MODEL = "meta-llama/Llama-Guard-3-8B"

# Default save name
DEFAULT_SAVE_FNAME = "{judge}_autoeval.json"

# Mapping of social group to name in template `{social_group} group`
SOCIAL_GROUP_FORMAT = {
    "age": "age",
    "gender": "gender/sex",
    "gender_and_sex": "gender/sex",
    "race": "racial/ethnic",
    "race_ethnicity": "racial/ethnic",
    "religion": "religious",
}

# Specify the number of GPUs to use
NUM_GPUS = 0
if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()

# Get default prompt version
JUDGE_PROMPT_VER = os.environ.get("JUDGE_PROMPT_VER", 2)

# Cache judge
JUDGE_CACHE = {}


################################################################################
#                                   Classes                                    #
################################################################################
class OpenJudgeEvaluator:
    """
    OpenJudgeEvaluator class.

    Notes
    -----
    Used to evaluate LLM responses via Prometheus 2.0 or Atla
    """

    def __init__(
            self,
            judge_choice="prometheus",
            model_path=None,
            prompt=None,
            prompt_version=JUDGE_PROMPT_VER,
            save_dir=None,
        ):
        """
        Initialize the OpenJudgeEvaluator class.

        Parameters
        ----------
        judge_choice : str
            Choice of open LLM judge (prometheus/atla)
        model_path : str, optional
            HuggingFace path to model to be used for evaluation
        prompt : str, optional
            Prompt template to use. Defaults to absolute grading without
            reference.
        prompt_version : int, optional
            Prompt version (see `config.py` for more details)
        save_dir : str, optional
            The directory to save evaluation results. Defaults to a directory
            within config.DIR_EVALUATIONS based on the model name.
        """
        assert save_dir, "Please pass a valid `save_dir` to save evaluation results!"
        assert judge_choice in ["prometheus", "atla"], "Only supports Prometheus/Atla evaluators currently!"
        self.save_dir = save_dir
        self.model_path = model_path
        self.prompt = prompt
        self.prompt_version = int(prompt_version)
        self.judge_choice = judge_choice
        # Lazy load LLM, on first call
        self.judge = None

        # Default keys
        self.prompt_col = "prompt"
        self.llm_input_col = "res"
        self.llm_response_col = "eval_res"
        self.save_fname = DEFAULT_SAVE_FNAME.format(judge=judge_choice)

        # Default model based on judge choice
        if self.model_path is None:
            judge_to_path = {"prometheus": PROMETHEUS_MODEL, "atla": ATLA_MODEL}
            self.model_path = judge_to_path[self.judge_choice]


    def load_judge(self):
        """
        Load judge LLM
        """
        if self.judge is None:
            # Load vLLM model
            model = VLLM(
                model=self.model_path,
                tensor_parallel_size=NUM_GPUS,
                gpu_memory_utilization=0.8,
            )

            # Create Judge class
            judge_to_cls = {"prometheus": PrometheusEval, "atla": AtlaEval}
            kwargs = {}
            if self.prompt:
                kwargs["absolute_grade_template"] = self.prompt
            self.judge = judge_to_cls[self.judge_choice](model=model, **kwargs)


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
        self, data, task,
        resume=True,
        save_fname=None,
        prompt_col="prompt",
        llm_input_col="res",
        llm_response_col="eval_res",
    ):
        """
        Evaluate a dataset using an open judge LLM.

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt to
            evaluate
        task : str
            Name of the task to evaluate.
        resume : bool, optional
            If True, then try to resume evaluation from a saved progress file
            with the same filename as `save_fname`. Default is True.
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
            LOGGER.warning("No data provided for evaluation!")
            # Save empty JSON list in the folder
            self.save_progress([], filename=save_fname)
            return []

        # If specified, resume from previous evaluation
        if resume:
            load_path = os.path.join(self.save_dir, save_fname)
            data = json_utils.update_with_existing_data(data, prev_path=load_path, prompt_col=prompt_col)

        # Perform input sanitization
        assert isinstance(data, list), f"Data must be a list. data={data}"
        assert data, "Data provided is empty!"
        assert task is not None, "Task must be specified for evaluation."

        # Early return, if all data has been evaluated
        unevaluated_data = [row for row in data if not row.get(self.llm_response_col)]
        if not unevaluated_data:
            LOGGER.info("All data has already been evaluated!")
            return data

        # Save progress
        self.perform_eval(unevaluated_data, task)
        self.save_progress(data, filename=save_fname)

        return data


    def perform_eval(self, data, task):
        """
        Parameters
        ----------
        data : list of dict, optional
            Each dict is a question to be evaluated
        task : str
            Name of the task to evaluate.
        """
        # Early return, if no data
        if not data:
            return

        self.load_judge()

        # Get social axis and groups present
        social_axis_to_groups = {}
        for row in data:
            social_axis = row["axis"]
            social_group = row["descriptor"]
            if social_axis not in social_axis_to_groups:
                social_axis_to_groups[social_axis] = set([])
            social_axis_to_groups[social_axis].add(social_group)

        # Evaluate data for each social axis and group, separately
        for axis, groups in social_axis_to_groups.items():
            social_axis_data = [row for row in data if row["axis"] == axis]
            for group in groups:
                social_group_data = [row for row in social_axis_data if row["descriptor"] == group]
                self.perform_eval_single_group(social_group_data, task)


    def perform_eval_single_group(self, data, task):
        # Ensure that social axis and social group are present
        first_row = data[0]
        assert "axis" in first_row, "Social axis column `axis` not found in data!"
        assert "descriptor" in first_row, "Social group column `descriptor` not found in data!"
        social_axis = SOCIAL_GROUP_FORMAT[first_row["axis"]]
        social_group = first_row["descriptor"]

        self.load_judge()

        # Get rubric for stereotype/toxicity
        task_to_rubric = config.PROMPT_VER_TO_RUBRICS[self.prompt_version]
        rubric_data = task_to_rubric[task]

        # Get initial instructions and their LLM responses
        instructions = [row[self.prompt_col] for row in data]
        responses = [row[self.llm_input_col] for row in data]

        # Modify rubric string to be targeted towards the social group
        fairness_score_rubric = prepare_rubric_string(
            rubric_data,
            social_axis=social_axis,
            social_group=social_group,
        )

        # CASE: If 2nd prompt version, then perform instruction-following evaluation too
        if self.prompt_version != 1:
            # Evaluate instruction following
            rta_rubric_data = task_to_rubric["instruction_following"]
            rta_score_rubric = prepare_rubric_string(rta_rubric_data)
            rta_feedbacks, rta_scores = self.judge.absolute_grade(
                instructions=instructions,
                responses=responses,
                rubric=rta_score_rubric,
            )
            # Store judge responses
            for idx, row in enumerate(data):
                rta_col = f"{self.llm_response_col}_rta"
                row[rta_col] = f"Score: {rta_scores[idx]}\n\nFeedback: ```{rta_feedbacks[idx]}```"

        # Evaluate fairness
        feedbacks, fairness_scores = self.judge.absolute_grade(
            instructions=instructions,
            responses=responses,
            rubric=fairness_score_rubric,
        )
        # Store judge responses
        for idx, row in enumerate(data):
            row[self.llm_response_col] = f"Score: {fairness_scores[idx]}\n\nFeedback: ```{feedbacks[idx]}```"


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


################################################################################
#                               Helper Functions                               #
################################################################################
def prepare_rubric_string(rubric_data, **str_formatters):
    """
    Prepare rubric string

    Parameters
    ----------
    rubric_data : dict
        Dictionary containing rubric items
    **str_formatters : Any
        Keyword arguments that contain string formatters to substitute into
        the rubric dictionary's items
    """
    # Update rubric dictionary with substituted items
    rubric_data = rubric_data.copy()
    for key, description in rubric_data.items():
        rubric_data[key] = description.format(**str_formatters)

    # Format score rubric
    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    return score_rubric
