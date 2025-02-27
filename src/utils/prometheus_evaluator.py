"""
prometheus_evaluator.py

Description: Uses Prometheus-Eval 2.0 as a judge to evaluate stereotype/toxicity
             in LLM-generated text.
"""

# Standard libraries
import logging
import os

# Non-standard libraries
import torch
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT_WO_REF, SCORE_RUBRIC_TEMPLATE

# Custom libraries
import config
from src.utils import json_utils


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Set vLLM multi-processing to spawn to fix the issue: https://github.com/vllm-project/vllm/issues/6152
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Default Prometheus model for evaluation
PROMETHEUS_MODEL = "prometheus-eval/prometheus-7b-v2.0"
# PROMETHEUS_MODEL = "prometheus-eval/prometheus-bgb-8x7b-v2.0"

# Default save name
DEFAULT_SAVE_FNAME = "prometheus_autoeval.json"

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

# Get default prometheus prompt version
PROMETHEUS_PROMPT_VER = os.environ.get("PROMETHEUS_PROMPT_VER", 2)


################################################################################
#                                   Classes                                    #
################################################################################
# TODO: Consider a relative grading between full-precision and quantized model
class PrometheusEvaluator:
    """
    PrometheusEvaluator class.

    Notes
    -----
    Used to evaluate LLM responses via Prometheus 2.0
    """

    def __init__(
            self,
            model_path=PROMETHEUS_MODEL,
            prompt=ABSOLUTE_PROMPT_WO_REF,
            prompt_version=PROMETHEUS_PROMPT_VER,
            save_dir=None,
        ):
        """
        Initialize the PrometheusEvaluator class.

        Parameters
        ----------
        model_path : str, optional
            HuggingFace path to Prometheus model to be used for evaluation, by
            default PROMETHEUS_MODEL.
        prompt : str, optional
            Prompt template to use. Defaults to absolute grading without
            reference.
        prompt_version : int, optional
            If 1, perform both refusal to answer and stereotype/toxicity evaluation
            in one step.
            If 2, perform refusal to answer and stereotype/toxicity evaluation
            separately.
        save_dir : str, optional
            The directory to save evaluation results. Defaults to a directory
            within config.DIR_EVALUATIONS based on the model name.
        """
        assert save_dir, "Please pass a valid `save_dir` to save evaluation results!"
        self.save_dir = save_dir
        self.model_path = model_path
        self.prompt = prompt
        self.prompt_version = int(prompt_version)
        # Lazy load LLM, on first call
        self.judge = None

        # Default keys
        self.prompt_key = "prompt"
        self.llm_input_col = "res"
        self.llm_response_col = "eval_res"


    def load_prometheus(self):
        """
        Load Prometheus
        """
        if self.judge is None:
            model = VLLM(
                model=self.model_path,
                tensor_parallel_size=NUM_GPUS,
                gpu_memory_utilization=0.8,
            )
            self.judge = PrometheusEval(model=model, absolute_grade_template=self.prompt)


    def save_progress(self, data, filename=DEFAULT_SAVE_FNAME, **save_kwargs):
        """
        Save evaluation progress to a JSON file.

        Args:
            data: Data to be saved.
            filename (str): Name of the file for saving the data.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, filename)
        json_utils.save_json(data, save_path, **save_kwargs)


    def evaluate(
        self, data, task,
        resume=True,
        progress_filename=DEFAULT_SAVE_FNAME,
        prompt_key="prompt",
        llm_input_col="res",
        llm_response_col="eval_res",
    ):
        """
        Evaluate a dataset using the OpenAI API.

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt to
            evaluate
        task : str
            Name of the task to evaluate.
        resume : bool, optional
            If True, then try to resume evaluation from a saved progress file
            with the same filename as `progress_filename`. Default is True.
        progress_filename : str, optional
            Filename for saving or resuming progress.
        prompt_key : str, optional
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
        # Modify keys
        self.prompt_key = prompt_key or self.prompt_key
        self.llm_input_col = llm_input_col or self.llm_input_col
        self.llm_response_col = llm_response_col or self.llm_response_col

        # Get rubric
        assert task in config.PROMETHEUS_EVAL_RUBRIC_DATA, f"Invalid task! Please ensure that task `{task}` is in `config.py` / PROMETHEUS_EVAL_RUBRIC_DATA..."

        # Early return, if no data provided
        if not data:
            LOGGER.warning("No data provided for evaluation!")
            # Save empty JSON list in the folder
            self.save_progress([], filename=progress_filename)
            return []

        # If specified, resume from previous evaluation
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            data = json_utils.update_with_existing_data(data, prev_path=load_path)

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
        self.save_progress(data, filename=progress_filename)

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

        self.load_prometheus()

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

        self.load_prometheus()

        # Get rubric for stereotype/toxicity
        rubric_data = config.PROMETHEUS_EVAL_RUBRIC_DATA[task]
        task_to_rubric = config.PROMETHEUS_VER_TO_RUBRICS[self.prompt_version]
        rubric_data = task_to_rubric[task]

        # Get initial instructions and their LLM responses
        instructions = [row[self.prompt_key] for row in data]
        responses = [row[self.llm_input_col] for row in data]

        # Modify rubric string to be targeted towards the social group
        fairness_score_rubric = prepare_rubric_string(
            rubric_data,
            social_axis=social_axis,
            social_group=social_group,
        )

        # CASE: If 2nd prompt version, then perform refusal to answer evaluation too
        if self.prompt_version != 1:
            # Evaluate instruction following
            rta_rubric_data = task_to_rubric["refusal_to_answer"]
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
