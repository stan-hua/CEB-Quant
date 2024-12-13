"""
prometheus_evaluator.py

Description: Uses Prometheus-Eval 2.0 as a judge to evaluate stereotype/toxicity
             in LLM-generated text.
"""

# Standard libraries
import concurrent.futures
import logging
import os
import threading

# Non-standard libraries
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

# Default Prometheus model for evaluation
PROMETHEUS_MODEL = "prometheus-eval/prometheus-7b-v2.0"

# Default save name
DEFAULT_SAVE_FNAME = "prometheus_autoeval.json"

# Mapping of social group to name in template `{social_group} group`
SOCIAL_GROUP_FORMAT = {
    "age": "age",
    "gender_and_sex": "gender/sex",
    "race_ethnicity": "racial/ethnic",
    "religion": "religious",
}


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

    def __init__(self, model_path=PROMETHEUS_MODEL, prompt=ABSOLUTE_PROMPT_WO_REF,
                 save_dir=None):
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
        save_dir : str, optional
            The directory to save evaluation results. Defaults to a directory
            within config.DIR_EVALUATIONS based on the model name.
        """
        model_name = os.path.basename(model_path)
        self.save_dir = save_dir or os.path.join(config.DIR_EVALUATIONS, "prometheus")

        # Lazy load LLM, on first call
        self.model_path = model_path
        self.prompt = prompt
        self.judge = None


    def load_prometheus(self):
        """
        Load Prometheus
        """
        if self.judge is None:
            model = VLLM(model=self.model_path)
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
        self.load_prometheus()

        # Get rubric
        assert task in config.PROMETHEUS_EVAL_RUBRIC_DATA, f"Invalid task! Please ensure that task `{task}` is in `config.py` / PROMETHEUS_EVAL_RUBRIC_DATA..."
        rubric_data = config.PROMETHEUS_EVAL_RUBRIC_DATA[task]

        # Early return, if no data provided
        if not data:
            LOGGER.warning("No data provided for evaluation!")
            return []

        # Ensure that social axis and social group are present
        first_row = data[0]
        assert "axis" in first_row, "Social axis column `axis` not found in data!"
        assert "descriptor" in first_row, "Social group column `descriptor` not found in data!"
        social_axis = SOCIAL_GROUP_FORMAT[first_row["axis"]]
        social_group = first_row["descriptor"]

        # Modify rubric string to be targeted towards the social group
        score_rubric = prepare_rubric_string(
            rubric_data,
            social_axis=social_axis,
            social_group=social_group,
        )

        # If specified, resume from previous evaluation
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            data = json_utils.update_with_existing_data(data, prev_path=load_path)

        # Perform input sanitization
        assert isinstance(data, list), f"Data must be a list. data={data}"
        assert data, "Data provided is empty!"
        assert task is not None, "Task must be specified for evaluation."

        # Get initial instructions and their LLM responses
        instructions = [row[prompt_key] for row in data]
        responses = [row[llm_input_col] for row in data]

        # Perform batched LLM evaluation
        feedbacks, scores = self.judge.absolute_grade(
            instructions=instructions,
            responses=responses,
            rubric=score_rubric,
        )

        # Store judge responses
        for idx, row in enumerate(data):
            feedback = feedbacks[idx]
            score = scores[idx]
            row["eval_res"] = f"Score: {score}\n\nFeedback: ```{feedback}```"

        # Save progress
        self.save_progress(data, filename=progress_filename)

        return data

        """
        Perform inference on a dataset using the OpenAI API.

        Parameters
        ----------
        data : list of dict
            Each dict contains a prompt in the `llm_input_col` to perform
            inference on.
        resume : bool, optional
            If True, then try to resume inference from a saved progress file
            with the same filename as `progress_filename`. Default is True.
        progress_filename : str, optional
            Filename for saving or resuming progress.
        llm_input_col : str, optional
            Key to prompt to perform inference on
        llm_response_col : str, optional
            Key to store LLM's response.

        Returns
        -------
        list
            The evaluated data.
        """
        def save_progress_callback(future):
            if future.exception() is not None:
                LOGGER.error("An error occurred: %s", str(future.exception()))
                self.save_progress(data, filename=progress_filename)

        def process_row(prompt, row):
            try:
                if not row.get(llm_response_col):
                    llm_response = openai_chat_completion(prompt, model=self.model)
                    row[llm_response_col] = llm_response
            except Exception as error_msg:
                raise error_msg

        # Early return, if no data provided
        if not data:
            return []

        # Ensure all rows have a prompt
        assert all(llm_input_col in row for row in data), "All rows must have a prompt specified!"

        # Assume full prompt is specified in the llm input column
        prompts = [data.get(llm_input_col) for data in data]

        # If specified, resume from previous inference
        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            data = json_utils.update_with_existing_data(data, prev_path=load_path)

        # Perform input sanitization
        assert isinstance(data, list), f"Data must be a list. data={data}"
        assert data, "Data provided is empty!"

        # Create thread lock
        lock = threading.Lock()

        # Perform LLM generation requests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(process_row, prompt, row) for prompt, row in zip(prompts, data)]

            # Add a callback to handle completion and errors
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                future.add_done_callback(save_progress_callback)
                if idx % 10 == 0:
                    self.save_progress(data, filename=progress_filename, lock=lock)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        # Save progress
        self.save_progress(data, filename=progress_filename)

        return data


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
