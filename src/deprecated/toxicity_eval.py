# Standard libraries
import logging
import os
import time

# Non-standard libraries
from filelock import FileLock
from googleapiclient import discovery
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm

# Custom libraries
from config import (
    DIR_EVALUATIONS, PERSPECTIVE_EVAL_FNAME, PERSPECTIVE_API_URL,
    PERSPECTIVE_KEY, PERSPECTIVE_LOCK_FNAME,
)
from src.utils import chatgpt_eval, eval_utils, judge_evaluator, json_utils, metric_utils


################################################################################
#                                    Setup                                     #
################################################################################
# Configure logger
LOGGER = logging.getLogger(__name__)


################################################################################
#                                   Classes                                    #
################################################################################
class ToxicityEval:
    """
    A class for evaluating text for toxicity using the Perspective API.
    """

    def __init__(self, save_dir=DIR_EVALUATIONS, alpha=0.05,
                 filter_kwargs=None,
                 evaluator_choice="chatgpt",
                 **kwargs):
        """
        Initialize the ToxicityEval class.

        Parameters
        ----------
        save_dir : str
            Directory to save the evaluation results.
        alpha : float
            Alpha level for confidence interval
        filter_kwargs : bool
            If True, filter for harmful prompts
        evaluator_choice : str
            Choice of evaluator: ("chatgpt", "prometheus", "atla")
        **kwargs : Any
            Keyword arguments to be passed to the ChatGPT/Prometheus evaluator
        """
        self.metric_kwargs = {"alpha": alpha}
        self.filter_kwargs = filter_kwargs
        self.evaluator_choice = evaluator_choice

        # Create save directory if it doesn't exist
        self.save_dir = save_dir

        # Create Perplexity API client
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=PERSPECTIVE_KEY,
            discoveryServiceUrl=PERSPECTIVE_API_URL,
            static_discovery=False,
        )

        # Create evaluator
        # CASE 1: ChatGPT evaluator
        if evaluator_choice == "chatgpt":
            eval_utils.pop_invalid_kwargs(kwargs, ["model"])
            self.evaluator = chatgpt_eval.ChatGPTEvaluator(save_dir=save_dir, **kwargs)
        # CASE 2: Prometheus/Atla evaluator
        elif evaluator_choice in ["prometheus", "atla"]:
            eval_utils.pop_invalid_kwargs(kwargs, ["model_path", "prompt", "prompt_version", "judge_choice"])
            kwargs["judge_choice"] = evaluator_choice
            self.evaluator = judge_evaluator.OpenJudgeEvaluator(save_dir=save_dir, **kwargs)
        else:
            raise NotImplementedError(f"Invalid evaluator choice! `{evaluator_choice}`")


    def eval_toxicity(self, dataset_name, data):
        """
        Evaluate toxicity task based on dataset name.

        Parameters
        ----------
        dataset_name : str
            Name of the toxicity dataset
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            recognizing toxicity

        Returns
        -------
        dict
            Contains evaluation metrics specific to the dataset
        """
        if "Recognition" in dataset_name:
            return self.eval_toxicity_recognition(data)
        elif "Selection" in dataset_name:
            return self.eval_toxicity_selection(data)
        elif "Continuation" in dataset_name:
            return self.eval_toxicity_continuation(data)
        elif "Conversation" in dataset_name:
            return self.eval_toxicity_conversation(data)
        elif dataset_name == "CEB-Jigsaw":
            return self.eval_toxicity_classification_jigsaw(data)

        raise RuntimeError(f"Unsupported dataset name: {dataset_name}!")


    ############################################################################
    #                          Direct Evaluation                               #
    ############################################################################
    def eval_toxicity_recognition(self, data):
        """
        Compute micro F1-score for toxicity recognition

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            recognizing toxicity

        Returns
        -------
        dict
            Contains micro F1-score and proportion of invalid responses
        """
        # Split data into those needing an LLM to parse
        direct_eval_data, to_parse_data = [], []
        for row in data:
            answer = metric_utils.extract_toxicity_recognition_answer(row["res"])
            if answer is None and metric_utils.is_text_truthy(row["res"]):
                to_parse_data.append(row)
            else:
                direct_eval_data.append(row)

        # Use LLM to parse response
        if to_parse_data:
            parsed_eval_data = self.evaluator.evaluate(
                to_parse_data,
                task="toxicity_recognition",
            )
        else:
            parsed_eval_data = []

        # Filter for harmful data
        # NOTE: This is done after, so that we don't overwrite existing evaluations
        if self.filter_kwargs:
            direct_eval_data = eval_utils.filter_data_by_kwargs(direct_eval_data, self.filter_kwargs)
            parsed_eval_data = eval_utils.filter_data_by_kwargs(parsed_eval_data, self.filter_kwargs)

        # Combine and compute F1-score
        all_rows = list(direct_eval_data) + list(parsed_eval_data)
        metrics = metric_utils.compute_score_for_recognition(
            all_rows, metric_utils.extract_toxicity_recognition_answer,
            **self.metric_kwargs,
        )
        return metrics


    def eval_toxicity_selection(self, data):
        """
        Compute micro F1-score for identifying toxicity

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            selecting toxicity

        Returns
        -------
        dict
            Contains micro F1-score of correctly identified toxicity, and
            proportion of invalid responses
        """
        # Filter for harmful data
        # NOTE: This is done after, so that we don't overwrite existing evaluations
        if self.filter_kwargs:
            data = eval_utils.filter_data_by_kwargs(data, self.filter_kwargs)
        return metric_utils.compute_score_for_selection(data, **self.metric_kwargs)


    ############################################################################
    #                         Indirect Evaluation                              #
    ############################################################################
    def eval_toxicity_continuation(
        self, data,
        resume=True,
        save_fname=PERSPECTIVE_EVAL_FNAME,
    ):
        """
        Compute toxicity score for Toxicity / Continuation dataset using the
        Perspective API.

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            generating a toxic or non-toxic continuation. The dict must contain
            a 'res' key with the response text.
        resume : bool, optional
            If True, then try to resume evaluation from a saved progress file
            with the same filename as `save_fname`. Default is True.
        save_fname : str, optional
            Filename for saving or resuming progress. Default is
            `PERSPECTIVE_EVAL_FNAME`.

        Returns
        -------
        dict
            Contains average toxicity score, proportion of invalid responses,
            and number of samples.
        """
        # Add resume path
        resume_from = None
        if resume:
            resume_from = os.path.join(self.save_dir, save_fname)

        # Keyword arguments for evaluate functions
        eval_func_kwargs = {
            "input_col": "res_wo_prefix",
            "save_fname": save_fname
        }

        # Compute indirect bias score
        metrics = eval_utils.score_open_ended_responses(
            data, self.evaluator, self.evaluator_choice,
            bias_type="toxicity", instruction_type="continuation",
            resume_from=resume_from,
            filter_kwargs=self.filter_kwargs,
            evaluate_func=self.assign_toxicity_values,
            eval_func_kwargs=eval_func_kwargs,
            **self.metric_kwargs,
        )
        return metrics


    def eval_toxicity_conversation(
        self, data,
        resume=True,
        save_fname=PERSPECTIVE_EVAL_FNAME,
    ):
        """
        Compute toxicity score for Toxicity / Conversation dataset using the
        Perspective API.

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            generating a toxic or non-toxic continuation. The dict must contain
            a 'res' key with the response text.
        resume : bool, optional
            If True, then try to resume evaluation from a saved progress file
            with the same filename as `save_fname`. Default is True.
        save_fname : str, optional
            Filename for saving or resuming progress. Default is
            `PERSPECTIVE_EVAL_FNAME`.

        Returns
        -------
        dict
            Contains average toxicity score, proportion of invalid responses,
            and number of samples.
        """
        # Add resume path
        resume_from = None
        if resume:
            resume_from = os.path.join(self.save_dir, save_fname)

        # Keyword arguments for evaluate functions
        eval_func_kwargs = {
            "input_col": "res_wo_prefix",
            "save_fname": save_fname
        }

        # Compute indirect bias score
        metrics = eval_utils.score_open_ended_responses(
            data, self.evaluator, self.evaluator_choice,
            bias_type="toxicity", instruction_type="conversation",
            resume_from=resume_from,
            filter_kwargs=self.filter_kwargs,
            evaluate_func=self.assign_toxicity_values,
            eval_func_kwargs=eval_func_kwargs,
            **self.metric_kwargs,
        )
        return metrics


    def eval_toxicity_classification_jigsaw(self, data):
        """
        Compute fairness metrics on indirect toxicity evaluation, specifically
        CEB Classification / Jigsaw dataset 

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            predicting if a comment is toxic or not

        Returns
        -------
        dict
            Contains demographic parity difference, equalized odds difference,
            proportion of invalid responses, and number of samples
        """
        # Split data into those needing an LLM to parse
        direct_eval_data, to_parse_data = [], []
        for row in data:
            # 1. First attempt with capital YES and NO
            answer = metric_utils.extract_valid_choice(row["res"], ["YES", "NO"])
            # 2. Second attempt with lowercase YES and NO, and lowercase text
            if answer is None:
                answer = metric_utils.extract_valid_choice(row["res"].lower(), ["yes", "no"])

            # Based on above result, check if needs LLM parsing
            if answer is None and metric_utils.is_text_truthy(row["res"]):
                to_parse_data.append(row)
            else:
                direct_eval_data.append(row)

        # Use LLM to parse response
        if to_parse_data:
            parsed_eval_data = self.evaluator.evaluate(
                to_parse_data,
                task="toxicity-classification_jigsaw"
            )
        else:
            parsed_eval_data = []

        # Filter for harmful data
        # NOTE: This is done after, so that we don't overwrite existing evaluations
        if self.filter_kwargs:
            direct_eval_data = eval_utils.filter_data_by_kwargs(direct_eval_data, self.filter_kwargs)
            parsed_eval_data = eval_utils.filter_data_by_kwargs(parsed_eval_data, self.filter_kwargs)

        # Combine and compute fairness metrics
        all_rows = list(direct_eval_data) + list(parsed_eval_data)
        return metric_utils.compute_score_for_classification_datasets(all_rows, **self.metric_kwargs)


    ############################################################################
    #                           Helper Functions                               #
    ############################################################################
    @retry(wait=wait_exponential(min=2, max=8), stop=stop_after_attempt(3))
    def get_toxicity_value(self, sentence, language="en"):
        """
        Get the toxicity score of a sentence using the Perspective API.

        Parameters
        ----------
        sentence : str
            Sentence to identify toxicity of
        language : str
            Language of sentence

        Returns
        -------
            float: The toxicity score of the sentence.
        """
        analyze_request = {
            "comment": {"text": sentence},
            "requestedAttributes": {"TOXICITY": {}},
            "languages": [language]
        }
        response = self.client.comments().analyze(body=analyze_request).execute()
        return response["attributeScores"]["TOXICITY"]["spanScores"][0]["score"]["value"]


    def assign_toxicity_values(
            self,
            data,
            save_fname=PERSPECTIVE_EVAL_FNAME,
            input_col="res",
            output_col="toxicity",
    ):
        """
        For each row in the data, use the Perspective API to get the toxicity
        value of the response and save it back to the row.

        Parameters
        ----------
        data : list of dict
            The data to evaluate
        save_fname : str, optional
            The filename to save the progress to, by default PERSPECTIVE_EVAL_FNAME
        input_col : str, optional
            The column name of the input text, by default "res"
        output_col : str, optional
            The column name to save the toxicity value, by default "toxicity"

        Returns
        -------
        list of dict
            The data with toxicity values added
        """
        # Acquire file lock on Perspective API, so no one else can use it
        # NOTE: Can only send 1 request at a time
        with FileLock(PERSPECTIVE_LOCK_FNAME, timeout=-1):
            # For each row, use Perspective API to get toxicity value
            for idx, row in tqdm(enumerate(data)):
                # Save on every 10 responses
                if idx and idx % 10 == 0:
                    self.save_progress(data, filename=save_fname)

                # Skip, if already has toxicity
                if output_col in row:
                    continue

                # Skip, if refused to answer, or invalid answer
                if row["rta"] or not row.get(input_col):
                    continue

                # Use Perspective API to get toxicity value serially
                toxicity_score = None
                num_attempts = 0
                while toxicity_score is None:
                    # Skip after more than 3 attempts
                    if num_attempts >= 3:
                        break

                    try:
                        num_attempts += 1
                        text = row[input_col]
                        toxicity_score = self.get_toxicity_value(text)
                        time.sleep(0.2)  # Delay to prevent API rate limit issues
                        row[output_col] = toxicity_score
                    except Exception as error_msg:
                        # Attempt to get HTTPS request error
                        if hasattr(error_msg, "response") and hasattr(getattr(error_msg, "response"), "text"):
                            error_msg = getattr(getattr(error_msg, "response"), "text")
                        else:
                            error_msg = str(error_msg)
                        LOGGER.error("Error occurred calling the Perspective API! (%s) \n\tText: %s", error_msg, text)

        # Final save
        self.save_progress(data, filename=save_fname)
        return data


    def save_progress(self, data, filename=PERSPECTIVE_EVAL_FNAME, **save_kwargs):
        """
        Save the evaluation progress to a JSON file.

        Parameters
        ----------
        data : list of dict
            The evaluation progress data
        filename : str, optional
            The filename of the JSON file to save to. Default is
            `PERSPECTIVE_EVAL_FNAME`.
        **save_kwargs : Any
            Additional keyword arguments to pass to `json_utils.save_json`.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, filename)
        json_utils.save_json(data, save_path, **save_kwargs)
