# Standard libraries
import logging

# Custom libraries
from src.task import eval_utils
from src.utils import chatgpt_eval, prometheus_evaluator, metric_utils


################################################################################
#                                    Setup                                     #
################################################################################
# Configure logger
LOGGER = logging.getLogger(__name__)


################################################################################
#                                   Classes                                    #
################################################################################
class StereotypeEval:
    """
    StereotypeEval class.

    Note
    ----
    Used to compute metrics on one stereotype dataset at a time
    """

    def __init__(self, alpha=0.05, filter_kwargs=None,
                 evaluator_choice="chatgpt",
                 **kwargs):
        """
        Initializes the FairnessEval class.

        Parameters
        ----------
        alpha : float
            Alpha level for confidence interval
        filter_kwargs : dict
            Contains keyword arguments to filter prompts for
        evaluator_choice : str
            Choice of evaluator: ("chatgpt", "prometheus")
        **kwargs : Any
            Keyword arguments for the evaluator
        """
        self.metric_kwargs = {"alpha": alpha}
        self.filter_kwargs = filter_kwargs
        self.evaluator_choice = evaluator_choice

        # CASE 1: ChatGPT evaluator
        if evaluator_choice == "chatgpt":
            eval_utils.pop_invalid_kwargs(kwargs, ["model", "save_dir"])
            self.evaluator = chatgpt_eval.ChatGPTEvaluator(**kwargs)
        # CASE 2: Prometheus evaluator
        elif evaluator_choice == "prometheus":
            eval_utils.pop_invalid_kwargs(kwargs, ["model_path", "prompt", "prompt_version", "save_dir"])
            self.evaluator = prometheus_evaluator.PrometheusEvaluator(**kwargs)


    def eval_stereotype(self, dataset_name, data):
        """
        Evaluate stereotype task based on dataset name.

        Parameters
        ----------
        dataset_name : str
            Name of the stereotype dataset
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            recognizing stereotype

        Returns
        -------
        dict
            Contains evaluation metrics specific to the dataset
        """
        if "Recognition" in dataset_name:
            return self.eval_stereotype_recognition(data)
        elif "Selection" in dataset_name:
            return self.eval_stereotype_selection(data)
        elif "Continuation" in dataset_name:
            return self.eval_stereotype_continuation(data)
        elif "Conversation" in dataset_name:
            return self.eval_stereotype_conversation(data)
        elif dataset_name == "CEB-Adult":
            return self.eval_stereotype_classification_adult(data)
        elif dataset_name == "CEB-Credit":
            return self.eval_stereotype_classification_credit(data)
        raise RuntimeError(f"Unsupported dataset name: {dataset_name}!")


    ############################################################################
    #                          Direct Evaluation                               #
    ############################################################################
    def eval_stereotype_recognition(self, data):
        """
        Compute micro F1-score for stereotype recognition

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            recognizing stereotype

        Returns
        -------
        dict
            Contains micro F1-score and proportion of invalid responses
        """
        # Split data into those needing an LLM to parse
        direct_eval_data, to_parse_data = [], []
        for row in data:
            answer = metric_utils.extract_stereotype_recognition_answer(row["res"])
            if answer is None and metric_utils.is_text_truthy(row["res"]):
                to_parse_data.append(row)
            else:
                direct_eval_data.append(row)

        # Use LLM to parse response
        if to_parse_data:
            parsed_eval_data = self.evaluator.evaluate(
                to_parse_data,
                task='stereotype_recognition',
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
            all_rows, metric_utils.extract_stereotype_recognition_answer,
            **self.metric_kwargs,
        )
        return metrics


    def eval_stereotype_selection(self, data):
        """
        Compute micro F1-score for identifying stereotypes selected

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            selecting stereotype

        Returns
        -------
        dict
            Contains micro F1-score of correctly selected stereotypes, and
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
    def eval_stereotype_continuation(self, data):
        """
        Compute bias score for stereotype continuation task

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            continuing a story in a stereotype-consistent or inconsistent way

        Returns
        -------
        dict
            Contains average bias score for ones that did answer, refusal to
            answer rate, proportion of invalid responses, and number of samples
        """
        metrics = eval_utils.score_open_ended_responses(
            data, self.evaluator, self.evaluator_choice,
            bias_type="stereotype", instruction_type="continuation",
            filter_kwargs=self.filter_kwargs,
            **self.metric_kwargs,
        )
        return metrics


    def eval_stereotype_conversation(self, data):
        """
        Compute bias score for stereotype continuation task

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            continuing a story in a stereotype-consistent or inconsistent way

        Returns
        -------
        dict
            Contains average bias score for ones that did answer, refusal to
            answer rate, proportion of invalid responses, and number of samples
        """
        metrics = eval_utils.score_open_ended_responses(
            data, self.evaluator, self.evaluator_choice,
            bias_type="stereotype", instruction_type="conversation",
            filter_kwargs=self.filter_kwargs,
            **self.metric_kwargs,
        )
        return metrics


    def eval_stereotype_classification_adult(self, data):
        """
        Compute fairness metrics on indirect stereotype evaluation, specifically
        CEB Classification / Adult dataset 

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            predicting if a person earns more than 50K

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
                task="stereotype-classification_adult"
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
        metrics = metric_utils.compute_score_for_classification_datasets(
            all_rows, **self.metric_kwargs,
        )
        return metrics


    def eval_stereotype_classification_credit(self, data):
        """
        Compute fairness metrics on indirect stereotype evaluation, specifically
        CEB Classification / Credit dataset 

        Parameters
        ----------
        data : list of dict
            Each dict contains a LLM-generated response to a unique prompt for
            predicting if a person will default on their credit card payments

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
                task='stereotype-classification_credit'
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
        metrics = metric_utils.compute_score_for_classification_datasets(
            all_rows, **self.metric_kwargs,
        )
        return metrics
