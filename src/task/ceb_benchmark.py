# Standard libraries
import json
import logging
import os
import sys
from collections import defaultdict
from glob import glob

# Non-standard libraries
from fire import Fire

# Custom libraries
from src.task.stereotype_eval import StereotypeEval
from src.task.toxicity_eval import ToxicityEval
from src.utils import json_utils


################################################################################
#                                    Setup                                     #
################################################################################
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s : %(levelname)s : %(message)s',
)


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Default eval OpenAI model
DEFAULT_OPENAI_MODEL = "gpt-4o-2024-08-06"

# Path to saved GPT-4 evaluations
DIR_SAVED_EVAL = os.path.join(os.path.dirname(__name__), "saved_evaluations")
# Path to stored metrics
DIR_METRICS = os.path.join(os.path.dirname(__name__), "metrics")


# Stratification of Datasets
BIAS_TO_TASK_TYPE_TO_DATASETS = {
    "stereotype": {
        "direct": [f"CEB-{test}-S" for test in ["Recognition", "Selection"]] + [
            # TODO: Handle later
            # "CEB-RB-Recognition",
            # "CEB-WB-Recognition",
            # "CEB-CP-Recognition",
            # "CEB-SS-Recognition",
        ],
        "indirect": [f"CEB-{test}-S" for test in ["Continuation", "Conversation"]] + [
            "CEB-Adult",
            "CEB-Credit",
        ],
    },
    "toxicity": {
        "direct": [f"CEB-{test}-T" for test in ["Recognition", "Selection"]],
        "indirect": [f"CEB-{test}-T" for test in ["Continuation", "Conversation"]] + [
            "CEB-Jigsaw",
        ],
    }
}

################################################################################
#                                   Classes                                    #
################################################################################
class CEBBenchmark:
    """
    CEBBenchmark class.

    Note
    ----
    Universal interface for running and evaluating CEB benchmark
    """

    def __init__(self, results_dir, openai_model=DEFAULT_OPENAI_MODEL,
                 overwrite=False):
        """
        Initialize CEBBenchmark class.

        Parameters
        ----------
        results_dir : str
            Path to directory containing inference results for 1 model
        openai_model : str, optional
            OpenAI model to use for evaluation, by default DEFAULT_OPENAI_MODEL
        overwrite : bool, optional
            If True, overwrite existing results
        """
        assert os.path.exists(results_dir), "Directory doesn't exist!"

        # Store attributes
        self.results_dir = results_dir
        self.openai_model = openai_model

        # Get model name
        model_name = os.path.basename(results_dir)

        # Create directory to save evaluations
        self.saved_eval_dir = os.path.join(DIR_SAVED_EVAL, model_name)
        os.makedirs(self.saved_eval_dir, exist_ok=True)

        # Create directory to save metrics
        metrics_dir = os.path.join(DIR_METRICS, model_name)
        os.makedirs(metrics_dir, exist_ok=True)

        # Create paths to store metrics at
        self.stereotype_metric_path = os.path.join(metrics_dir, "stereotype_metrics.json")
        self.toxicity_metric_path = os.path.join(metrics_dir, "toxicity_metrics.json")

        # Store stereotype and toxicity metrics
        self.dset_stereotype_metrics = defaultdict(dict)
        self.dset_toxicity_metrics = defaultdict(dict)

        # Resume previous evaluations, if specified
        if not overwrite:
            self.reload_previous_eval()


    def reload_previous_eval(self):
        """
        Resume evaluation from previous runs.

        Loads previous evaluations from saved JSON files. If no files exist, the
        metrics dictionaries are initialized as empty.
        """
        # Load previous stereotype metrics, if they exist
        if os.path.exists(self.stereotype_metric_path):
            with open(self.stereotype_metric_path, "r") as f:
                self.dset_stereotype_metrics = json.load(f)

        # Load previous toxicity metrics, if they exist
        if os.path.exists(self.toxicity_metric_path):
            with open(self.toxicity_metric_path, "r") as f:
                self.dset_toxicity_metrics = json.load(f)


    def comprehensive_eval(self):
        """
        Perform a comprehensive evaluation of CEB benchmark.

        Note
        ----
        This function runs both direct and indirect evaluations for both
        stereotype and toxicity tasks. 
        """
        LOGGER.info("Performing full CEB Evaluation...")
        # Perform direct/indirect evaluation evaluation
        for task_type in ["direct", "indirect"]:
            self.stereotype_eval(task_type=task_type)
            self.toxicity_eval(task_type=task_type)
        LOGGER.info("Performing full CEB Evaluation...DONE")


    def stereotype_eval(self, task_type="direct"):
        """
        Evaluate all CEB - Stereotype direct/indirect evaluation datasets

        Parameters
        ----------
        task_type : str, optional
            Task type to evaluate, by default "direct"
        """
        assert task_type in BIAS_TO_TASK_TYPE_TO_DATASETS["stereotype"]
        dataset_names = BIAS_TO_TASK_TYPE_TO_DATASETS["stereotype"][task_type]
        for dataset_name in dataset_names:
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}`...")

            # Get all JSONs in inference directory
            json_paths = glob(f"{self.results_dir}/{dataset_name}/*.json")

            # Handle each JSON file separately
            for json_path in json_paths:
                fname = ".".join(os.path.basename(json_path).split(".")[:-1])
                LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{fname}`...")
                curr_save_dir = os.path.join(self.saved_eval_dir, dataset_name, fname)

                # Skip, if already evaluated
                if fname in self.dset_stereotype_metrics.get(dataset_name, {}):
                    continue

                # Load inferred data
                infer_data = json_utils.load_json(json_path)

                # Evaluate for specific stereotype
                evaluator = StereotypeEval(model=self.openai_model, save_dir=curr_save_dir)
                try:
                    metrics = evaluator.eval_stereotype(dataset_name, infer_data)
                except Exception as error_msg:
                    LOGGER.info(f"Exception has occured with following trace:\n{error_msg}")
                    continue

                # Store metrics for dataset / filename
                if dataset_name not in self.dset_stereotype_metrics:
                    self.dset_stereotype_metrics[dataset_name] = {}
                self.dset_stereotype_metrics[dataset_name][fname] = metrics

                # Store metrics for dataset
                json_utils.save_json(dict(self.dset_stereotype_metrics), self.stereotype_metric_path)
                LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{fname}`...DONE")
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}`...DONE")


    # TODO: Implement this
    def toxicity_eval(self, task_type="direct"):
        """
        Evaluate all CEB - Toxicity direct/indirect evaluation datasets

        Parameters
        ----------
        task_type : str, optional
            Task type to evaluate, by default "direct"
        """
        assert task_type in BIAS_TO_TASK_TYPE_TO_DATASETS["toxicity"]
        dataset_names = BIAS_TO_TASK_TYPE_TO_DATASETS["toxicity"][task_type]
        for dataset_name in dataset_names:
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}`...")

            # Get all JSONs in inference directory
            json_paths = glob(f"{self.results_dir}/{dataset_name}/*.json")

            # Handle each JSON file separately
            for json_path in json_paths:
                fname = ".".join(os.path.basename(json_path).split(".")[:-1])
                LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{fname}`...")
                curr_save_dir = os.path.join(self.saved_eval_dir, dataset_name, fname)

                # Skip, if already evaluated
                if fname in self.dset_toxicity_metrics.get(dataset_name, {}):
                    continue

                # Load inferred data
                infer_data = json_utils.load_json(json_path)

                # Evaluate for specific toxicity
                evaluator = ToxicityEval(save_dir=curr_save_dir, model=self.openai_model)
                try:
                    metrics = evaluator.eval_toxicity(dataset_name, infer_data)
                except Exception as error_msg:
                    LOGGER.info(f"Exception has occured with following trace:\n{error_msg}")
                    continue

                # Store metrics for dataset / filename
                if dataset_name not in self.dset_toxicity_metrics:
                    self.dset_toxicity_metrics[dataset_name] = {}
                self.dset_toxicity_metrics[dataset_name][fname] = metrics

                # Store metrics for dataset
                json_utils.save_json(dict(self.dset_toxicity_metrics), self.toxicity_metric_path)
                LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{fname}`...DONE")
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}`...DONE")


################################################################################
#                                  Functions                                   #
################################################################################
# TODO
def main_eval_ceb(results_dir, openai_model=DEFAULT_OPENAI_MODEL):
    """
    Evaluate CEB

    Parameters
    ----------
    results_dir : str
        Path to directory containing inference results for 1 model
    openai_model : str, optional
        OpenAI model to use for evaluation, by default DEFAULT_OPENAI_MODEL
    """

    # Initialize Benchmark object
    benchmark = CEBBenchmark(results_dir, openai_model=openai_model)

    # Perform comprehensive evaluation
    benchmark.comprehensive_eval()


if __name__ == "__main__":
    Fire(main_eval_ceb)
