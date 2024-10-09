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
from src.generation.generation import STEREOTYPE_DATASETS, TOXICITY_DATASETS
from task.stereotype_eval import StereotypeEval
from task.toxicity_eval import ToxicityEval
from utils import json_utils


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


################################################################################
#                                  Functions                                   #
################################################################################
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
    assert os.path.exists(results_dir), "Directory doesn't exist!"

    LOGGER.info("Beginning CEB Evaluation!")

    # Get model name
    model_name = os.path.basename(results_dir)

    # Create directory to save evaluations
    saved_eval_dir = os.path.join(DIR_SAVED_EVAL, model_name)
    os.makedirs(saved_eval_dir, exist_ok=True)

    # Create directory to save metrics
    metrics_dir = os.path.join(DIR_METRICS, model_name)
    os.makedirs(metrics_dir, exist_ok=True)

    # Create paths to store metrics at
    stereotype_metric_path = os.path.join(metrics_dir, "stereotype_metrics.json")
    toxicity_metric_path = os.path.join(metrics_dir, "toxicity_metrics.json")

    # Load previous stereotype metrics, if they exist
    if os.path.exists(stereotype_metric_path):
        with open(stereotype_metric_path, "r") as f:
            dset_stereotype_metrics = json.load(f)
    else:
        dset_stereotype_metrics = defaultdict(dict)

    # Load previous toxicity metrics, if they exist
    if os.path.exists(toxicity_metric_path):
        with open(toxicity_metric_path, "r") as f:
            dset_toxicity_metrics = json.load(f)
    else:
        dset_toxicity_metrics = defaultdict(dict)

    # 1. Handle all stereotype datasets
    for dataset in STEREOTYPE_DATASETS:
        # TODO: Handle Continuation and Conversation datasets differently
        if any(task_type in dataset for task_type in ["Continuation", "Conversation"]):
            continue

        LOGGER.info(f"Beginning CEB Evaluation / `{dataset}`...")

        # Get all JSONs in inference directory
        json_paths = glob(f"{results_dir}/{dataset}/*.json")

        # Handle each JSON file separately
        for json_path in json_paths:
            fname = ".".join(os.path.basename(json_path).split(".")[:-1])
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset}` / `{fname}`...")
            curr_save_dir = os.path.join(saved_eval_dir, dataset, fname)

            # Skip, if already evaluated
            if fname in dset_stereotype_metrics.get(dataset, {}):
                continue

            # Load inferred data
            infer_data = json_utils.load_json(json_path)

            # Evaluate for stereotype
            evaluator = StereotypeEval(openai_model=openai_model, save_dir=curr_save_dir)
            try:
                metric_val = evaluator.stereotype_recognition_eval(infer_data)
            except Exception as error_msg:
                LOGGER.info(f"Exception has occured with following trace:\n{error_msg}")
                continue

            # Store metrics for dataset / filename
            if dataset not in dset_stereotype_metrics:
                dset_stereotype_metrics[dataset] = {}
            dset_stereotype_metrics[dataset][fname] = metric_val

            # Store metrics for dataset
            json_utils.save_json(dict(dset_stereotype_metrics), stereotype_metric_path)
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset}` / `{fname}`...DONE")
        LOGGER.info(f"Beginning CEB Evaluation / `{dataset}`...DONE")

    # 2. Handle all toxicity benchmarks
    for dataset in TOXICITY_DATASETS:
        LOGGER.info(f"Beginning CEB Evaluation / `{dataset}`...")

        # Get all JSONs in inference directory
        json_paths = glob(f"{results_dir}/{dataset}/*.json")
        for json_path in json_paths:
            fname = ".".join(os.path.basename(json_path).split(".")[:-1])
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset}` / `{fname}`...")
            curr_save_dir = os.path.join(saved_eval_dir, dataset, fname)

            # Skip, if already evaluated
            if fname in dset_toxicity_metrics.get(dataset, {}):
                continue

            # Load inferred data
            infer_data = json_utils.load_json(json_path)

            # Evaluate for toxicity
            evaluator = ToxicityEval(save_dir=curr_save_dir)
            metric_val = evaluator.eval_toxicity(infer_data, resume=True)

            # If null returned, print data
            if metric_val is None:
                LOGGER.error(f"Unexpected error with {dataset} / {fname}")
                continue

            # Store metrics for dataset / filename
            if dataset not in dset_toxicity_metrics:
                dset_toxicity_metrics[dataset] = {}
            dset_toxicity_metrics[dataset][fname] = metric_val

            # Store metrics for dataset
            json_utils.save_json(dict(dset_toxicity_metrics), toxicity_metric_path)
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset}` / `{fname}`...DONE")
        LOGGER.info(f"Beginning CEB Evaluation / `{dataset}`...DONE")


if __name__ == "__main__":
    Fire(main_eval_ceb)
