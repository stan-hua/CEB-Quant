# Standard libraries
import json
import logging
import os
import sys
from collections import defaultdict
from glob import glob

# Non-standard libraries
import pandas as pd
import torch
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
        assert os.path.exists(results_dir), f"Directory doesn't exist!\n\tDirectory: {results_dir}"

        # Store attributes
        self.results_dir = results_dir
        self.openai_model = openai_model

        # Get model name
        model_name = os.path.basename(results_dir)

        # Create directory to save evaluations
        self.saved_eval_dir = os.path.join(DIR_SAVED_EVAL, model_name)
        os.makedirs(self.saved_eval_dir, exist_ok=True)

        # Create directory to save metrics
        self.metrics_dir = os.path.join(DIR_METRICS, model_name)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Create paths to store metrics at
        self.stereotype_metric_path = os.path.join(self.metrics_dir, "stereotype_metrics.json")
        self.toxicity_metric_path = os.path.join(self.metrics_dir, "toxicity_metrics.json")

        # Store stereotype and toxicity metrics
        self.dset_stereotype_metrics = defaultdict(dict)
        self.dset_toxicity_metrics = defaultdict(dict)

        # Resume previous evaluations, if specified
        if not overwrite:
            self.reload_computed_metrics()


    def reload_computed_metrics(self):
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
                    LOGGER.info(f"Error occured while evaluating Stereotype Dataset: {dataset_name}\n\tError: {error_msg}")
                    continue

                # Store metrics for dataset / filename
                if dataset_name not in self.dset_stereotype_metrics:
                    self.dset_stereotype_metrics[dataset_name] = {}
                self.dset_stereotype_metrics[dataset_name][fname] = metrics

                # Store metrics for dataset
                json_utils.save_json(dict(self.dset_stereotype_metrics), self.stereotype_metric_path)
                LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{fname}`...DONE")
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}`...DONE")


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
                    LOGGER.info(f"Error occured while evaluating Toxicity Dataset: {dataset_name}\n\tError: {error_msg}")
                    continue

                # Store metrics for dataset / filename
                if dataset_name not in self.dset_toxicity_metrics:
                    self.dset_toxicity_metrics[dataset_name] = {}
                self.dset_toxicity_metrics[dataset_name][fname] = metrics

                # Store metrics for dataset
                json_utils.save_json(dict(self.dset_toxicity_metrics), self.toxicity_metric_path)
                LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{fname}`...DONE")
            LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}`...DONE")


    def save_metric_tables(self):
        """
        Save all metrics as individual tables.

        Saves a table for each bias type (stereotype/toxicity) and
        direct/indirect eval, with columns for each dataset and social group,
        containing the score, dp_diff, and eo_diff, as well as the proportion of
        invalid and refused-to-answer responses (if applicable).

        Table is saved as a CSV file in the `metrics_dir` directory, with the
        filename `metrics_{bias_type}_{task_type}.csv`.
        """
        # Stratify by bias type / direct vs. indirect eval
        for bias_type, task_dict in BIAS_TO_TASK_TYPE_TO_DATASETS.items():
            dataset_to_metric_dict = self.dset_stereotype_metrics if bias_type == "stereotype" else self.dset_toxicity_metrics

            # For each bias type / direct vs. indirect eval, save a table
            for task_type, datasets in task_dict.items():
                row = {}
                for dataset in datasets:
                    # Raise error, if doesn't exist
                    if dataset not in dataset_to_metric_dict:
                        raise ValueError(f"Dataset {dataset} not found in metrics.\n\tFound metrics for: {list(dataset_to_metric_dict.keys())}")
                    social_group_to_metric_dict = dataset_to_metric_dict[dataset]
                    for social_group, metric_dict in social_group_to_metric_dict.items():
                        # Add scores
                        for score_col in ("score", "dp_diff", "eo_diff"):
                            if score_col not in metric_dict:
                                continue
                            metric_val = f"{metric_dict[score_col]:.2f} {metric_dict[score_col+'_ci']}"
                            row[f"{dataset}/{social_group}/{score_col}"] = metric_val

                        # Add percentage of incorrect samples
                        if "prop_rta" in metric_dict:
                            row[f"{dataset}/{social_group}/ %RTA - Invalid"] = f"{100*metric_dict['prop_rta']:.2f} / {100*metric_dict['prop_invalid']:.2f}"
                        elif "prop_invalid" in metric_dict:
                            row[f"{dataset}/{social_group}/ %Invalid"] = f"{100*metric_dict['prop_invalid']:.2f}"

                # Save as table
                df_metrics = pd.DataFrame.from_dict([row])
                save_path = os.path.join(
                    self.metrics_dir,
                    f"metrics_{bias_type}_{task_type}.csv"
                )
                df_metrics.to_csv(save_path, index=False)


################################################################################
#                                  Functions                                   #
################################################################################
def ceb_generate(model_path: str, dataset_name: str = "all", online_model : bool = False):
    """
    Generate LLM responses for specific or all evaluation datasets.

    Parameters
    ----------
    model_path : str
        Path to the model.
    dataset_name : str
        Name of the dataset. If not specififed or "all", generate for all
        datasets.
    online_model : bool
        Whether to use the online model or not (vLLM).
    """
    # Late import to prevent slowdown
    from src.utils.llm_gen_wrapper import LLMGeneration

    # CASE 1: Online APIs (e.g., ChatGPT)
    if online_model:
        print("Using online model")
        llm_gen = LLMGeneration(
            model_path=model_path,
            data_path="./data/",
            dataset_name=dataset_name,
            online_model=True,
            use_deepinfra=False,
            use_replicate=False,
            use_vllm=False,
            repetition_penalty=1.0,
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
            num_gpus=min(torch.cuda.device_count(), 4), 
            max_new_tokens=512, 
            debug=False
        )

    # Perform inference
    llm_gen.infer_dataset()


def ceb_evaluate(results_dir: str, openai_model: str = DEFAULT_OPENAI_MODEL):
    """
    Evaluate LLM responses task for specified or all evaluation datasets.

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

    # Convert to table
    benchmark.save_metric_tables()


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire({
        "generate": ceb_generate,
        "evaluate": ceb_evaluate,
    })