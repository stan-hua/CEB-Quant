# Standard libraries
import json
import logging
import os
import sys
import time
from collections import defaultdict
from glob import glob

# Non-standard libraries
import pandas as pd
import shutil
import torch
from fire import Fire
from tqdm import tqdm

# Custom libraries
from config import (
    DIR_GENERATIONS, DIR_EVALUATIONS, DIR_METRICS,
    BIAS_TO_TASK_TYPE_TO_DATASETS, DEFAULT_OPENAI_MODEL,
    PERSPECTIVE_LOCK_FNAME, ALL_DATASETS
)
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

    def __init__(self, results_dir,
                 openai_model=DEFAULT_OPENAI_MODEL,
                 alpha=0.05,
                 overwrite=False):
        """
        Initialize CEBBenchmark class.

        Parameters
        ----------
        results_dir : str
            Path to directory containing inference results for 1 model
        openai_model : str, optional
            OpenAI model to use for evaluation, by default DEFAULT_OPENAI_MODEL
        alpha : float
            Alpha level for confidence interval
        overwrite : bool, optional
            If True, overwrite existing computed metrics. Does NOT overwrite
            existing generations.
        """
        # If exists in `DIR_GENERATIONS`, then prepend directory
        if not os.path.exists(results_dir) and os.path.exists(os.path.join(DIR_GENERATIONS, results_dir)):
            results_dir = os.path.join(DIR_GENERATIONS, results_dir)

        assert os.path.exists(results_dir), f"Directory doesn't exist!\n\tDirectory: {results_dir}"

        # Store attributes
        self.results_dir = results_dir
        self.openai_model = openai_model
        self.alpha = alpha

        # Get model name
        model_name = os.path.basename(results_dir)

        # Create directory to save evaluations
        self.saved_eval_dir = os.path.join(DIR_EVALUATIONS, model_name)
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


    def comprehensive_eval(self, task_type="all", overwrite=False):
        """
        Perform a comprehensive evaluation of CEB benchmark.

        Parameters
        ----------
        task_type : str
            One of ("all", "direct", "indirect"). Chooses datasets to evaluate
        overwrite : bool
            If True, overwrite existing computed metrics. Does NOT overwrite
            existing generations.

        Note
        ----
        This function runs both direct and indirect evaluations for both
        stereotype and toxicity tasks. 
        """
        LOGGER.info(f"Performing full CEB Evaluation...\n\tDirectory: {self.results_dir}")
        task_types = ["direct", "indirect"] if task_type == "all" else [task_type]

        # Perform direct/indirect evaluation evaluation
        for task_type in task_types:
            # 1. Stereotype
            LOGGER.info(f"Starting CEB Evaluation / {task_type} / Stereotype...")
            self.stereotype_eval(task_type=task_type, overwrite=overwrite)
            LOGGER.info(f"Starting CEB Evaluation / {task_type} / Stereotype...DONE")

            # 2. Toxicity
            LOGGER.info(f"Starting CEB Evaluation / {task_type} / Toxicity...")
            self.toxicity_eval(task_type=task_type, overwrite=overwrite)
            LOGGER.info(f"Starting CEB Evaluation / {task_type} / Toxicity...DONE")
        LOGGER.info("Performing full CEB Evaluation...DONE")


    def stereotype_eval(self, task_type="direct", overwrite=False):
        """
        Evaluate all CEB - Stereotype direct/indirect evaluation datasets

        Parameters
        ----------
        task_type : str, optional
            Task type to evaluate, by default "direct"
        overwrite : bool, optional
            If True, overwrite existing computed metrics
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
                if not overwrite and fname in self.dset_stereotype_metrics.get(dataset_name, {}):
                    continue

                # Load inferred data
                infer_data = json_utils.load_json(json_path)

                # Evaluate for specific stereotype
                evaluator = StereotypeEval(
                    model=self.openai_model,
                    save_dir=curr_save_dir,
                    alpha=self.alpha,
                )
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


    def toxicity_eval(self, task_type="direct", overwrite=False):
        """
        Evaluate all CEB - Toxicity direct/indirect evaluation datasets

        Parameters
        ----------
        task_type : str, optional
            Task type to evaluate, by default "direct"
        overwrite : bool, optional
            If True, overwrite existing computed metrics
        """
        # Warn users if Perspective API file lock exists
        if os.path.exists(PERSPECTIVE_LOCK_FNAME):
            LOGGER.warning(
                f"Perspective API lock file exists! `{PERSPECTIVE_LOCK_FNAME}`"
                "\nPlease delete if you're not running multiple of `ceb_benchmark.py` at once!"
                " This may be a result from a previously cancelled run."
            )

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
                if not overwrite and fname in self.dset_toxicity_metrics.get(dataset_name, {}):
                    continue

                # Load inferred data
                infer_data = json_utils.load_json(json_path)

                # Evaluate for specific toxicity
                evaluator = ToxicityEval(
                    save_dir=curr_save_dir,
                    alpha=self.alpha,
                    model=self.openai_model,
                )
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


    def save_metric_tables(self, save=True):
        """
        Save all metrics as individual tables.

        Saves a table for each bias type (stereotype/toxicity) and
        direct/indirect eval, with columns for each dataset and social group,
        containing the score, dp_diff, and eo_diff, as well as the proportion of
        invalid and refused-to-answer responses (if applicable).

        Table is saved as a CSV file in the `metrics_dir` directory, with the
        filename `metrics_{bias_type}_{task_type}.csv`.

        Parameters
        ----------
        save : bool, optional
            If True, save table

        Returns
        -------
        dict
            Mapping of filename to metrics row (pd.DataFrame)
        """
        # Store the save filename to the metrics
        fname_to_metrics = {}

        # Stratify by bias type / direct vs. indirect eval
        for bias_type, task_dict in BIAS_TO_TASK_TYPE_TO_DATASETS.items():
            dataset_to_metric_dict = self.dset_stereotype_metrics if bias_type == "stereotype" else self.dset_toxicity_metrics

            # For each bias type / direct vs. indirect eval, save a table
            for task_type, datasets in task_dict.items():
                row = {}
                for dataset in datasets:
                    # Raise error, if doesn't exist
                    if dataset not in dataset_to_metric_dict:
                        raise ValueError(
                            f"Dataset {dataset} not found in metrics for directory "
                            f"`{self.results_dir}`.\n\tFound metrics for: "
                            f"{list(dataset_to_metric_dict.keys())}"
                        )
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

                # Convert to table
                df_metrics = pd.DataFrame.from_dict([row])
                save_fname = f"metrics_{bias_type}_{task_type}.csv"
                save_path = os.path.join(self.metrics_dir, save_fname)

                # Store metrics
                fname_to_metrics[save_fname] = row

                # Save, if specified
                if save:
                    df_metrics.to_csv(save_path, index=False)

        return fname_to_metrics


################################################################################
#                                  Functions                                   #
################################################################################
def ceb_generate(model_path, dataset_name="all", model_provider="vllm"):
    """
    Generate LLM responses for specific or all evaluation datasets.

    Parameters
    ----------
    model_path : str
        Path to the model.
    dataset_name : str
        Name of the dataset. If not specififed or "all", generate for all
        datasets.
    model_provider : str
        One of local hosting: ("vllm", "huggingface", "vptq"), or one of online
        hosting: ("deepinfra", "replicate", "other")
    """
    # Late import to prevent slowdown
    from src.utils.llm_gen_wrapper import LLMGeneration

    # Shared keyword arguments
    shared_kwargs = {
        "model_path": model_path,
        "data_path": "data/",
        "dataset_name": dataset_name,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "debug": False
    }

    # Add number of GPUs if available
    if torch.cuda.is_available():
        shared_kwargs["num_gpus"] = min(torch.cuda.device_count(), 4)

    # Instantiate LLMGeneration wrapper
    llm_gen = LLMGeneration(model_provider=model_provider, **shared_kwargs)

    # Perform inference
    llm_gen.infer_dataset()


def ceb_evaluate(results_dir, openai_model=DEFAULT_OPENAI_MODEL, **kwargs):
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
    benchmark.comprehensive_eval(**kwargs)

    # Convert to table and save
    benchmark.save_metric_tables(save=True)


def ceb_compare_multiple(
        *results_dirs,
        save_dir="metrics_comparisons",
        pairwise=False,
        model_comparisons=-1,
        openai_model=DEFAULT_OPENAI_MODEL
    ):
    """
    Re-computes metrics with confidence intervals adjusted for multiple
    comparisons.

    Note
    ----
    Usage of `model_comparisons`. If you specify 4 result directories, e.g.,
        - LLaMA 3.1 8B
        - LLaMA 3.1 8B Instruct
        - LLaMA 3.1 70B
        - LLaMA 3.1 70B Instruct
    And you're only comparing between base and instruct models, then you have
    2 model comparisons

    Parameters
    ----------
    results_dirs : *args
        List of result directories to compare
    save_dir : str
        Directory to save aggregated files
    pairwise : bool, optional
        If True, adjusts significance level to account for all possible pairwise
        comparisons. Otherwise, assumes one-vs-all comparisons, by default False
    model_comparisons : bool, optional
        Number of 1:1 comparisons to make with the provided models, excluding
        the number of datasets compared, that is suppllied. If
        `model_comparisons` >= 1, then `pairwise` argument is ignored
    openai_model : str, optional
        Name of OpenAI model to use for evaluation, by default DEFAULT_OPENAI_MODEL
    """
    LOGGER.info(
        "[CEB Benchmark] Performing multiple comparisons with the following directories:\n"
        "\n\t" + "\n\t".join(results_dirs) + "\n"
    )
    assert len(results_dirs) > 1, f"[CEB Benchmark] Comparison requires >1 models to compare! Input: {results_dirs}"

    # Determine number of (model) comparisons, if not provided
    if not model_comparisons:
        # CASE 1: All possible pairwise comparisons (N*(N-1) comparisons)
        # CASE 2: One vs all comparison (N-1 comparisons)
        model_comparisons = (len(results_dirs) * (len(results_dirs)-1)) if pairwise \
            else (len(results_dirs)-1)

    # Determine number of actual comparisons (model comparisons x dataset comparisons)
    total_comparisons = model_comparisons * len(ALL_DATASETS)

    # Compute alpha score
    alpha = 0.05 / (total_comparisons)
    LOGGER.info(f"[CEB Benchmark] Adjusting significance level for pairwise comparisons (a={alpha})")

    # Re-compute metrics with new significance level
    fname_to_accum_metrics = {}
    for results_dir in results_dirs:
        # Initialize Benchmark object
        benchmark = CEBBenchmark(
            results_dir,
            openai_model=openai_model,
            alpha=alpha,
            overwrite=True,
        )

        # Perform comprehensive evaluation
        benchmark.comprehensive_eval()

        # Get metrics stratified by eval
        fname_to_metrics = benchmark.save_metric_tables(save=False)

        # Store metrics
        for fname, metrics in fname_to_metrics.items():
            if fname not in fname_to_accum_metrics:
                fname_to_accum_metrics[fname] = []
            fname_to_accum_metrics[fname].append(metrics)

    # Extract model names from result directories
    model_names = [os.path.basename(d) for d in results_dirs]

    # Save each aggregated table
    for fname, accum_metrics in fname_to_accum_metrics.items():
        df_curr_metrics = pd.DataFrame.from_dict(accum_metrics)
        df_curr_metrics["Model"] = model_names
        cols = df_curr_metrics.columns.tolist()
        df_curr_metrics = df_curr_metrics[["Model"] + cols[:-1]]

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname)
        df_curr_metrics.to_csv(save_path, index=False)


def ceb_find_unfinished(pattern="*"):
    """
    Find all models, matching pattern, who are unfinished with inference

    Parameters
    ----------
    pattern : str
        Pattern to identify model result directories
    """
    model_to_missing_results = defaultdict(list)

    # Iterate over model directories
    for result_dir in tqdm(glob(os.path.join(DIR_GENERATIONS, pattern))):
        model_name = os.path.basename(result_dir)

        # Check each dataset
        for dataset_name in ALL_DATASETS:
            json_paths = glob(os.path.join(result_dir, dataset_name, "*.json"))

            # Early return if missing JSON files
            if not json_paths:
                model_to_missing_results[model_name].extend([
                    f"{dataset_name}/{os.path.basename(json_path)}"
                    for json_path in json_paths
                ])
                break

            # Check if any of the `res` are missing
            for json_path in json_paths:
                # Load json
                infer_data = json_utils.load_json(json_path)
                # Check if any of the `res` are missing
                if any(not row.get("res") for row in infer_data):
                    model_to_missing_results[model_name].append(
                        f"{dataset_name}/{os.path.basename(json_path)}"
                    )
                    print("[CEB Benchmark] Missing results for:", model_name, dataset_name, os.path.basename(json_path))
                    print(f"Res: " + "'".join(['"{}"'.format(row.get("res")) for row in infer_data if not row.get("res")]))

    # Log all incomplete models
    if model_to_missing_results:
        LOGGER.error(
            "[CEB Benchmark] The following models are incomplete:"
            "\n" + json.dumps(model_to_missing_results, indent=4)
        )


def ceb_delete(
        model_regex="*", dataset_regex="*", social_regex="*", file_regex="*",
        inference=False,
        evaluation=False,
    ):
    """
    Delete inference and evaluation results for all models for the following
    dataset.

    Note
    ----
    Used when the benchmark has changed.

    Parameters
    ----------
    model_regex : str
        Regex that matches model name in saved LLM generations folder
    dataset_regex : str
        Regex that matches dataset
    social_regex : str
        Regex that matches social axis (e.g., race, religion, gender, age) or "all"
    file_regex : str
        Regex that matches a specific filename
    inference : bool
        If True, delete inference results (produced by LLMs)
    evaluation : bool
        If True, delete intermediate evaluation files (from Perspective/ChatGPT)
    """
    assert inference or evaluation
    regex_suffix = f"{model_regex}/{dataset_regex}/{social_regex}/{file_regex}"
    print("[CEB Benchmark] Deleting inference and evaluation results matching following regex: ", regex_suffix)
    time.sleep(3)

    # 1. Remove all generations
    if inference:
        for infer_file in tqdm(glob(DIR_GENERATIONS + "/" + regex_suffix)):
            if os.path.isdir(infer_file):
                shutil.rmtree(infer_file)
            else:
                os.remove(infer_file)

    # 2. Remove all saved evaluations
    if evaluation:
        for eval_file in tqdm(glob(DIR_EVALUATIONS + "/" + regex_suffix)):
            if os.path.isdir(eval_file):
                shutil.rmtree(eval_file)
            else:
                os.remove(eval_file)


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire({
        "generate": ceb_generate,
        "evaluate": ceb_evaluate,
        "compare": ceb_compare_multiple,
        "find_unfinished": ceb_find_unfinished,
        "delete": ceb_delete,
    })
