"""
ceb_benchmark.py

Description: Contains high-level functions for the CEB benchmark to:
    (i) Perform inference (text generation) using specified LLMs for each CEB dataset
    (ii) Evaluate LLM generations (programmatically or using a judge LLM)
    (iii) Compute metrics with bootstrapped confidence intervals and print to an
          HTML page with the highlighted significant differences
    (iv) Other utilities: delete certain results/evals, and find which models
         don't have complete generations

Note
----
`./save_data/llm_evaluations` contains all LLM-eval results. Under `llm_evaluations`,
it is divided into `chatgpt` / `prometheus` for storing artifacts from evaluating
open-ended generation responses. Importantly, we store Perplexity API toxicity
scores under `chatgpt` for convenience.

In other words, an `evaluator_choice` of `chatgpt` implies open-ended generation
style datasets for toxicity specifically are evaluated with Perplexity API and
stored under `./save_data/llm_evaluations/chatgpt/...`
"""

# Standard libraries
import concurrent.futures
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from glob import glob

# Non-standard libraries
import numpy as np
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

# Custom libraries
import config
from src.task.stereotype_eval import StereotypeEval
from src.task.toxicity_eval import ToxicityEval
from src.utils import json_utils, metric_utils


################################################################################
#                                    Setup                                     #
################################################################################
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s : %(levelname)s : %(message)s",
)


################################################################################
#                                  Constants                                   #
################################################################################
LOGGER = logging.getLogger(__name__)

# Default evaluator
DEFAULT_EVALUATOR = "chatgpt"


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

    def __init__(
            self, results_dir,
            openai_model=config.DEFAULT_OPENAI_MODEL,
            alpha=0.05,
            filter_kwargs=None,
            evaluator_choice=DEFAULT_EVALUATOR,
            save_metrics=True,
        ):
        """
        Initialize CEBBenchmark class.

        Parameters
        ----------
        results_dir : str
            Path to directory containing inference results for 1 model
        openai_model : str, optional
            OpenAI model to use for evaluation, by default config.DEFAULT_OPENAI_MODEL
        alpha : float
            Alpha level for confidence interval
        filter_kwargs : dict, optional
            Keyword arguments to filter prompts based on harmfulness, etc.
        evaluator_choice : str, optional
            Choice of evaluator: ("chatgpt", "prometheus").
        save_metrics : bool, optional
            If True, save metrics to the metrics directory.
        """
        # If exists in `config.DIR_GENERATIONS`, then prepend directory
        if not os.path.exists(results_dir) and os.path.exists(os.path.join(config.DIR_GENERATIONS, results_dir)):
            results_dir = os.path.join(config.DIR_GENERATIONS, results_dir)

        assert os.path.exists(results_dir), f"Directory doesn't exist!\n\tDirectory: {results_dir}"

        # Store attributes
        self.results_dir = results_dir
        self.openai_model = openai_model
        self.alpha = alpha
        self.filter_kwargs = filter_kwargs
        self.evaluator_choice = evaluator_choice
        self.save_metrics = save_metrics

        # Get model name
        model_name = os.path.basename(results_dir)

        # Create directory to save evaluations
        self.saved_eval_dir = os.path.join(config.DIR_EVALUATIONS, self.evaluator_choice, model_name)

        # Create directory to save metrics
        self.metrics_dir = os.path.join(config.DIR_METRICS, model_name)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Create paths to store metrics at
        self.stereotype_metric_path = os.path.join(self.metrics_dir, "stereotype_metrics.json")
        self.toxicity_metric_path = os.path.join(self.metrics_dir, "toxicity_metrics.json")

        # Store stereotype and toxicity metrics
        self.dset_stereotype_metrics = defaultdict(dict)
        self.dset_toxicity_metrics = defaultdict(dict)

        # Resume previous evaluations, if specified
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


    def comprehensive_eval(self, bias_type="all", task_type="all", filter_kwargs=None, overwrite=False):
        """
        Perform a comprehensive evaluation of CEB benchmark.

        Parameters
        ----------
        bias_type : str
            One of ("all", "stereotype", "toxicity"). Chooses bias types to evaluate
        task_type : str
            One of ("all", "direct", "indirect"). Chooses datasets to evaluate
        filter_kwargs : dict, optional
            Keyword arguments to filter prompts based on harmfulness, etc.
        overwrite : bool
            If True, overwrite existing computed metrics. Does NOT overwrite
            existing generations.

        Note
        ----
        This function runs both direct and indirect evaluations for both
        stereotype and toxicity tasks. 
        """
        LOGGER.info(f"[CEB Benchmark] Performing full CEB Evaluation...\n\tDirectory: {self.results_dir}")
        task_types = ["direct", "indirect"] if task_type == "all" else [task_type]

        # Overwrite, filter harmful if provided
        if filter_kwargs is not None:
            self.filter_kwargs = filter_kwargs
            LOGGER.info(f"[CEB Benchmark] Filter arguments: {filter_kwargs}")

        # Perform direct/indirect evaluation evaluation
        for task_type in task_types:
            # 1. Stereotype
            if bias_type in ["all", "stereotype"]:
                LOGGER.info(f"Starting CEB Evaluation / {task_type} / Stereotype...")
                self.stereotype_eval(task_type=task_type, overwrite=overwrite)
                LOGGER.info(f"Starting CEB Evaluation / {task_type} / Stereotype...DONE")

            # 2. Toxicity
            if bias_type in ["all", "toxicity"]:
                LOGGER.info(f"Starting CEB Evaluation / {task_type} / Toxicity...")
                self.toxicity_eval(task_type=task_type, overwrite=overwrite)
                LOGGER.info(f"Starting CEB Evaluation / {task_type} / Toxicity...DONE")
        LOGGER.info("[CEB Benchmark] Performing full CEB Evaluation...DONE")


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
        assert task_type in config.BIAS_TO_TASK_TYPE_TO_DATASETS["stereotype"]
        dataset_names = config.BIAS_TO_TASK_TYPE_TO_DATASETS["stereotype"][task_type]

        # Get class attributes
        class_attrs = {k:v for k,v in self.__dict__.items() if not callable(v)}
        class_attrs["overwrite"] = overwrite

        # NOTE: If using Prometheus, can only be done serially
        num_workers = min(config.MAX_WORKER_AUTOEVAL, os.cpu_count())
        num_workers = 1 if self.evaluator_choice == "prometheus" else num_workers
        LOGGER.info(f"Beginning CEB Evaluation / `{dataset_names}`...with {num_workers} workers")
        # CASE 1: Serial evaluation
        if num_workers <= 1:
            for dataset_name in dataset_names:
                # Get all JSONs in inference directory
                json_paths = glob(f"{self.results_dir}/{dataset_name}/*.json")
                for json_path in json_paths:
                    ret = stereotype_process_json(class_attrs, dataset_name, json_path)
                    social_axis = extract_social_axis(json_path)
                    metrics = ret[dataset_name][social_axis]
                    json_utils.update_nested_dict(
                        self.dset_stereotype_metrics, dataset_name, social_axis,
                        value=metrics,
                    )
        # CASE 2: Parallelize evaluation across datasets
        else:
            with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
                futures = []
                for dataset_name in dataset_names:
                    # Get all JSONs in inference directory
                    json_paths = glob(f"{self.results_dir}/{dataset_name}/*.json")
                    futures.extend([
                        executor.submit(stereotype_process_json, class_attrs, dataset_name, json_path)
                        for json_path in json_paths
                    ])

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    ret = future.result()
                    # Skip errored results
                    if ret is None:
                        continue

                    # Store metrics computed for a single dataset / JSON
                    for dataset_name, axis_to_metrics in ret.items():
                        for social_axis, metrics in axis_to_metrics.items():
                            json_utils.update_nested_dict(
                                self.dset_stereotype_metrics, dataset_name, social_axis,
                                value=metrics,
                            )

        # Store metrics for dataset
        if self.save_metrics:
            json_utils.save_json(dict(self.dset_stereotype_metrics), self.stereotype_metric_path)

        LOGGER.info(f"Beginning CEB Evaluation / `{dataset_names}`...DONE")


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
        if os.path.exists(config.PERSPECTIVE_LOCK_FNAME):
            LOGGER.warning(
                f"Perspective API lock file exists! `{config.PERSPECTIVE_LOCK_FNAME}`"
                "\nPlease delete if you're not running multiple of `ceb_benchmark.py` at once!"
                " This may be a result from a previously cancelled run."
            )

        assert task_type in config.BIAS_TO_TASK_TYPE_TO_DATASETS["toxicity"]
        dataset_names = config.BIAS_TO_TASK_TYPE_TO_DATASETS["toxicity"][task_type]

        # Get class attributes
        class_attrs = {k:v for k,v in self.__dict__.items() if not callable(v)}
        class_attrs["overwrite"] = overwrite

        # NOTE: If using Prometheus, can only be done serially
        num_workers = min(config.MAX_WORKER_AUTOEVAL, os.cpu_count())
        num_workers = 1 if self.evaluator_choice == "prometheus" else num_workers
        LOGGER.info(f"Beginning CEB Evaluation / `{dataset_names}`...with {num_workers} workers")
        # CASE 1: Serial evaluation
        if num_workers <= 1 or self.evaluator_choice == "prometheus":
            for dataset_name in dataset_names:
                # Get all JSONs in inference directory
                json_paths = glob(f"{self.results_dir}/{dataset_name}/*.json")
                for json_path in json_paths:
                    ret = toxicity_process_json(class_attrs, dataset_name, json_path)
                    social_axis = extract_social_axis(json_path)
                    metrics = ret[dataset_name][social_axis]
                    json_utils.update_nested_dict(
                        self.dset_toxicity_metrics, dataset_name, social_axis,
                        value=metrics,
                    )
        # CASE 2: Parallelize evaluation across datasets
        else:
            with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
                futures = []
                for dataset_name in dataset_names:
                    # Get all JSONs in inference directory
                    json_paths = glob(f"{self.results_dir}/{dataset_name}/*.json")
                    futures.extend([
                        executor.submit(toxicity_process_json, class_attrs, dataset_name, json_path)
                        for json_path in json_paths
                    ])

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    ret = future.result()
                    # Skip errored results
                    if ret is None:
                        continue

                    # Store metrics computed for a single dataset / JSON
                    for dataset_name, axis_to_metrics in ret.items():
                        for social_axis, metrics in axis_to_metrics.items():
                            json_utils.update_nested_dict(
                                self.dset_toxicity_metrics, dataset_name, social_axis,
                                value=metrics,
                            )

        # Store metrics
        if self.save_metrics:
            json_utils.save_json(dict(self.dset_toxicity_metrics), self.toxicity_metric_path)
        LOGGER.info(f"Beginning CEB Evaluation / `{dataset_names}`...DONE")


    def flatten_metrics(self, higher_is_better=False):
        """
        Return stereotype and toxicity metrics flattened into one dictionary.

        Parameters
        ----------
        higher_is_better : bool, optional
            If True, store negative of metrics that are "lower is better", by
            default False

        Returns
        -------
        dict
            Flattened dictionary of metrics, where the keys are "[Dataset] - [Social Axis] / score"
        """
        all_metrics = self.dset_stereotype_metrics + self.dset_toxicity_metrics
        flattened_metrics = {}
        for dataset_name, social_axis_to_metrics in all_metrics.items():
            for social_axis, metrics_dict in social_axis_to_metrics.items():
                task_type = dataset_name.split("-")[1]
                score_key = f"{dataset_name} - {social_axis} / score"
                # CASE 1: If score column exists
                if "score" in metrics_dict:
                    metric_val = metrics_dict["score"]
                    # If specified, ensure the higher the metric the better
                    if higher_is_better and task_type not in ["Selection", "Recognition"]:
                        metric_val = -metric_val

                    # Store value
                    flattened_metrics[score_key] = metric_val
                # CASE 2: Average of fairness metrics
                elif "scaled_dp_diff" in metrics_dict:
                    # Store average of both DP and EO difference
                    dp_val = metrics_dict["scaled_dp_diff"]
                    eo_val = metrics_dict["scaled_eo_diff"]

                    # If either DP/EO diff is missing, store None
                    if dp_val is None or eo_val is None:
                        flattened_metrics[score_key] = None
                    else:
                        metric_val = (dp_val + eo_val) / 2
                        # If specified, ensure the higher the metric the better
                        if higher_is_better:
                            metric_val = -metric_val
                        flattened_metrics[score_key] = metric_val
                # NOTE: The following may only occur, if neither metric is stored
                else:
                    raise RuntimeError(f"[Flatten CEB Metrics] Missing score key in metrics for {dataset_name} / {social_axis}")
        return flattened_metrics


    def save_metric_tables(self, bias_type="all", task_type="all", save=True):
        """
        Save all metrics as individual tables.

        Saves a table for each bias type (stereotype/toxicity) and
        direct/indirect eval, with columns for each dataset and social group,
        containing the score, scaled_dp_diff, and scaled_eo_diff, as well as the proportion of
        invalid and refused-to-answer responses (if applicable).

        Table is saved as a CSV file in the `metrics_dir` directory, with the
        filename `metrics_{bias_type}_{task_type}.csv`.

        Parameters
        ----------
        bias_type : str, optional
            Bias type to save metrics for, by default "all"
        task_type : str, optional
            Task type to save metrics for, by default "all"
        save : bool, optional
            If True, save table

        Returns
        -------
        dict
            Mapping of filename to metrics row (pd.DataFrame)
        """
        # Store the save filename to the metrics
        fname_to_metrics = {}
        valid_bias_types = ["stereotype", "toxicity"] if bias_type == "all" else [bias_type]
        valid_task_types = ["direct", "indirect"] if task_type == "all" else [task_type]

        # Stratify by bias type / direct vs. indirect eval
        for bias_type, task_dict in config.BIAS_TO_TASK_TYPE_TO_DATASETS.items():
            # Skip, if not valid bias type
            if bias_type not in valid_bias_types:
                continue

            # For each bias type / direct vs. indirect eval, save a table
            dataset_to_metric_dict = self.dset_stereotype_metrics if bias_type == "stereotype" else self.dset_toxicity_metrics
            for task_type, datasets in task_dict.items():
                # Skip, if not valid task type
                if task_type not in valid_task_types:
                    continue

                # Save bias type / task type
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
                        for score_col in ("score", "scaled_dp_diff", "scaled_eo_diff"):
                            if score_col not in metric_dict:
                                continue
                            metric_val = f"{metric_dict[score_col]:.2f} {metric_dict[score_col+'_ci']}"
                            row[f"{dataset}/{social_group}/{score_col}"] = metric_val

                        # Add percentages for refusal to answer, invalid responses or positive responses
                        if "prop_rta" in metric_dict:
                            row[f"{dataset}/{social_group}/ %RTA - Invalid"] = f"{100*metric_dict['prop_rta']:.2f} / {100*metric_dict['prop_invalid']:.2f}"
                        elif "prop_invalid" in metric_dict:
                            row[f"{dataset}/{social_group}/ %Invalid"] = f"{100*metric_dict['prop_invalid']:.2f}"
                        elif "prop_positive" in metric_dict:
                            row[f"{dataset}/{social_group}/ %Positive"] = f"{100*metric_dict['prop_positive']:.2f}"

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
def ceb_generate(
        model_path_or_name,
        dataset_name="all",
        model_provider="vllm",
        use_chat_template=False,
    ):
    """
    Generate LLM responses for specific or all evaluation datasets.

    Parameters
    ----------
    model_path_or_name : str
        Path to model, or model (nick)name in `config.py`
    dataset_name : str
        Name of the dataset. If not specififed or "all", generate for all
        datasets.
    model_provider : str
        One of local hosting: ("vllm", "huggingface", "vptq"), or one of online
        hosting: ("deepinfra", "replicate", "other")
    use_chat_template : str
        If True, use chat template for local models
    """
    # Late import to prevent slowdown
    from src.utils.llm_gen_wrapper import LLMGeneration

    # Shared keyword arguments
    shared_kwargs = {
        # Provided arguments
        "model_path_or_name": model_path_or_name,
        "dataset_name": dataset_name,
        "model_provider": model_provider,
        "use_chat_template": use_chat_template,
        # Default arguments
        "data_path": config.DIR_CEB_DATA,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "debug": False,
    }

    # Add number of GPUs if available
    if torch.cuda.is_available():
        shared_kwargs["num_gpus"] = min(torch.cuda.device_count(), 4)

    # Instantiate LLMGeneration wrapper
    llm_gen = LLMGeneration(**shared_kwargs)

    # Perform inference
    LOGGER.info(f"[CEB Generate] Performing inference for {model_path_or_name}...")
    llm_gen.infer_dataset()
    LOGGER.info(f"[CEB Generate] Performing inference for {model_path_or_name}...DONE")


def ceb_evaluate(
        results_dir,
        openai_model=config.DEFAULT_OPENAI_MODEL,
        evaluator_choice=DEFAULT_EVALUATOR,
        **kwargs):
    """
    Evaluate LLM responses task for specified or all evaluation datasets.

    Parameters
    ----------
    results_dir : str
        Path to directory containing inference results for 1 model
    openai_model : str, optional
        OpenAI model to use for evaluation, by default config.DEFAULT_OPENAI_MODEL
    evaluator_choice : str, optional
        Evaluator to use, by default DEFAULT_EVALUATOR
    **kwargs : Any
        Keyword arguments to pass into `comprehensive_eval`
    """
    # Initialize Benchmark object
    benchmark = CEBBenchmark(results_dir, openai_model=openai_model, evaluator_choice=evaluator_choice)

    # Perform comprehensive evaluation
    benchmark.comprehensive_eval(**kwargs)

    # Convert to table and save
    benchmark.save_metric_tables(save=True)


def ceb_compare_multiple(
        *results_dirs,
        bias_type="all",
        task_type="all",
        save_dir=config.DIR_COMPARISONS,
        pairwise=False,
        model_comparisons=-1,
        **kwargs,
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
    bias_type : str
        Bias type to compare, by default "all"
    task_type : str
        Task type to compare, by default "all"
    save_dir : str
        Directory to save aggregated files
    pairwise : bool, optional
        If True, adjusts significance level to account for all possible pairwise
        comparisons. Otherwise, assumes one-vs-all comparisons, by default False
    model_comparisons : bool, optional
        Number of 1:1 comparisons to make with the provided models, excluding
        the number of datasets compared, that is suppllied. If
        `model_comparisons` >= 1, then `pairwise` argument is ignored
    **kwargs : Keyword arguments
        Keyword arguments to pass into CEBBenchmark
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
    total_comparisons = model_comparisons * len(config.ALL_DATASETS)

    # Compute alpha score
    alpha = 0.05 / (total_comparisons)
    LOGGER.info(f"[CEB Benchmark] Adjusting significance level for (model x dataset) comparisons (a={alpha})")

    # Re-compute metrics with new significance level
    fname_to_accum_metrics = {}
    for results_dir in results_dirs:
        # Initialize Benchmark object
        benchmark = CEBBenchmark(
            results_dir,
            alpha=alpha,
            **kwargs,
        )

        # Perform comprehensive evaluation
        # NOTE: Need to overwrite to re-do bootstrapped confidence intervals
        benchmark.comprehensive_eval(
            bias_type=bias_type,
            task_type=task_type,
            overwrite=True,
        )

        # Get metrics stratified by eval
        fname_to_metrics = benchmark.save_metric_tables(
            bias_type=bias_type,
            task_type=task_type,
            save=False,
        )

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


def ceb_concatenate_comparisons(*save_dirs, save_dir=config.DIR_COMPARISONS):
    """
    Concatenate evaluation metrics from multiple directories into a single CSV file 
    per evaluation type.

    Note
    ----
    This function reads metric CSV files from multiple specified directories, concatenates 
    them with an empty row between each dataset to improve readability, and saves the 
    aggregated results for each evaluation type (stereotype direct/indirect, toxicity 
    direct/indirect) into a specified save directory.

    Parameters
    ----------
    *save_dirs : *args
        List of directories containing metric CSV files to concatenate.
    save_dir : str, optional
        Directory to save the concatenated CSV files, by default config.DIR_COMPARISONS.
    """
    accum_dict = {
        "stereotype_direct": [],
        "stereotype_indirect": [],
        "toxicity_direct": [],
        "toxicity_indirect": [],
    }
    for curr_save_dir in save_dirs:
        if not os.path.exists(curr_save_dir) and os.path.exists(os.path.join(save_dir, curr_save_dir)):
            LOGGER.info(f"[CEB Concatenate Comparisons] Using updated directory: {os.path.join(save_dir, curr_save_dir)}")
            curr_save_dir = os.path.join(save_dir, curr_save_dir)
        assert os.path.exists(curr_save_dir), f"Directory provided doesn't exist! \nInvalid: {curr_save_dir}"
        for eval_key in list(accum_dict.keys()):
            df_curr = pd.read_csv(os.path.join(curr_save_dir, f"metrics_{eval_key}.csv"))

            # Process certain columns
            for col in df_curr.columns:
                # If column ends with "%Positive", ensure it's rounded
                if col.endswith("%Positive") and pd.api.types.is_float_dtype(df_curr[col]):
                    df_curr[col] = df_curr[col].round()

            # Color significant differences
            df_curr = color_significant_differences(df_curr, curr_save_dir)
            accum_dict[eval_key].append(df_curr)

    # Concatenate tables with space (row) in between
    for eval_key, accum_tables in tqdm(accum_dict.items()):
        # Get first table
        df_accum_style = accum_tables[0]
        # Create empty row with the same columns
        empty_row = pd.DataFrame({col: [""] for col in df_accum_style.columns})
        # Insert empty row between questions
        accum_tables = insert_between_elements(accum_tables, empty_row.copy())

        # Add borders to first table
        df_accum_style = (df_accum_style
            .hide(axis="index")
            .set_properties(**{'text-align': 'center'})
            .map(lambda x: 'border: 1px solid black;')
            .map(lambda x: 'border-right: 3px solid black;', subset=pd.IndexSlice[:, ["Model"]])
        )

        # Concatenate iteratively
        for table_idx, df_curr in enumerate(accum_tables[1:]):
            # Convert to styled dataframe, if not already
            if not isinstance(df_curr, pd.io.formats.style.Styler):
                df_curr = df_curr.style

            # Add borders and center text, if not an empty row
            df_curr = (df_curr
                .hide(axis="index")
                .set_properties(**{'text-align': 'center'})
                .map(lambda x: 'border: 1px solid black;')
                .map(lambda x: 'border-right: 3px solid black;', subset=pd.IndexSlice[:, ["Model"]])
            )
            try:
                df_accum_style.concat(df_curr)
            except Exception as error_msg:
                LOGGER.error(
                    f"Error occured on `{eval_key}` - Table {table_idx} \n"
                    f"Accum Columns: {df_accum_style.columns.tolist()}\n"
                    f"All Columns: {df_curr.columns.tolist()}")
                raise error_msg

        # Save to directory
        save_path = os.path.join(save_dir, f"accum_{eval_key}.csv")
        LOGGER.info(f"Saving to `{save_path}`")
        # df_accum.to_csv(save_path, index=False)
        df_accum_style.to_html(save_path.replace(".csv", ".html"), index=False, border=1)


def ceb_find_unfinished(pattern="*", eval_models=None, generation=False, evaluation=False):
    """
    Find all models, matching pattern, who are unfinished with inference

    Parameters
    ----------
    pattern : str
        If `generation`, pattern to identify model result directories.
        If `evaluation`, pattern containing the evaluator choice and optionally
        the model names
    eval_models : list
        If `evaluation`, list of model names to check if already evaluated
    generation : bool
        If True, check generations, based on pattern.
    evaluation : bool
        If True, check evaluation, based on model list.
    """
    model_to_missing_results = defaultdict(list)

    # 1. Generation
    if generation:
        LOGGER.info("[CEB Find Unfinished] Finding models with unfinished generations")
        # Iterate over model directories
        for result_dir in tqdm(glob(os.path.join(config.DIR_GENERATIONS, pattern))):
            model_name = os.path.basename(result_dir)

            # Check each dataset
            for dataset_name in config.ALL_DATASETS:
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
                        LOGGER.info(f"[CEB Benchmark] Missing results for: {os.path.join(model_name, dataset_name, os.path.basename(json_path))}")
                        LOGGER.info(f"Res: " + "'".join(['"{}"'.format(row.get("res")) for row in infer_data if not row.get("res")]))

        # Log all incomplete models
        if model_to_missing_results:
            LOGGER.error(
                "[CEB Find Unfinished - Generations] The following models are incomplete:"
                "\n" + json.dumps(model_to_missing_results, indent=4)
            )

    # 2. Evaluation
    if evaluation:
        assert eval_models, "[CEB Find Unfinished - Evaluations] Please provide list of models via `eval_models`, if searching through evaluations!"
        eval_models = list(set(eval_models))

        LOGGER.info("[CEB Find Unfinished - Evaluations] Finding models with unfinished evaluations")
        model_dirs = glob(os.path.join(config.DIR_GENERATIONS, pattern))
        split_func = lambda path: path.split("chatgpt" + os.sep)[1] if "chatgpt" in path else path.split("prometheus" + os.sep)[1]
        model_names = [os.path.dirname(split_func(path)) for path in model_dirs]

        # Filter valid model directories
        is_valid = [model_name in set(eval_models) for model_name in model_names]
        valid_model_dirs = [
            (model_dir, model_names[idx])
            for idx, model_dir in enumerate(model_dirs)
            if is_valid[idx]
        ]

        # Datasets and social axes to check
        indirect_datasets = [f"CEB-{test}-{bias}" for test in ["Continuation", "Conversation"] for bias in ["S", "T"]]
        social_axes = ["age", "gender", "race", "religion"]

        # Get all missing directories
        missing_evals = set(eval_models) - set(model_names)

        # Check each model directory for those missing evals
        unfinished_evals = []
        for curr_model_dir, curr_model_name in valid_model_dirs:
            unfinished = False
            # Check if eval for each dataset is present
            for dataset_name in indirect_datasets:
                curr_dataset_dir = os.path.join(curr_model_dir, dataset_name)
                if not os.path.exists(curr_dataset_dir) or unfinished:
                    unfinished = True
                    break
                # Check if eval for each dataset / social axis is present
                for social_axis in social_axes:
                    curr_axis_dir = os.path.join(curr_dataset_dir, social_axis)
                    if not os.path.exists(curr_axis_dir) and not os.path.listdir(curr_axis_dir):
                        unfinished = True
                        break
            if unfinished:
                unfinished_evals.append(curr_model_name)

        # Add missing evaluations
        unfinished_evals.extend(list(missing_evals))

        # Print models with unfinished evals
        unfinished_evals = sorted(unfinished_evals)
        unfinished_evals_str = "\n\t".join(unfinished_evals)
        LOGGER.error(f"[CEB Find Unfinished - Evaluations] The following models are missing evaluations: \n\t{unfinished_evals_str}")


def ceb_delete(
        model_regex="*", dataset_regex="*", social_regex="*", file_regex="*",
        evaluator_choice="*",
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
    evaluator_choice : str
        Evaluator choice
    inference : bool
        If True, delete inference results (produced by LLMs)
    evaluation : bool
        If True, delete intermediate evaluation files (from Perspective/ChatGPT)
    """
    assert inference or evaluation, "At least one of `inference` or `evaluation` must be True"

    # 1. Remove all generations
    if inference:
        regex_suffix = f"{model_regex}/{dataset_regex}/{file_regex}"
        print("[CEB Delete] Deleting inference results matching following regex: ", regex_suffix)
        time.sleep(3)
        for infer_file in tqdm(glob(config.DIR_GENERATIONS + "/" + regex_suffix)):
            if os.path.isdir(infer_file):
                shutil.rmtree(infer_file)
            else:
                os.remove(infer_file)

    # 2. Remove all saved evaluations
    if evaluation:
        regex_suffix = f"{evaluator_choice}/{model_regex}/{dataset_regex}/{social_regex}/{file_regex}"
        print("[CEB Delete] Deleting evaluation results matching following regex: ", regex_suffix)
        time.sleep(3)
        for eval_file in tqdm(glob(config.DIR_EVALUATIONS + "/" + regex_suffix)):
            if os.path.isdir(eval_file):
                shutil.rmtree(eval_file)
            else:
                os.remove(eval_file)


def ceb_compute_prometheus_correlation(bin_chatgpt_scores=False):
    """
    Compute Spearman correlation between Prometheus evaluation scores and
    ChatGPT stereotype bias / Perplexity API toxicity scores.

    Parameters
    ----------
    bin_chatgpt_scores : bool
        If True, bin ChatGPT stereotype bias scores to [0, 33, 66, 99], as done
        in Prometheus Eval
    """
    # Find all Prometheus eval files
    path_components = [config.DIR_EVALUATIONS, "prometheus"] + ["*"] * 4
    file_regex = os.path.join(*path_components)
    matching_paths = glob(file_regex)
    baseline_dir = os.path.join(config.DIR_EVALUATIONS, "chatgpt")
    valid_fname_regex = "*eval_progress.json"
    accum_eval_scores = []
    for curr_path in tqdm(matching_paths):
        # Find path to JSON file containing baseline evaluation (using ChatGPT/Perplexity)
        eval_subdir = os.path.dirname(curr_path.split("prometheus" + os.sep)[1])
        baseline_paths = glob(os.path.join(baseline_dir, eval_subdir, valid_fname_regex))
        # Continue, if no corresponding file path
        if not baseline_paths:
            continue
        assert len(baseline_paths) == 1, f"More than 1 baseline eval file not expected! Files Found: {baseline_paths}"
        # Load baseline evaluations
        df_baseline_eval = pd.DataFrame(json_utils.load_json(baseline_paths[0]))
        if "bias_score" in df_baseline_eval.columns:
            extracted_scores = df_baseline_eval["bias_score"].map(
                metric_utils.extract_number_from_0_to_99).fillna(-1).to_numpy()
            # Round off values to nearest score bin
            if bin_chatgpt_scores:
                bins = np.array([-1, 0, 33, 66, 99])
                indices = np.abs(extracted_scores[:, None] - bins).argmin(axis=1)
                mapped_scores = bins[indices].astype(float)
                mapped_scores[mapped_scores == -1] = np.nan
                extracted_scores = mapped_scores
            df_baseline_eval["base_eval_score"] = mapped_scores
        else:
            # Skip, if toxicity key not available
            if "toxicity" not in df_baseline_eval.columns:
                continue
            df_baseline_eval["base_eval_score"] = df_baseline_eval["toxicity"]
        # Load Prometheus evaluations
        df_prometheus_eval = pd.DataFrame(json_utils.load_json(curr_path))
        df_prometheus_eval["prometheus_eval_score"] = df_prometheus_eval["eval_res"].map(
            metric_utils.extract_prometheus_bias_score)
        # Join on prompt
        df_eval = pd.merge(
            df_baseline_eval, df_prometheus_eval,
            how="right", on="prompt", suffixes=("__dup", ""),
        )
        # Remove duplicate columns
        df_eval = df_eval[[col for col in df_eval.columns.tolist() if not col.endswith("__dup")]]
        # Append dataset name and social axis
        df_eval["social_axis"] = os.path.basename(eval_subdir)
        df_eval["dataset_name"] = os.path.basename(os.path.dirname(eval_subdir))
        df_eval["model_name"] = os.path.basename(os.path.dirname(os.path.dirname(eval_subdir)))
        # Append eval scores
        keep_cols = [
            "model_name", "dataset_name", "social_axis", "axis", "bucket", "rta", "res",
            "base_eval_score", "prometheus_eval_score"
        ]
        accum_eval_scores.append(df_eval[keep_cols])

    # Accumulate all eval score
    df_accum = pd.concat(accum_eval_scores, ignore_index=True)

    # Drop rows with invalid Prometheus eval scores (i.e., Prometheus-Eval failed)
    mask = df_accum["prometheus_eval_score"] != -1
    df_accum = df_accum[mask]

    # Check how well refusal to answer matches
    df_accum["chatgpt_rta"] = df_accum["base_eval_score"].isna()
    df_accum["prometheus_rta"] = df_accum["prometheus_eval_score"].isna()

    # Count correspondence in refusal to answer
    rta_cols = ["prometheus_rta", "chatgpt_rta"]
    print(df_accum[rta_cols].value_counts(normalize=True))

    # Compute correlation between scores
    # NOTE: Only compute on rows where scores in both rows is not present
    eval_cols = ["base_eval_score", "prometheus_eval_score"]
    mask = ~df_accum[eval_cols].isna().any(axis=1)
    overall_corr = df_accum.loc[mask, eval_cols].corr("spearman").iloc[1, 0]
    print(f"[Prometheus x ChatGPT/Toxicity] Correlation: {overall_corr:.2f}")

    # Compute correlation, stratified by dataset and social axis
    group_cols = ["dataset_name", "social_axis"]
    print("[Prometheus x ChatGPT/Toxicity] Correlation Stratified by Dataset / Social Axis:")
    print(df_accum[mask].groupby(group_cols).apply(lambda df: round(df[eval_cols].corr("spearman").iloc[1, 0], 2)))

    # Check the proportion of base eval scores for stereotyping
    mask = df_accum["dataset_name"].str.endswith("-S")
    print("[ChatGPT] Stereotyping Bias Eval Scores:")
    print(df_accum[mask].groupby("social_axis").apply(lambda df: df["base_eval_score"].value_counts(normalize=True)))
    print("[Prometheus] Stereotyping Bias Eval Scores:")
    print(df_accum[mask].groupby("social_axis").apply(lambda df: df["prometheus_eval_score"].value_counts(normalize=True)))


def ceb_compute_fairness_vs_lm_eval_correlation():
    # 1. LM-Eval
    # Get names of all models with LM-Eval
    paths = glob(os.path.join(config.DIR_LM_EVAL, "*"))
    lm_eval_models = [
        os.path.basename(path)
        for path in paths if not path.endswith(".db")
    ]
    benchmark_to_metric_cols = {
        "arc_challenge": ["acc,none"],
        "hellaswag": ["acc,none"],
        "piqa": ["acc,none"],
        "truthfulqa_mc1": ["acc,none"],
        "lambada_openai": ["acc,none", "perplexity,none"],
        "mmlu_pro": ["exact_match,custom-extract"],
    }

    # Get all the metrics associated with them
    accum_lm_eval_metrics = []
    for model_name in lm_eval_models:
        json_paths = glob(os.path.join(config.DIR_LM_EVAL, model_name, "*", "*.json"))
        # If multiple results files exist, take the latest
        if len(json_paths) > 1:
            latest_json_path, latest_time = None, None
            for json_path in json_paths:
                time_str = json_path.split("_")[1].split(".")[0]
                curr_time = datetime.striptime(time_str, "%Y-%m-%dT%H-%M-%S")
                if latest_time is None or curr_time > latest_time:
                    latest_json_path = json_path
            json_path = latest_json_path
        else:
            json_path = json_paths[0]
        # Load metric file and extract metrics
        # NOTE: Additionally, average over accuracies
        metric_json = json_utils.load_json(json_path)["results"]
        curr_model_metrics = {"model": model_name}
        accum_accuracies = []
        for benchmark, metric_cols in benchmark_to_metric_cols.items():
            for metric_col in metric_cols:
                assert metric_col in metric_json[benchmark], f"[LM-Eval] `{metric_col}` is missing from benchmark `{benchmark}` in the results file! \n\tModel: {model_name}"
                metric_val = metric_json[benchmark][metric_col]
                curr_model_metrics[f"{benchmark} / {metric_col.split(',')[0]}"] = metric_val
                if "acc" in metric_col:
                    accum_accuracies.append(metric_val)
        curr_model_metrics["avg_acc"] = round(sum(accum_accuracies)/len(accum_accuracies), 4)
        accum_lm_eval_metrics.append(curr_model_metrics)

    # 2. Get all the fairness metrics for these models
    accum_ceb_metrics = []
    for model_name in lm_eval_models:
        # Compute metrics for each dataset (with no confidence interval)
        benchmark = CEBBenchmark(model_name, alpha=0, evaluator_choice="prometheus", save_metrics=False)
        benchmark.comprehensive_eval(overwrite=True)
        curr_ceb_metrics = benchmark.flatten_metrics(higher_is_better=True)
        # Compute average score among datasets that don't use fairness gap metrics
        filtered_scores = [v for k,v in curr_ceb_metrics.items() if k.split("-")[1] not in ["Adult", "Credit", "Jigsaw"]]
        curr_ceb_metrics["avg_score"] = sum(filtered_scores) / len(filtered_scores)
        curr_ceb_metrics["model"] = model_name
        accum_ceb_metrics.append(curr_ceb_metrics)




################################################################################
#                   Dataset / Social Axis - Level Processing                   #
################################################################################
def stereotype_process_json(class_attrs, dataset_name, json_path):
    """
    Evaluate the following dataset for stereotypes across all prompts and
    social axes.

    Parameters
    ----------
    class_attrs : dict
        Class attributes from `CEBBenchmark` object
    dataset_name : str
        Name of the dataset
    json_path : str
        Path to the JSON file containing the prompt information

    Returns
    -------
    dset_to_axis_to_metrics : dict
        A dictionary mapping from dataset name to social axis to stereotype
        metrics
    """
    saved_eval_dir = class_attrs["saved_eval_dir"]
    dset_stereotype_metrics = class_attrs["dset_stereotype_metrics"]
    openai_model = class_attrs["openai_model"]
    evaluator_choice = class_attrs["evaluator_choice"]
    alpha = class_attrs["alpha"]
    filter_kwargs = class_attrs["filter_kwargs"]
    overwrite = class_attrs["overwrite"]

    social_axis = extract_social_axis(json_path)
    LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{social_axis}`...")
    curr_save_dir = os.path.join(saved_eval_dir, dataset_name, social_axis)

    # Skip, if already evaluated
    if not overwrite and social_axis in dset_stereotype_metrics.get(dataset_name, {}):
        return dset_stereotype_metrics[dataset_name][social_axis]

    # Load inferred data
    infer_data = json_utils.load_json(json_path)

    # If specified, add `is_harmful` key to distinguish harmful prompts
    if filter_kwargs:
        LOGGER.info(f"[CEB / `{dataset_name}` / `{social_axis}`] Filtering data with: {filter_kwargs}")
        infer_data = add_is_harmful_key(
            dataset_name, social_axis,
            infer_data,
            model_choice="chatgpt",
        )

    # Evaluate for specific stereotype
    evaluator = StereotypeEval(
        model=openai_model,
        evaluator_choice=evaluator_choice,
        save_dir=curr_save_dir,
        alpha=alpha,
        filter_kwargs=filter_kwargs,
    )
    try:
        metrics = evaluator.eval_stereotype(dataset_name, infer_data)
    except Exception as error_msg:
        LOGGER.info(f"Error occurred while evaluating Stereotype Dataset: {dataset_name}\n\tError: {error_msg}")
        LOGGER.error(traceback.format_exc())
        return None

    LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{social_axis}`...DONE")
    # Return metrics
    dset_to_axis_to_metrics = {dataset_name: {social_axis: metrics}}
    return dset_to_axis_to_metrics


def toxicity_process_json(class_attrs, dataset_name, json_path):
    """
    Evaluate the following dataset for toxicity across all prompts and
    social axes.

    Parameters
    ----------
    class_attrs : dict
        Class attributes from `CEBBenchmark` object
    dataset_name : str
        Name of the dataset
    json_path : str
        Path to the JSON file containing the prompt information

    Returns
    -------
    dset_to_axis_to_metrics : dict
        A dictionary mapping from dataset name to social axis to toxicity
        metrics
    """
    saved_eval_dir = class_attrs["saved_eval_dir"]
    dset_toxicity_metrics = class_attrs["dset_toxicity_metrics"]
    openai_model = class_attrs["openai_model"]
    evaluator_choice = class_attrs["evaluator_choice"]
    alpha = class_attrs["alpha"]
    filter_kwargs = class_attrs["filter_kwargs"]
    overwrite = class_attrs["overwrite"]

    social_axis = extract_social_axis(json_path)
    LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{social_axis}`...")
    curr_save_dir = os.path.join(saved_eval_dir, dataset_name, social_axis)

    # Skip, if already evaluated
    if not overwrite and social_axis in dset_toxicity_metrics.get(dataset_name, {}):
        return dset_toxicity_metrics[dataset_name][social_axis]

    # Load inferred data
    infer_data = json_utils.load_json(json_path)

    # If specified, add `is_harmful` key to distinguish harmful prompts
    if filter_kwargs:
        LOGGER.info(f"[CEB / `{dataset_name}` / `{social_axis}`] Filtering data with: {filter_kwargs}")
        infer_data = add_is_harmful_key(
            dataset_name, social_axis,
            infer_data,
            model_choice="chatgpt",
        )

    # Evaluate for specific toxicity
    evaluator = ToxicityEval(
        model=openai_model,
        evaluator_choice=evaluator_choice,
        save_dir=curr_save_dir,
        alpha=alpha,
        filter_kwargs=filter_kwargs,
    )
    try:
        metrics = evaluator.eval_toxicity(dataset_name, infer_data)
    except Exception as error_msg:
        LOGGER.info(f"Error occurred while evaluating Toxicity Dataset: {dataset_name}\n\tError: {error_msg}")
        LOGGER.error(traceback.format_exc())
        return None

    LOGGER.info(f"Beginning CEB Evaluation / `{dataset_name}` / `{social_axis}`...DONE")
    # Return metrics
    dset_to_axis_to_metrics = {dataset_name: {social_axis: metrics}}
    return dset_to_axis_to_metrics


################################################################################
#                               Helper Functions                               #
################################################################################
def color_significant_differences(df_results, metric_comparisons_dir):
    """
    Colors significant differences in a DataFrame of results.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame of results from CEB benchmark evaluation
    metric_comparisons_dir : str
        Directory that contains results for a specific metric comparison.
        Used to determine which anchor models to compare against.

    Returns
    -------
    pd.DataFrame
        DataFrame with background color style strings for each cell.
    """
    dir_name = os.path.basename(metric_comparisons_dir)

    # If metric comparison dir has no anchor models, then create warning and return
    if dir_name not in config.ANCHOR_MODELS:
        LOGGER.warning(
            f"[CEB Benchmark] Metric comparison directory `{dir_name}` has no "
            "anchor models defined! Consider adding in `config.py / ANCHOR_MODELS` "
            "for automatic coloring! Skipping for now.."
        )
        return df_results

    # Get anchor models
    anchor_models = config.ANCHOR_MODELS[dir_name]

    # Color significant differences
    df_styled = df_results.style.apply(
        highlight_significant, anchor_models=anchor_models,
        axis=None)

    return df_styled


def parse_score(score):
    """
    Parse a string of form "mean [lower, upper]" into individual mean,
    lower, and upper values.

    Parameters
    ----------
    score : str
        String of form "mean [lower, upper]"

    Returns
    -------
    mean : float
        Mean value
    lower : float
        Lower confidence interval
    upper : float
        Upper confidence interval

    Raises
    ------
    ValueError
        If mean, lower, and upper are all -1, suggests placeholder metric
    """
    mean, ci = score.split(" [")
    ci = ci.rstrip("]").split(", ")
    mean, lower, upper = float(mean), float(ci[0]), float(ci[1])
    # Raise value error, if all values are -1, suggests placeholder
    if mean == -1 and lower == -1 and upper == -1:
        raise ValueError("Found placeholder metric!")
    return mean, lower, upper


def compare_scores(anchor, compare, better_direction="higher"):
    """
    Compare two scores to determine if their pairwise difference is significant
    and return a corresponding background color based on the comparison.

    Parameters
    ----------
    anchor : str
        Score string of the anchor in the format "mean [lower, upper]".
    compare : str
        Score string of the comparator in the format "mean [lower, upper]".
    better_direction : str, optional
        Indicates the preferred direction of a better score ("higher" or "lower"),
        by default "higher".

    Returns
    -------
    str
        Background color as a string. Possible values are:
        - config.STYLE_WORSE if the anchor is better than the comparator
          and better_direction is "higher", or vice versa for "lower".
        - config.STYLE_BETTER if the comparator is better than the anchor
          and better_direction is "higher", or vice versa for "lower".
        - config.STYLE_EQUIVALENT if the pairwise difference is not significant.

    Raises
    ------
    ValueError
        If parsing of the score strings fails, indicating placeholder metrics.
    """
    # NOTE: If parsing fails, this suggests that either anchor or comparator had
    #       no valid responses, so placeholder metric was placed
    try:
        anchor_mean, anchor_lower, anchor_upper = parse_score(anchor)
        compare_mean, compare_lower, compare_upper = parse_score(compare)
    except ValueError:
        return ""

    # Check if pairwise difference is significant
    if (anchor_mean < compare_lower or anchor_mean > compare_upper) and (compare_mean < anchor_lower or compare_mean > anchor_upper):
        if anchor_mean > compare_mean:
            return config.STYLE_WORSE if better_direction == "higher" else config.STYLE_BETTER
        else:
            return config.STYLE_BETTER if better_direction == "higher" else config.STYLE_WORSE
    return config.STYLE_EQUIVALENT


def compare_rta_invalid(anchor, compare):
    """
    Compare two strings of form "RTA / Invalid" and return a corresponding background
    color based on the comparison.

    Parameters
    ----------
    anchor : str
        String of form "RTA / Invalid" for the anchor model.
    compare : str
        String of form "RTA / Invalid" for the comparator model.

    Returns
    -------
    str
        Background color as a string. Possible values are:
        - config.STYLE_WORSE if refusal to answer rate increased (worse).
        - config.STYLE_BETTER if refusal to answer rate decreased (better).
        - config.STYLE_EQUIVALENT if neither refusal to answer rate nor invalid
          rate showed significant difference.
        - config.STYLE_BETTER_AND_WORSE if both refusal to answer rate and invalid rate
          showed significant difference in opposite directions.
    """
    anchor_rta, anchor_invalid = map(float, anchor.split(" / "))
    compare_rta, compare_invalid = map(float, compare.split(" / "))

    # Check difference
    rta_diff = compare_rta - anchor_rta
    invalid_diff = compare_invalid - anchor_invalid
    rta_diff_significant = abs(rta_diff) >= 5
    invalid_diff_significant = abs(invalid_diff) >= 5

    # CASE 0: If neither are significant, then color orange
    if not rta_diff_significant and not invalid_diff_significant:
        return config.STYLE_EQUIVALENT

    # SPECIAL CASE 1: If both are significant in different directions, then color as cyan
    if (rta_diff_significant and invalid_diff_significant) and (rta_diff > 0 and invalid_diff < 0):
        return config.STYLE_BETTER_AND_WORSE

    # CASE 2: If RTA or Invalid is significant, then color
    value_diff = rta_diff if rta_diff_significant else invalid_diff
    # CASE 1: Refusal to answer rate increased (worse)
    if value_diff > 0:
        return config.STYLE_WORSE
    # CASE 2: Refusal to answer rate decreased (better)
    else:
        return config.STYLE_BETTER


# Function to apply the comparison
def highlight_significant(df_results, anchor_models):
    """
    Highlights rows in a DataFrame that are significantly different from anchor models.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame with columns that are either scores, percentage positive, or refusal to answer.
    anchor_models : list[str]
        List of model names to use as anchor models.

    Returns
    -------
    pd.DataFrame
        DataFrame with background color style strings for each cell.
    """
    styles = pd.DataFrame("", index=df_results.index, columns=df_results.columns)
    for anchor in anchor_models:
        anchor_idx = df_results.index[df_results["Model"] == anchor][0]
        styles.loc[anchor_idx] = config.STYLE_EQUIVALENT
        for idx in range(anchor_idx, len(df_results)):
            if idx != anchor_idx and df_results.loc[idx, "Model"] in anchor_models:
                break
            for col in df_results.columns.tolist():
                # Get value
                comparison_val = df_results.loc[idx, col]

                # CASE 1: Score column
                simplified_col = os.path.basename(col).strip().lower()
                if simplified_col in ["score", "scaled_dp_diff", "scaled_eo_diff"]:
                    better_direction = "higher" if any(dset in col for dset in ["Recognition", "Selection"]) else "lower"
                    styles.loc[idx, col] = compare_scores(
                        df_results.loc[anchor_idx, col], comparison_val,
                        better_direction=better_direction,
                    )
                # CASE 2: Percentage positive column
                elif simplified_col == "%positive":
                    styles.loc[idx, col] = config.STYLE_WORSE if comparison_val == 100 else config.STYLE_EQUIVALENT
                # CASE 3: Refusal to Answer Column
                elif simplified_col == "%rta - invalid":
                    styles.loc[idx, col] = compare_rta_invalid(df_results.loc[anchor_idx, col], comparison_val)
    return styles


def insert_between_elements(lst, value):
    """
    Insert the given value between elements in the given list.

    Parameters
    ----------
    lst : list
        List of elements
    value : any
        Value to insert between elements

    Returns
    -------
    list
        List with value inserted between elements
    """
    result = []
    for i in range(len(lst)):
        result.append(lst[i])
        if i < len(lst) - 1:
            result.append(value)
    return result


def add_is_harmful_key(dataset_name, social_axis, data, model_choice="chatgpt"):
    """
    Add `is_harmful` key to each row to denote prompts that were detected to be
    harmful.

    Parameters
    ----------
    dataset_name : str
        Name of dataset
    social_axis : str
        Social axis (e.g., age)
    data : list of dict
        List of question/response dicts
    model_choice : str, optional
        Choice of model to detect prompt harmfulness. If ChatGPT/WildGuard, use
        binary prompt harmfulness classifications. If GUS-Net, use binarized
        token bias classifications. 
    """
    # Raise error for not implemented methods
    if model_choice in ["gus_net"]:
        raise NotImplementedError(f"Method `{model_choice}` is not yet implemented!")

    # Get directory containing prompt harmfulness scores
    harmful_path = os.path.join(
        config.DIR_WILDGUARD_HARMFUL, model_choice,
        dataset_name, social_axis, f"{model_choice}_eval.json"
    )

    # Load harmful predictions
    harmful_predictions = json_utils.load_json(harmful_path)

    # CASE 1: ChatGPT/Wildguard
    prompt_whitelist = []
    if model_choice in ["chatgpt", "wildguard"]:
        eval_key = f"{model_choice}_bias_analysis"
        # Extract if prompt was harmful
        for row in harmful_predictions:
            pattern = re.compile("Harmful request: (yes|no)", re.DOTALL)
            match = pattern.search(row[eval_key])
            assert match, "[add_is_harmful_key] Failed to extract prompt harmfulness from string!"
            if match and match.group(1) == "yes":
                prompt_whitelist.append(row["prompt"])

    # Convert to set for faster querying
    prompt_whitelist = set(prompt_whitelist)

    # Add `is_harmful` key to distinguish data with biased prompts
    for row in data:
        row["is_harmful"] = row["prompt"] in prompt_whitelist

    return data


def extract_social_axis(json_path):
    # NOTE: Assumes path is in the LLM generation directory
    return ".".join(os.path.basename(json_path).split(".")[:-1])


################################################################################
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire({
        "generate": ceb_generate,
        "evaluate": ceb_evaluate,
        "compare": ceb_compare_multiple,
        "format_comparisons": ceb_concatenate_comparisons,
        "find_unfinished": ceb_find_unfinished,
        "delete": ceb_delete,
    })
