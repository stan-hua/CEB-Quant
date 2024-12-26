# Standard libraries
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from glob import glob

# Non-standard libraries
import pandas as pd
import shutil
import torch
from fire import Fire
from tqdm import tqdm

# Custom libraries
import config
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
        filter_kwargs : bool, optional
            Keyword arguments to filter prompts based on harmfulness, etc.
        evaluator_choice : str, optional
            Choice of evaluator: ("chatgpt", "prometheus").
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
        LOGGER.info(f"Performing full CEB Evaluation...\n\tDirectory: {self.results_dir}")
        task_types = ["direct", "indirect"] if task_type == "all" else [task_type]

        # Overwrite, filter harmful if provided
        if filter_kwargs is not None:
            self.filter_kwargs = filter_kwargs

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
        json_utils.save_json(dict(self.dset_toxicity_metrics), self.toxicity_metric_path)
        LOGGER.info(f"Beginning CEB Evaluation / `{dataset_names}`...DONE")


    def save_metric_tables(self, bias_type="all", task_type="all", save=True):
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
                        for score_col in ("score", "dp_diff", "eo_diff"):
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
        model_path,
        dataset_name="all",
        model_provider="vllm",
        use_chat_template=False,
    ):
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
    use_chat_template : str
        If True, use chat template for local models
    """
    # Late import to prevent slowdown
    from src.utils.llm_gen_wrapper import LLMGeneration

    # Shared keyword arguments
    shared_kwargs = {
        # Provided arguments
        "model_path": model_path,
        "dataset_name": dataset_name,
        "model_provider": model_provider,
        "use_chat_template": use_chat_template,
        # Default arguments
        "data_path": "data/",
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
    llm_gen.infer_dataset()


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
        evaluator_choice=DEFAULT_EVALUATOR,
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
        LOGGER.info(f"[CEB / `{dataset_name}` / `{social_axis}`] Filtering for harmful prompts...")
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
        LOGGER.info(f"[CEB / `{dataset_name}` / `{social_axis}`] Filtering for harmful prompts...")
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
                if simplified_col in ["score", "dp_diff", "eo_diff"]:
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
