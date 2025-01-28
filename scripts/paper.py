"""
test_hypotheses.py

Description: Used to perform analysis used in the paper
"""

# Standard libraries
import json

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from arch.bootstrap import IIDBootstrap

# Custom libraries
from ceb_benchmark import *


################################################################################
#                                  Constants                                   #
################################################################################
SEED = 42


################################################################################
#                                  Functions                                   #
################################################################################
# RQ0. What is the impact of instruction-finetuning?
def ceb_check_impact_of_instruction(save_dir="temp"):
    # Mapping of instruct to base model
    instruct_to_base = {
        "llama3.2-1b-instruct": "llama3.2-1b",
        "llama3.2-3b-instruct": "llama3.2-3b",
        "llama3.1-8b-instruct": "llama3.1-8b",
        "llama3.1-70b-instruct": "llama3.1-70b",
        "mistral-v0.3-7b-instruct": "mistral-v0.3-7b",
        "qwen2-7b-instruct": "qwen2-7b",
        "qwen2-72b-instruct": "qwen2-72b",
        "qwen2.5-0.5b-instruct": "qwen2.5-0.5b",
        "qwen2.5-1.5b-instruct": "qwen2.5-1.5b",
        "qwen2.5-3b-instruct": "qwen2.5-3b",
        "qwen2.5-7b-instruct": "qwen2.5-7b",
        "qwen2.5-14b-instruct": "qwen2.5-14b",
        "qwen2.5-32b-instruct": "qwen2.5-32b",
        "qwen2.5-72b-instruct": "qwen2.5-72b",
    }

    ret = load_pairwise_differences(instruct_to_base)
    df_valid = ret["accum_valid"]
    df_invalid = ret["accum_invalid"]
    df_accum = pd.concat([df_valid, df_invalid], ignore_index=True)

    results_accum = {}

    # 1. What happens if you only consider the average
    # The difference is small!
    results_accum["Average Fairness (Across Valid/Invalid Samples)"] = {
        "Base": df_accum["score_base"].mean(),
        "Instruct": df_accum["score_modified"].mean(),
        "Difference (Agg)": df_accum["score_modified"].mean() - df_accum["score_base"].mean(),
        "Difference (Pairwise)": df_accum["score_diff"].mean(),
    }

    # But base models spew garbage most of the time
    results_accum["Response Distributions"] = {
        "Base": df_accum["response_type_base"].value_counts(normalize=True),
        "Instruct": df_accum["response_type_modified"].value_counts(normalize=True),
    }
    # But the evaluation models would consider their output largely biased
    results_accum["Bias Distributions"] = {
        "Base": df_accum["score_base"].value_counts(normalize=True),
        "Instruct": df_accum["score_modified"].value_counts(normalize=True),
    }

    # When filtering on valid responses, the average bias scores drop
    results_accum["Average Fairness (Across Valid Samples)"] = {
        "Base": df_valid["score_base"].mean(),
        "Instruct": df_valid["score_modified"].mean(),
        "Difference (Agg)": df_valid["score_modified"].mean() - df_valid["score_base"].mean(),
        "Difference (Pairwise)": df_valid["score_diff"].mean(),
    }

    # Get average scores for valid responses (only in base) and (only in instruct)
    df_accum["is_valid_base"] = df_accum["response_type_base"].map(lambda x: "Valid" in x)
    df_accum["is_valid_modified"] = df_accum["response_type_modified"].map(lambda x: "Valid" in x)
    df_accum.loc[df_accum["is_valid_base"], "score_base"].mean()
    df_accum.loc[df_accum["is_valid_modified"], "score_modified"].mean()

    # Does instruction FT improve performance?
    results_accum["Does instruction FT improve fairness?"] = metric_utils.grouped_hypothesis_test(
        df_valid, group_cols=None, score_col="score_diff",
        side=">",
    )

    # TODO: Demonstrate that confidence intervals are smaller for pairwise differences
    # Compute CI for difference in average metrics
    # bootstrap_ci([df_accum["score_diff"].to_numpy()])
    # Compute CI for pairwise differences
    # bootstrap_diff_ci(
    #     [df_accum["score_modified"].to_numpy()],
    #     [df_accum["score_base"].to_numpy()],
    # )


# TODO: Instead of per-question variation, look into per-attribute (sensitive group) variation
def ceb_check_impact_of_quantization(save_dir="temp"):
    # Get all evaluated models
    all_models = os.listdir(os.path.join(config.DIR_EVALUATIONS, "prometheus", str(PROMETHEUS_PROMPT_VER)))
    # Get the base model for every model
    base_models = [extract_model_metadata_from_name(m)["base_model"] for m in all_models]

    # Filter for model pairs that exist
    quantized_to_base = {
        q_model: b_model
        for q_model, b_model in dict(zip(all_models, base_models)).items()
        if b_model != q_model
    }

    ret = load_pairwise_differences(quantized_to_base)
    df_valid = ret["accum_valid"]
    df_invalid = ret["accum_invalid"]
    df_accum = pd.concat([df_valid, df_invalid], ignore_index=True)
    results_accum = {}

    # Print average difference with 95% CI
    results_accum["Average Fairness (Across Valid Samples)"] = {
        "avg": df_valid["score_diff"].mean(),
        "ci": bootstrap_ci([df_valid["score_diff"].to_numpy()]),
    }

    # Get the average reason for invalid responses
    df_accum["is_valid_base"] = df_accum["response_type_base"].map(lambda x: "Valid" in x)
    df_accum["is_valid_modified"] = df_accum["response_type_modified"].map(lambda x: "Valid" in x)
    df_accum["is_alignment_base"] = df_accum["response_type_base"].map(lambda x: "Alignment" in x)
    df_accum["is_alignment_modified"] = df_accum["response_type_modified"].map(lambda x: "Alignment" in x)

    # Look at invalid responses
    mask = (~df_accum["is_valid_base"]) & (df_accum["is_alignment_base"])
    df_accum.loc[mask, "res_base"]


    # But base models spew garbage most of the time
    results_accum["Response Validity Distributions"] = {
        "Full-Precision": df_accum["is_valid_base"].value_counts(normalize=True),
        "Quantized": df_accum["is_valid_modified"].value_counts(normalize=True),
    }
    results_accum["Response Alignment Distributions (Valid Samples)"] = {
        "Full-Precision": df_accum["is_alignment_base"].value_counts(normalize=True),
        "Quantized": df_accum["is_alignment_modified"].value_counts(normalize=True),
    }

    # But the evaluation models would consider their output largely biased
    results_accum["Bias Distributions"] = {
        "Base": df_valid["score_base"].value_counts(normalize=True),
        "Instruct": df_valid["score_modified"].value_counts(normalize=True),
    }

    # Does quantization worsen performance?
    results_accum["Does quantization worsen fairness?"] = metric_utils.grouped_hypothesis_test(
        df_valid, group_cols=None, score_col="score_diff",
        side="<",
    )

    # Get scores separately
    df_full_precision = df_valid[["model_base", "score_base"]].rename(columns={"model_base": "model", "score_base": "score"})
    df_full_precision["Is Quantized"] = False
    df_quantized = df_valid[["model_modified", "score_modified"]].rename(columns={"model_modified": "model", "score_modified": "score"})
    df_quantized["Is Quantized"] = True
    df_accum_scores = pd.concat([df_full_precision, df_quantized], ignore_index=True)
    df_accum_scores["score"] = df_accum_scores["score"].astype(int)
    # df_accum_scores = df_accum_scores[df_accum_scores["score"] != 0]

    # TODO: Create a histogram plot for distribution of fairness scores
    viz_utils.set_theme(tick_scale=1.7, figsize=(10, 10))
    fig, axs = plt.subplots(2, 2)
    viz_utils.catplot(
        df_accum_scores,
        plot_type="count",
        x="score", hue="Is Quantized",
        xlabel="Distribution of Fairness Scores",
        ylabel="Count",
        title="Impact of Quantization",
        legend=True,
        save_dir=save_dir,
        save_fname="fp_vs_quantized-fairness_dist-countplot.png",
    )

    # Plot average difference in fairness
    viz_utils.catplot(
        df_valid,
        plot_type="count", stat="percent",
        x="score_diff",
        xlabel="Point Change in Fairness",
        ylabel="Percentage",
        title="Impact of Quantization",
        legend=True,
        save_dir=save_dir,
        save_fname="per_question_impact_of_quant-histplot.png",
    )

    get_stats = lambda df: pd.DataFrame({"mean": df.mean().round(2), "std": df.std().round(2)})

    # TODO: Analyze in the context of dataset / social axis / descriptors
    df_valid.groupby("descriptor")["score_diff"].mean().sort_values().round(2)
    df_valid.groupby("social_axis")["score_diff"].mean().sort_values().round(2)
    df_valid.groupby("descriptor")["score_diff"].std().sort_values().round(2)
    df_valid.groupby("descriptor")["score_diff"].std().sort_values().round(2)

    
    # Add model details from name
    model_metadata = pd.DataFrame(df_valid["model_modified"].map(extract_model_metadata_from_name).tolist())
    df_valid = pd.concat([df_valid, model_metadata], axis=1)

    # TODO: 


def ceb_compute_fairness_vs_lm_eval_correlation(save_dir="."):
    # 1. LM-Eval
    # Get names of all models with LM-Eval
    paths = glob(os.path.join(config.DIR_LM_EVAL, "*"))
    lm_filter_models = [
        os.path.basename(path)
        for path in paths if os.path.isdir(path)
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
    for model_name in lm_filter_models:
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
    for model_name in lm_filter_models:
        # Compute metrics for each dataset (with no confidence interval)
        benchmark = CEBBenchmark(model_name, alpha=0, evaluator_choice="prometheus", save_metrics=False)
        benchmark.comprehensive_eval(task_type="indirect", overwrite=True)
        curr_ceb_metrics = benchmark.flatten_metrics(higher_is_better=True)
        # Compute average score among datasets that don't use fairness gap metrics
        # filtered_scores = [v for k,v in curr_ceb_metrics.items() if k.split("-")[1].strip() not in ["Adult", "Credit", "Jigsaw"]]
        filtered_scores = [v for k,v in curr_ceb_metrics.items() if k.split("-")[1].strip() in ["Continuation", "Conversation"]]
        filtered_stereotype_scores = [v for k,v in curr_ceb_metrics.items() if k.split(" ")[0].endswith("-S") and k.split("-")[1].strip() in ["Continuation", "Conversation"]]
        filtered_toxicity_scores = [v for k,v in curr_ceb_metrics.items() if k.split(" ")[0].endswith("-T") and k.split("-")[1].strip() in ["Continuation", "Conversation"]]
        curr_ceb_metrics["avg_score"] = sum(filtered_scores) / len(filtered_scores)
        curr_ceb_metrics["avg_stereotype_score"] = sum(filtered_stereotype_scores) / len(filtered_stereotype_scores)
        curr_ceb_metrics["avg_toxicity_score"] = sum(filtered_toxicity_scores) / len(filtered_toxicity_scores)
        curr_ceb_metrics["model"] = model_name
        accum_ceb_metrics.append(curr_ceb_metrics)

    # Combine dataframes
    df_lm_eval = pd.DataFrame(accum_lm_eval_metrics).set_index("model")
    df_ceb = pd.DataFrame(accum_ceb_metrics).set_index("model")
    df_all = pd.merge(df_ceb, df_lm_eval, on="model", how="inner").reset_index()

    # Add model details from name
    model_metadata = pd.DataFrame(df_all["model"].map(extract_model_metadata_from_name).tolist())
    df_all = pd.concat([df_all, model_metadata], axis=1)

    # LM-Eval columns
    lm_eval_cols = [
        'hellaswag / acc',
        'piqa / acc',
        'truthfulqa_mc1 / acc',
        'lambada_openai / acc',
        'lambada_openai / perplexity',
        'mmlu_pro / exact_match'
    ]

    # 1. Only full-precision instruct models
    # Filter for base model vs. not base model
    is_full_precision = df_all["base_model"] == df_all["model"]
    is_instruct = df_all["instruct_tuned"]
    df_full_precision = df_all.loc[is_full_precision & is_instruct].copy()
    # Normalize score columns
    scaler = StandardScaler()
    cols = ["avg_stereotype_score", "avg_toxicity_score", "avg_score", "avg_acc"] + lm_eval_cols
    df_full_precision[cols] = scaler.fit_transform(df_full_precision[cols])

    # Create scatterplot
    viz_utils.set_theme(tick_scale=1.7, figsize=(10, 10),)
    viz_utils.numplot(
        df_full_precision,
        x="avg_score", y="avg_acc", hue="Model Size (GB)",
        size="Model Size (GB)", sizes=(300, 3000),
        plot_type="scatter",
        xlabel="Avg. CEB Fairness Score",
        ylabel="Avg. LM-Eval Accuracy",
        title="Full-Precision Instruct Models",
        palette="flare",
        save_dir=save_dir,
        save_fname="fp16_instruct_models-scatterplot.png",
    )
    # Compute correlation between CEB Fairness and individual benchmarks
    fairness_corrs = df_full_precision[cols].corr().iloc[0:3, 3:]
    fairness_corrs = fairness_corrs.round(2)
    fairness_corrs.to_csv(f"{save_dir}/fp16_instruct_models-corrs.csv")

    # 2. Plot base and quantized models
    df_together = df_all.copy()
    # Remove AQLM, since they skew the distribution
    # df_together = df_together[~df_together["model"].str.contains("aqlm")]
    # df_together[["model", "avg_score", "avg_acc"]]
    # Normalize score columns
    scaler = StandardScaler()
    df_together[cols] = scaler.fit_transform(df_together[cols])
    df_together["Is Full Precision"] = df_together["base_model"] == df_together["model"]
    # Create scatterplot
    viz_utils.set_theme(tick_scale=1.7, figsize=(10, 10),)
    viz_utils.numplot(
        df_together,
        x="avg_score", y="avg_acc", hue="Is Full Precision",
        size="Model Size (GB)", sizes=(300, 3000),
        plot_type="scatter",
        xlabel="Avg. CEB Fairness Score",
        ylabel="Avg. LM-Eval Accuracy",
        title="Full-Precision & Quantized Instruct Models",
        palette="viridis",
        save_dir=save_dir,
        save_fname="all_models-scatterplot.png",
    )
    # Compute correlation between CEB Fairness and individual benchmarks
    fairness_corrs = df_together[cols].corr().iloc[0:3, 3:]
    fairness_corrs = fairness_corrs.round(2)
    fairness_corrs.to_csv(f"{save_dir}/all_models-corrs.csv")

    # 3. Create plot of fairness against effective size
    viz_utils.set_theme(tick_scale=1.7, figsize=(10, 10),)
    viz_utils.numplot(
        df_together,
        x="Model Size (GB)", y="avg_score",
        hue="Is Full Precision",
        size="Model Size (GB)", sizes=(300, 3000),
        plot_type="scatter",
        xlabel="Model Size (GB)",
        ylabel="Avg. CEB Fairness Score",
        title="Full-Precision & Quantized Instruct Models",
        palette="viridis",
        save_dir=save_dir,
        legend=True,
        save_fname="all_models-fairness_by_size-scatterplot.png",
    )


def ceb_check_smoothquant(save_dir="temp"):
    models = os.listdir(os.path.join(config.DIR_EVALUATIONS, "prometheus"))
    smooth_to_base = {
        "llama3.1-70b-instruct-lc-smooth-rtn-w4a16": "llama3.1-70b-instruct-lc-rtn-w4a16",
        "llama3.1-70b-instruct-lc-smooth-rtn-w8a16": "llama3.1-70b-instruct-lc-rtn-w8a16",
        "llama3.1-70b-instruct-lc-smooth-rtn-w8a8": "llama3.1-70b-instruct-lc-rtn-w8a8",
        "llama3.1-8b-instruct-lc-smooth-rtn-w4a16": "llama3.1-8b-instruct-lc-rtn-w4a16",
        "llama3.1-8b-instruct-lc-smooth-rtn-w8a16": "llama3.1-8b-instruct-lc-rtn-w8a16",
        "llama3.1-8b-instruct-lc-smooth-rtn-w8a8": "llama3.1-8b-instruct-lc-rtn-w8a8",
        "llama3.1-8b-instruct-lc-smooth-gptq-w4a16": "nm-llama3.1-8b-instruct-gptq-w4a16",
        "llama3.2-1b-instruct-lc-smooth-gptq-w4a16": "llama3.2-1b-instruct-lc-gptq-w4a16",
        "llama3.2-1b-instruct-lc-smooth-rtn-w4a16": "llama3.2-1b-instruct-lc-rtn-w4a16",
        "llama3.2-1b-instruct-lc-smooth-rtn-w8a8": "llama3.2-1b-instruct-lc-rtn-w8a8",
        "llama3.2-3b-instruct-lc-smooth-gptq-w4a16": "llama3.2-3b-instruct-lc-gptq-w4a16",
        "llama3.2-3b-instruct-lc-smooth-rtn-w4a16": "llama3.2-3b-instruct-lc-rtn-w4a16",
        "llama3.2-3b-instruct-lc-smooth-rtn-w8a8": "llama3.2-3b-instruct-lc-rtn-w8a8",
        "ministral-8b-instruct-lc-smooth-gptq-w4a16": "ministral-8b-instruct-lc-gptq-w4a16",
        "ministral-8b-instruct-lc-smooth-rtn-w4a16": "ministral-8b-instruct-lc-rtn-w4a16",
        "ministral-8b-instruct-lc-smooth-rtn-w8a8": "ministral-8b-instruct-lc-rtn-w8a8",
        "mistral-small-22b-instruct-lc-smooth-gptq-w4a16": "mistral-small-22b-instruct-lc-gptq-w4a16",
        "mistral-small-22b-instruct-lc-smooth-rtn-w4a16": "mistral-small-22b-instruct-lc-rtn-w4a16",
        "mistral-small-22b-instruct-lc-smooth-rtn-w8a8": "mistral-small-22b-instruct-lc-rtn-w8a8",
        "qwen2-7b-instruct-lc-smooth-rtn-w4a16": "qwen2-7b-instruct-lc-rtn-w4a16",
        "qwen2-7b-instruct-lc-smooth-rtn-w8a8": "qwen2-7b-instruct-lc-rtn-w8a8",
        "qwen2-72b-instruct-lc-smooth-rtn-w4a16": "qwen2-72b-instruct-lc-rtn-w4a16",
        "qwen2-72b-instruct-lc-smooth-rtn-w8a8": "qwen2-72b-instruct-lc-rtn-w8a8",
        "qwen2.5-0.5b-instruct-lc-smooth-rtn-w4a16": "qwen2.5-0.5b-instruct-lc-rtn-w4a16",
        "qwen2.5-0.5b-instruct-lc-smooth-rtn-w8a8": "qwen2.5-0.5b-instruct-lc-rtn-w8a8",
        "qwen2.5-1.5b-instruct-lc-smooth-rtn-w4a16": "qwen2.5-1.5b-instruct-lc-rtn-w4a16",
        "qwen2.5-1.5b-instruct-lc-smooth-rtn-w8a8": "qwen2.5-1.5b-instruct-lc-rtn-w8a8",
        "qwen2.5-3b-instruct-lc-smooth-rtn-w4a16": "qwen2.5-3b-instruct-lc-rtn-w4a16",
        "qwen2.5-3b-instruct-lc-smooth-rtn-w8a8": "qwen2.5-3b-instruct-lc-rtn-w8a8",
        "qwen2.5-7b-instruct-lc-smooth-rtn-w4a16": "qwen2.5-7b-instruct-lc-rtn-w4a16",
        "qwen2.5-7b-instruct-lc-smooth-rtn-w8a8": "qwen2.5-7b-instruct-lc-rtn-w8a8",
        "qwen2.5-14b-instruct-lc-smooth-rtn-w4a16": "qwen2.5-14b-instruct-lc-rtn-w4a16",
        "qwen2.5-14b-instruct-lc-smooth-rtn-w8a8": "qwen2.5-14b-instruct-lc-rtn-w8a8",
        "qwen2.5-32b-instruct-lc-smooth-rtn-w4a16": "qwen2.5-32b-instruct-lc-rtn-w4a16",
        "qwen2.5-32b-instruct-lc-smooth-rtn-w8a8": "qwen2.5-32b-instruct-lc-rtn-w8a8",
        "qwen2.5-72b-instruct-lc-smooth-rtn-w4a16": "qwen2.5-72b-instruct-lc-rtn-w4a16",
        "qwen2.5-72b-instruct-lc-smooth-rtn-w8a8": "qwen2.5-72b-instruct-lc-rtn-w8a8"
    }

    ret = load_pairwise_differences(smooth_to_base)

    # Perform hypothesis tests
    p_value_dict = {}

    # 1. Hypothesis test across models




def ceb_check_aqlm():
    base_to_modified = {
        "llama3.2-1b": "hf-llama3.2-1b-aqlm-pv-2bit-2x8",
        "llama3.2-1b-instruct": "hf-llama3.2-1b-instruct-aqlm-pv-2bit-2x8",
        "llama3.2-3b": "hf-llama3.2-3b-aqlm-pv-2bit-2x8",
        "llama3.2-3b-instruct": "hf-llama3.2-3b-instruct-aqlm-pv-2bit-2x8",
        "llama3.1-8b-instruct": [
            "hf-llama3.1-8b-instruct-aqlm-pv-2bit-2x8",
            "hf-llama3.1-8b-instruct-aqlm-pv-1bit-1x16",
        ],
        "llama3.1-70b-instruct": "hf-llama3.1-70b-instruct-aqlm-pv-2bit-1x16",
    }

    # For each base model & SmoothQuant model, compute difference in score column
    accum_diff = []
    for base_model, modified_models in base_to_modified.items():
        if isinstance(modified_models, str):
            modified_models = [modified_models]
        df_base = pd.DataFrame(load_evaluated_generations(base_model))
        for modified_model in modified_models:
            df_modified = pd.DataFrame(load_evaluated_generations(modified_model))
            # Set index
            df_base = df_base.set_index(["dataset", "social_axis", "prompt"])
            df_modified = df_modified.set_index(["dataset", "social_axis", "prompt"])
            # Compute difference
            df_diff = df_modified["score"] - df_base["score"]
            df_diff = df_diff.reset_index()
            df_diff = df_diff.rename(columns={"score": "score_diff"})
            # Store modified model
            df_diff["model"] = modified_model
            # Drop missing scores
            df_diff = df_diff.dropna(subset=["score_diff"])
            # Accumulate differences
            accum_diff.append(df_diff)

    # Concatenate differences
    df_valid = pd.concat(accum_diff)

    # Perform hypothesis tests
    p_value_dict = {}

    # 1. Hypothesis test across models
    p_value_dict["overall"] = df_valid.groupby(["dataset"])["score_diff"].apply(
        lambda x: metric_utils.bootstrap_hypothesis_test_differences(x)[1]
    )
    # TODO: Need to know if open-ended score is less biased (lower/negative score)

    # 2. Hypothesis test on individual models
    p_value_dict["per_model"] = df_valid.groupby(["model", "dataset"])["score_diff"].apply(
        lambda x: metric_utils.bootstrap_hypothesis_test_differences(x)[1]
    )

    # TODO: Filter for Conversation/Continuation
    curr_results = p_value_dict["overall"].reset_index()

    # TODO: Perform analysis to see which factors play a role into when its significant


def ceb_check_impact_of_chat_template(save_dir="temp"):
    # Mapping of instruct model without vs with chat template
    chat_to_base = {
        "llama3.2-1b-instruct-chat": "llama3.2-1b-instruct",
        "llama3.2-3b-instruct-chat": "llama3.2-3b-instruct",
        "llama3.1-8b-instruct-chat": "llama3.1-8b-instruct",
        "llama3.1-70b-instruct-chat": "llama3.1-70b-instruct",
        "mistral-v0.3-7b-instruct-chat": "mistral-v0.3-7b-instruct",
        "ministral-8b-instruct-chat": "ministral-8b-instruct",
        "mistral-small-22b-instruct-chat": "mistral-small-22b-instruct",
        "qwen2-7b-instruct-chat": "qwen2-7b-instruct"
    }

    ret = load_pairwise_differences(chat_to_base)
    df_valid = ret["accum_valid"]
    df_invalid = ret["accum_invalid"]

    # NOTE: Null transitions
    # df_invalid.to_csv(f"{save_dir}/chat_template-accum_invalids.csv")

    # Perform hypothesis tests
    p_value_dict = {}

    # 1. At the dataset level, does Chat Template impact performance at all?
    p_value_dict["overall"] = metric_utils.grouped_hypothesis_test(
        df_valid,
        group_cols=["dataset"], score_col="score_diff",
        side="!=",
    )

    # 2. Hypothesis test on individual models
    p_value_dict["per_model"] = metric_utils.grouped_hypothesis_test(
        df_valid,
        group_cols=["model", "dataset"], score_col="score_diff",
        side="!=",
    )

    # Find models that are significant
    model_results = p_value_dict["per_model"].reset_index().rename(columns={"score_diff": "p_value"})
    model_results["p_value"] = model_results.apply(lambda row: row["p_value"][0], axis=1)
    # Get number of models that are significant
    sig_model_results = model_results[model_results["p_value"] <= 0.05]
    sig_models = sig_model_results.groupby("dataset")["model"].unique().map(lambda x: ", ".join(list(x)))
    sig_models.to_csv(f"{save_dir}/chat_template-sig_models.csv")

    # Get counts
    df_counts = df_valid.groupby(["dataset"])["score_diff"].value_counts().reset_index()
    df_transitions = df_counts.pivot(index='dataset', columns='score_diff', values='count')
    df_transitions = (100 * df_transitions.T / df_transitions.T.sum()).T.round(1)
    df_transitions.to_csv(f"{save_dir}/chat_template-transitions.csv")


################################################################################
#                               Helper Functions                               #
################################################################################
def bootstrap_ci(data_args, data_kwargs=None, alpha=0.05, func=np.mean, n_bootstrap=10000, seed=SEED):
    # 1. Bootstrap metric values using percentiles
    data_kwargs = data_kwargs or {}
    bootstrap = IIDBootstrap(*data_args, **data_kwargs, seed=seed)
    bs_metric = bootstrap.apply(func, n_bootstrap)

    # Compute confidence interval
    ci = np.quantile(bs_metric, [alpha/2, 1-alpha/2])
    ci = [round(bound, 2) for bound in ci]
    return ci



def bootstrap_diff_ci(
        data_args_A, data_args_B,
        data_kwargs_A=None, data_kwargs_B=None,
        alpha=0.05, func=np.mean, n_bootstrap=10000, seed=SEED,
    ):
    data_kwargs_A = data_kwargs_A or {}
    data_kwargs_B = data_kwargs_B or {}

    # 1.1 Bootstrap metric values using percentiles
    bootstrap_A = IIDBootstrap(*data_args_A, **data_kwargs_A, seed=seed)
    bootstrap_B = IIDBootstrap(*data_args_B, **data_kwargs_B, seed=seed)
    # TODO: Check if following needs to extract first element
    bs_metric_A = bootstrap_A.apply(func, n_bootstrap)
    bs_metric_B = bootstrap_B.apply(func, n_bootstrap)

    # 2. Compute difference between bootstrapped metrics between two groups
    bs_differences = bs_metric_A - bs_metric_B

    # 3. Calculate confidence intervals
    ci = np.quantile(bs_differences, [alpha/2, 1-alpha/2])
    ci = [round(bound, 2) for bound in ci]
    return ci