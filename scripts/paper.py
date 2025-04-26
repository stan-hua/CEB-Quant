"""
test_hypotheses.py

Description: Used to perform analysis used in the paper
"""

# Standard libraries
import os
from datetime import datetime

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from arch.bootstrap import IIDBootstrap
from fire import Fire
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

# Custom libraries
import config
from src.utils import json_utils, viz_utils, metric_utils
from benchmark import *


################################################################################
#                                  Constants                                   #
################################################################################
SEED = 42

# Judge LLM Parameters
EVALUATOR_CHOICE = os.environ.get("EVALUATOR_CHOICE", "atla")
JUDGE_PROMPT_VER = int(os.environ.get("JUDGE_PROMPT_VER", "4"))
SYSTEM_PROMPT_TYPE = os.environ.get("SYSTEM_PROMPT_TYPE", "no_sys_prompt")

# Default save directory
SAVE_DIR = f"temp_{EVALUATOR_CHOICE}"


################################################################################
#                                  Functions                                   #
################################################################################
# RQ0. What is the impact of instruction-finetuning?
def ceb_check_impact_of_instruction(save_dir=SAVE_DIR):
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

    df_accum, df_valid, df_invalid = load_pairwise_differences_extra(
        instruct_to_base, evaluator_choice=EVALUATOR_CHOICE)

    results_accum = {}

    print(f"""
    a total of N={len(df_accum)} paired responses. Of which, only {len(df_valid)} contain valid responses in both
    """.strip())

    # 1. What happens if you only consider the average
    # The difference is small!
    results_accum["Average Fairness (Across Valid/Invalid Samples)"] = {
        "Base": df_accum["score_base"].mean(),
        "Instruct": df_accum["score_modified"].mean(),
        "Difference (Agg)": df_accum["score_modified"].mean() - df_accum["score_base"].mean(),
        # "Difference (Pairwise)": df_accum["score_diff"].mean(),
    }
    ret = results_accum["Average Fairness (Across Valid/Invalid Samples)"]
    initial_gap = ret["Difference (Agg)"]
    print(f"""
        Naively aggregating across all responses for base and instruct models
        yields average fairness scores of {round(ret["Base"], 2)} and {round(ret["Instruct"], 2)},
        respectively, with an inflated improvement of {round(ret["Difference (Agg)"], 2)} points in fairness.
    """.strip())

    # But base models spew garbage most of the time
    results_accum["Response Distributions"] = {
        "Base": df_accum["response_type_base"].value_counts(normalize=True),
        "Instruct": df_accum["response_type_modified"].value_counts(normalize=True),
    }
    get_perc_invalid = lambda x: round(100*x[x.index.str.startswith("Invalid")].sum(), 2)
    perc_base = get_perc_invalid(results_accum["Response Distributions"]["Base"])
    perc_instruct = get_perc_invalid(results_accum["Response Distributions"]["Instruct"])
    print(f"""
        our results show that base models fail to continue text or conversations
        in more than {perc_base}% of responses, often continuing the instruction,
        repeating the biased text, or producing nonsensical output. Meanwhile, instruct models
        fail in only about {perc_instruct}% of responses.
    """.strip())

    # But the evaluation models would consider their output largely biased
    results_accum["Bias Distributions"] = {
        "Base": df_accum["score_base"].value_counts(normalize=True),
        "Instruct": df_accum["score_modified"].value_counts(normalize=True),
    }

    # When filtering on invalid responses, the gap increases is even larger
    results_accum["Average Fairness (Across Invalid Samples)"] = {
        "Base": df_invalid["score_base"].mean(),
        "Instruct": df_invalid["score_modified"].mean(),
        "Difference (Agg)": df_invalid["score_modified"].mean() - df_invalid["score_base"].mean(),
        "Difference (Pairwise)": df_invalid["score_diff"].mean(),
    }
    gap_in_invalid = results_accum["Average Fairness (Across Invalid Samples)"]["Difference (Agg)"]

    print(f"""
    When considering response pairs where at least one response is invalid,
    the gap in fairness increases from {round(initial_gap, 2)} to {round(gap_in_invalid, 2)} points.
    """.strip())

    # When filtering on valid responses, the average bias scores drop and the gap decreases
    results_accum["Average Fairness (Across Valid Samples)"] = {
        "Base": df_valid["score_base"].mean(),
        "Instruct": df_valid["score_modified"].mean(),
        "Difference (Agg)": df_valid["score_modified"].mean() - df_valid["score_base"].mean(),
        "Difference (Pairwise)": df_valid["score_diff"].mean(),
    }
    ret = results_accum["Average Fairness (Across Valid Samples)"]
    ret = {k: round(v, 2) for k, v in ret.items()}

    print(f"""
        On the other hand, filtering for pairs of valid responses results in
        fairness scores of {ret["Base"]} and {ret["Instruct"]} for base and instruct models respectively.
        Thus, instruction fine-tuning improves fairness by {ret["Difference (Agg)"]} points
    """)

    # Does instruction FT improve performance?
    results_accum["Does instruction FT improve fairness?"] = metric_utils.grouped_hypothesis_test(
        df_valid, group_cols=None, score_col="score_diff",
        side=">",
    )
    print("p-value =", results_accum["Does instruction FT improve fairness?"])

    # TODO: Demonstrate that confidence intervals are smaller for pairwise differences
    # Compute CI for difference in average metrics
    # bootstrap_ci([df_accum["score_diff"].to_numpy()])
    # Compute CI for pairwise differences
    # bootstrap_diff_ci(
    #     [df_accum["score_modified"].to_numpy()],
    #     [df_accum["score_base"].to_numpy()],
    # )


# RQ1. What is the impact of quantization?
def ceb_check_impact_of_quantization(save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    # Get all evaluated models
    all_models = os.listdir(os.path.join(config.DIR_EVALUATIONS, EVALUATOR_CHOICE, str(JUDGE_PROMPT_VER), SYSTEM_PROMPT_TYPE))
    # Get the base model for every model
    base_models = [extract_model_metadata_from_name(m)["base_model"] for m in all_models]

    # Filter for model pairs that exist
    quantized_to_base = {
        q_model: b_model
        for q_model, b_model in dict(zip(all_models, base_models)).items()
        if b_model != q_model
    }

    # Load pairwise differences
    df_accum, df_valid, df_invalid = load_pairwise_differences_extra(quantized_to_base, evaluator_choice=EVALUATOR_CHOICE)

    # Add columns for plotting
    add_columns_for_model_config(df_valid)
    add_columns_for_model_config(df_accum)

    # Print average difference with 95% CI
    results_accum = {}

    # Does model behavior change overall?
    # 1. The percentage of invalid responses increased
    results_accum["Response Validity Distributions"] = {
        "Full-Precision": df_accum["is_valid_base"].value_counts(normalize=True),
        "Quantized": df_accum["is_valid_modified"].value_counts(normalize=True),
    }
    # 2. While the percentage of alignment responses stayed about the same
    results_accum["Response Alignment Distributions (Valid Samples)"] = {
        "Full-Precision": df_accum["is_alignment_base"].value_counts(normalize=True),
        "Quantized": df_accum["is_alignment_modified"].value_counts(normalize=True),
    }
    # 3. Compute average fairness difference (on valid samples)
    results_accum["Average Fairness (Across Valid Samples)"] = {
        "avg": df_valid["score_diff"].mean(),
        "ci": bootstrap_ci([df_valid["score_diff"].to_numpy()]),
    }

    ret_valid = results_accum["Response Validity Distributions"]
    valid_full = round(100 * ret_valid["Full-Precision"][False])
    valid_quantized = round(100 * ret_valid["Quantized"][False])
    ret_fair = results_accum["Average Fairness (Across Valid Samples)"]
    diff_avg_decrease = -round(ret_fair["avg"], 2)
    diff_ci = [round(i, 2) for i in ret_fair["ci"]]
    print(f"""
    Quantization leads to an increase in invalid responses from {valid_full}\% to {valid_quantized}\%.
    Between valid response pairs ({len(df_valid)} pairs), fairness decreases by {diff_avg_decrease} points
    (95\% CI: {diff_ci}).
    """.strip())

    # The distribution of bias stays the same
    # NOTE: This is seen in Figure 2
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

    ############################################################################
    #                               Table 1                                    #
    ############################################################################
    # Create helper functions to view mean/std/size easily
    get_stats_row = lambda df: pd.DataFrame([{"mean": df.mean().round(2), "std": df.std().round(2), "size": len(df)}])
    get_stats_df = lambda df, col: df.groupby(col)["score_diff"].apply(get_stats_row).reset_index().drop(columns=["level_1"]).set_index(col)
    normalize_by_group = lambda df, col: df.groupby(col).apply(lambda df_: df_["score_diff"].value_counts(normalize=True).to_frame()).reset_index()

    # Parse dataset name into bias type and task type
    parse_dataset_metadata(df_valid)
    parse_dataset_metadata(df_accum)

    # Rename columns for plotting
    rename_cols = {
        "bias_type": "Bias Type",
        "task_type": "Task Type",
        "social_axis": "Social Axis",
        "descriptor": "Social Group",
    }
    for orig_col, new_col in rename_cols.items():
        df_valid[new_col] = df_valid[orig_col]
        df_accum[new_col] = df_accum[orig_col]

    get_stats_df(df_valid, "Bias Type")
    get_stats_df(df_valid, "Task Type")
    get_stats_df(df_valid, "Social Axis")
    get_stats_df(df_valid, "Social Group")

    # Accumulate changes
    cols = [
        "Task Type", "Bias Type", "Social Axis", "Social Group",
        "Model Family", "Param. Size", "Quantizer", "Bit Config.",
    ]
    accum_change = []
    for col in cols:
        curr_data = get_change_in_behavior(df_valid, df_accum, col)
        curr_data["GroupBy"] = col
        curr_data.rename(columns={col: "Value"}, inplace=True)
        accum_change.append(curr_data)

    # Take only top and bottom 3 in social group
    idx = cols.index("Social Group")
    social_group_copy = accum_change[idx].copy()
    curr_social_group = accum_change[idx].sort_values("proportion")

    # Filter for fairness flips
    curr_social_group_fair = curr_social_group[curr_social_group["Behavior"] == "Fairness"]
    avg_pairwise_diffs = df_valid.groupby("Social Group")["score_diff"].mean()
    diff_min, diff_max = round(avg_pairwise_diffs.min(), 2), round(avg_pairwise_diffs.max(), 2)
    print(f"Across {len(avg_pairwise_diffs)} social groups, the average pairwise difference varies from {diff_min} to {diff_max}")
    print("10 Least-Impacted Social Groups (Fairness Flips):", curr_social_group_fair["Value"].head(10).tolist())
    print("10 Most-Impacted Social Groups (Fairness Flips):", curr_social_group_fair["Value"].tail(10).tolist())

    # Find top and bottom 2 in social group based on Fairness
    filter_social_groups = curr_social_group_fair["Value"].head(2).tolist() + curr_social_group_fair["Value"].tail(2).tolist()
    accum_change[idx] = curr_social_group[curr_social_group["Value"].isin(filter_social_groups)]

    # Concatenate
    df_accum_behavior = pd.concat(accum_change, ignore_index=True)
    df_accum_behavior = df_accum_behavior.dropna()

    # Round proportion column and add confidence interval
    df_accum_behavior["Percentage"] = df_accum_behavior.apply(
        lambda row: str(round(100 * row["proportion"], 1)) + " " + "(" + str(round(100 * row["CI"][0], 1)) + ", " + str(round(100 * row["CI"][1], 1)) + ")",
        axis=1
    )
    df_behavior_flipping = df_accum_behavior[["GroupBy", "Value", "Percentage", "Behavior"]].pivot(
        index=["GroupBy", "Value"],
        columns="Behavior",
        values="Percentage"
    ).reset_index()
    df_behavior_flipping["GroupBy"] = pd.Categorical(df_behavior_flipping["GroupBy"], categories=cols, ordered=True)
    df_behavior_flipping.sort_values("GroupBy").to_csv(os.path.join(save_dir, "fairness_flipping.csv"), index=False)


    ############################################################################
    #                               Figure 2                                   #
    ############################################################################
    # Create plots for Impact of Quantization
    viz_utils.set_theme(tick_scale=2.3, figsize=(35, 10))
    fig, axs = plt.subplots(1, 3)

    # 1. Distribution of fairness scores
    viz_utils.catplot(
        df_accum_scores,
        plot_type="count", stat="percent",
        x="score", hue="Is Quantized",
        xlabel="Fairness Score",
        ylabel="Percentage of Responses",
        title="Distribution of Fairness Scores",
        legend=True,
        palette=["#C78E72", "#F6D07F"],
        ax=axs[0],
    )
    # 2. Pairwise difference in fairness
    df_valid["score_diff"] = df_valid["score_diff"].astype(int)
    viz_utils.catplot(
        df_valid,
        plot_type="count", stat="percent",
        palette="icefire",
        x="score_diff",
        xlabel="Pairwise Change in Fairness",
        ylabel="Percentage",
        title="Quant-Induced (%) Change in Fairness",
        ax=axs[1],
    )

    # 3. Pairwise transition from valid to invalid answers
    viz_utils.catplot(
        df_accum,
        plot_type="heatmap", stat="proportion",
        y="is_valid_base", x="is_valid_modified",
        order=[True, False],
        xlabel="Quantized Model",
        ylabel="Unquantized Model",
        title="(%) Change in Response Validity",
        ax=axs[2],
    )

    # Save
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"fig2-{EVALUATOR_CHOICE.lower()}.svg"), bbox_inches="tight")
    plt.close()

    ############################################################################
    #                               Figure 3                                   #
    ############################################################################
    viz_utils.set_theme(tick_scale=2.3, figsize=(35, 10), rcparams={"legend.fontsize": "small"})
    fig, axs = plt.subplots(1, 3)

    # 1. Social Groups (Most Variable, Least Variable)
    col = "Social Group"
    title = col
    # Get the group with the most/least score difference
    group_stds = df_valid.groupby("Social Group")["score_diff"].std().sort_values()
    filter_groups = [group_stds.index[0], group_stds.index[-1]]
    hue_order = filter_groups[::-1]
    ax = axs[0]
    df_valid_curr = df_valid[df_valid["Social Group"].isin(filter_groups)]
    # plot_change_in_fairness(df_valid_curr, col=col, hue_order=hue_order, title=title, ax=ax)
    plot_changes_in_behavior(df_valid_curr, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # 2. Bit Configuration
    col = "Bit Config."
    title = "Quant. Bit Configuration"
    hue_order = ['W8A16', 'W8A8', 'W4A16', 'W2A16', 'W1A16']
    ax = axs[1]
    curr_idx = 0
    palette = sns.color_palette("colorblind", 20)[curr_idx:curr_idx+len(hue_order)]
    curr_idx += len(hue_order)
    # plot_change_in_fairness(df_valid, col=col, hue_order=hue_order, title=title, ax=ax)
    plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # 3. Model Size
    title = "Parameter Size"
    col = "Param. Size"
    hue_order = sorted(df_valid[col].unique().tolist())
    ax = axs[2]
    curr_idx = 0
    palette = sns.color_palette("colorblind", 20)[curr_idx:curr_idx+len(hue_order)]
    curr_idx += len(hue_order)
    # plot_change_in_fairness(df_valid, col=col, hue_order=hue_order, title=title, ax=ax)
    plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # Save
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    fig.savefig(os.path.join(save_dir, f"fig3-{EVALUATOR_CHOICE.lower()}.svg"), bbox_inches="tight")
    plt.close()


    ############################################################################
    #                                 Text                                     #
    ############################################################################
    df_valid.groupby(["Quantizer", "Bit Config."])["score_diff"].mean()
    print(f"""
    Among the quantization methods tested, ...
    \t{df_valid.groupby(["Quantizer", "Bit Config."])["score_diff"].mean()}
    """.strip())

    ############################################################################
    #                            Unused Figures                                #
    ############################################################################
    # viz_utils.set_theme(tick_scale=2.3, figsize=(35, 10), rcparams={"legend.fontsize": "small"})
    # fig, axs = plt.subplots(1, 4)

    # TODO: 1. Bias Type
    # col = "Bias Type"
    # title = col
    # hue_order = ["Stereotype", "Toxicity"]
    # ax = axs[0]
    # palette = sns.color_palette("colorblind", 10)[:2]
    # # plot_change_in_fairness(df_valid, col=col, hue_order=hue_order, title=title, ax=ax)
    # plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # TODO: 2. Task Type
    # col = "Task Type"
    # title = col
    # hue_order = ["Continuation", "Conversation"]
    # ax = axs[1]
    # palette = sns.color_palette("colorblind", 10)[2:4]
    # # plot_change_in_fairness(df_valid, col=col, hue_order=hue_order, title=title, ax=ax)
    # plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # 3. Social Axis
    # col = "Social Axis"
    # title = col
    # hue_order = ["Age", "Gender", "Race", "Religion"]
    # ax = axs[2]
    # palette = sns.color_palette("colorblind", 10)[4:8]
    # # plot_change_in_fairness(df_valid, col=col, hue_order=hue_order, title=title, ax=ax)
    # plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # # 4. Social Groups (Most Variable, Least Variable)
    # col = "Social Group"
    # title = col
    # group_stds = df_valid.groupby("Social Group")["score_diff"].std().sort_values()
    # filter_groups = [group_stds.index[0], group_stds.index[-1]]
    # hue_order = sorted(filter_groups)
    # ax = axs[3]
    # palette = sns.color_palette("colorblind", 10)[8:10]
    # df_valid_curr = df_valid[df_valid["Social Group"].isin(filter_groups)]
    # # plot_change_in_fairness(df_valid_curr, col=col, hue_order=hue_order, title=title, ax=ax)
    # plot_changes_in_behavior(df_valid_curr, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # # Save
    # plt.subplots_adjust(hspace=0.25, wspace=0.25)
    # fig.savefig(os.path.join(save_dir, f"figX-{EVALUATOR_CHOICE.lower()}.svg"), bbox_inches="tight")
    # plt.close()

    ############################################################################
    #                            Unused Figure                                 #
    ############################################################################
    # viz_utils.set_theme(tick_scale=2.3, figsize=(35, 10), rcparams={"legend.fontsize": "small"})
    # fig, axs = plt.subplots(1, 3)

    # get_stats_df(df_valid, "Quantizer")
    # get_stats_df(df_valid, "Bit Config.")
    # get_stats_df(df_valid, "Model Family")
    # get_stats_df(df_valid, "param_size")

    # # 1. Quantization Strategy
    # col = "Quantizer"
    # title = "Quantization Method"
    # hue_order = ["RTN", "GPTQ", "AWQ", "SmoothQuant", "AQLM"]
    # ax = axs[0, 0]
    # plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # # 2. Bit Configuration
    # col = "Bit Config."
    # title = "Quant. Bit Configuration"
    # hue_order = ['W8A16', 'W8A8', 'W4A16', 'W2A16', 'W1A16']
    # ax = axs[0, 1]
    # # plot_change_in_fairness(df_valid, col=col, hue_order=hue_order, title=title, ax=ax)
    # plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # # 3. Model Family
    # col = "Model Family"
    # title = col
    # hue_order = sorted(df_valid[col].unique().tolist())
    # ax = axs[1, 0]
    # # plot_change_in_fairness(df_valid, col=col, hue_order=hue_order, title=title, ax=ax)
    # plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # # 4. Model Size
    # title = "Parameter Size"
    # col = "Param. Size"
    # hue_order = sorted(df_valid[col].unique().tolist())
    # ax = axs[1, 1]
    # # plot_change_in_fairness(df_valid, col=col, hue_order=hue_order, title=title, ax=ax)
    # plot_changes_in_behavior(df_valid, df_accum, col=col, hue_order=hue_order, title=title, ax=ax)

    # # Save
    # plt.subplots_adjust(hspace=0.25, wspace=0.25)
    # fig.savefig(os.path.join(save_dir, f"figXXX-{EVALUATOR_CHOICE.lower()}.svg"), bbox_inches="tight")
    # plt.close()


# RQ2. How does this relate to general model capabilities?
def ceb_compute_fairness_vs_lm_eval_correlation(save_dir=SAVE_DIR):
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
                time_str = os.path.basename(json_path).split("_")[1].split(".")[0]
                curr_time = datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S")
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
        try:
            df_curr = pd.DataFrame(load_evaluated_generations(model_name, dataset_names="all_ceb_open_ended", evaluator_choice=EVALUATOR_CHOICE))
        except:
            print(f"Unable to load metrics for model {model_name}")
            continue
        # Categorize responses
        assign_reason_type_single(df_curr)
        parse_dataset_metadata(df_curr)
        # Filter on valid responses
        df_curr = df_curr[df_curr["is_valid"]]
        # Skip, if model has not responsed to at least 50 prompts in each dataset
        valid_counts = df_curr.groupby(["bias_type", "task_type"]).size()
        if not (valid_counts >= 50).all():
            print(f"Not enough valid responses from model {model_name}")
            continue
        # Create average scores
        curr_ceb_metrics = {}
        curr_ceb_metrics["avg_score"] = df_curr["score"].mean()
        curr_ceb_metrics["avg_stereotype_score"] = df_curr[df_curr["bias_type"] == "Stereotype"]["score"].mean()
        curr_ceb_metrics["avg_toxicity_score"] = df_curr[df_curr["bias_type"] == "Toxicity"]["score"].mean()
        curr_ceb_metrics["avg_continuation_score"] = df_curr[df_curr["task_type"] == "Continuation"]["score"].mean()
        curr_ceb_metrics["avg_conversation_score"] = df_curr[df_curr["task_type"] == "Conversation"]["score"].mean()
        curr_ceb_metrics["model"] = model_name
        accum_ceb_metrics.append(curr_ceb_metrics)

    # Combine dataframes
    df_lm_eval = pd.DataFrame(accum_lm_eval_metrics).set_index("model")
    df_ceb = pd.DataFrame(accum_ceb_metrics).set_index("model")
    df_bench = pd.merge(df_ceb, df_lm_eval, on="model", how="inner").dropna().reset_index()

    # Add model details from name
    model_metadata = pd.DataFrame(df_bench["model"].map(extract_model_metadata_from_name).tolist())
    df_bench = pd.concat([df_bench, model_metadata], axis=1)
    df_bench["Is Quantized"] = df_bench["base_model"] != df_bench["model"]

    # Filter for instruct models
    df_bench = df_bench[df_bench["instruct_tuned"]]
    # Remove AQLM since it skews the distribution
    df_bench = df_bench[~df_bench["model"].str.lower().str.contains("AQLM")]

    # LM-Eval columns
    lm_eval_cols = [
        'hellaswag / acc',
        'piqa / acc',
        'mmlu_pro / exact_match',
        'truthfulqa_mc1 / acc',
        'lambada_openai / acc',
        'lambada_openai / perplexity',
    ]
    all_eval_cols = ["avg_stereotype_score", "avg_toxicity_score", "avg_continuation_score", "avg_conversation_score", "avg_score", "avg_acc"] + lm_eval_cols

    # 1. Only full-precision instruct models
    # Filter for base model vs. not base model
    is_full_precision = (df_bench["w_bits"] == 16) & (df_bench["a_bits"] == 16)
    df_full_precision = df_bench.loc[is_full_precision].copy()
    # Normalize score columns
    # scaler = StandardScaler()
    # df_full_precision[all_eval_cols] = scaler.fit_transform(df_full_precision[all_eval_cols])

    # 1. Create scatterplot with unquantized models
    # viz_utils.numplot(
    #     df_full_precision,
    #     x="avg_score", y="avg_acc", hue="Model Size (GB)",
    #     size="Model Size (GB)", sizes=(300, 3000),
    #     plot_type="scatter",
    #     xlabel="Normalized Avg. Fairness Score",
    #     ylabel="Normalized Avg. Benchmark Accuracy",
    #     tick_params={"bottom": False, "left": False, "labelbottom": False, "labelleft": False},
    #     title="Unquantized Instruct Models",
    #     palette="flare",
    #     ax=axs[0, 0],
    # )
    # Compute correlation between CEB Fairness and individual benchmarks
    fairness_corrs, pvals = corr_with_pvalues(df_full_precision[all_eval_cols])
    fairness_corrs = fairness_corrs.iloc[0:5, 5:].round(2)
    pvals = pvals.iloc[0:5, 5:].round(3)
    curr_save_dir = os.path.join(save_dir, "fairness_vs_lm_eval-fp16_instruct")
    os.makedirs(curr_save_dir, exist_ok=True)
    fairness_corrs.to_csv(os.path.join(curr_save_dir, "corrs.csv"))
    pvals.to_csv(os.path.join(curr_save_dir, "corrs_pvals.csv"))


    ############################################################################
    #                               Figure 4                                   #
    ############################################################################
    viz_utils.set_theme(tick_scale=2.3, figsize=(35, 10), rcparams={"legend.fontsize": "small"}, style="whitegrid")
    fig, axs = plt.subplots(1, 3)

    # 2. Plot base and quantized models
    df_together = df_bench.copy()
    # Normalize score columns
    scaler = StandardScaler()
    df_together[all_eval_cols] = scaler.fit_transform(df_together[all_eval_cols])
    
    # 1. Create scatterplot with both unquantized and quantized models
    viz_utils.numplot(
        df_together,
        x="avg_score", y="avg_acc", hue="Is Quantized",
        size="Model Size (GB)", sizes=(300, 3000),
        plot_type="scatter",
        xlabel="Normalized Avg. Fairness Score",
        ylabel="Normalized Avg. Benchmark Accuracy",
        tick_params={"bottom": False, "left": False, "labelbottom": False, "labelleft": False},
        ax=axs[0],
    )
    # Compute correlation between CEB Fairness and individual benchmarks
    fairness_corrs, pvals = corr_with_pvalues(df_together[all_eval_cols])
    fairness_corrs = fairness_corrs.iloc[0:5, 5:].round(2)
    pvals = pvals.iloc[0:5, 5:].round(3)
    curr_save_dir = os.path.join(save_dir, "fairness_vs_lm_eval-all_models")
    os.makedirs(curr_save_dir, exist_ok=True)
    fairness_corrs.to_csv(os.path.join(curr_save_dir, "corrs.csv"))
    pvals.to_csv(os.path.join(curr_save_dir, "corrs_pvals.csv"))

    # 2. Create plot of fairness against effective size
    viz_utils.numplot(
        df_bench,
        x="Model Size (GB)", y="avg_score",
        hue="Is Quantized",
        size="Model Size (GB)", sizes=(300, 3000),
        plot_type="scatter",
        xlabel="Model Size (GB)",
        ylabel="Avg. Fairness Score",
        legend=True,
        horizontal_legend=True,
        ax=axs[1],
    )

    # 3. Create plot of average pairwise difference against difference in LM-evals
    # Get pairwise differences
    quant_to_unquant = parse_quantized_to_unquantized_pairs(df_bench["model"].unique().tolist())

    # Compute difference in aggregate scores x difference in benchmarks
    df_bench_indexed = df_bench.set_index("model")
    accum_agg_differences = []
    for quantized_model, unquantized_model in quant_to_unquant.items():
        row_after = df_bench_indexed.loc[quantized_model, all_eval_cols]
        row_before = df_bench_indexed.loc[unquantized_model, all_eval_cols]
        diff_row = row_after - row_before
        diff_row["quantized_model"] = quantized_model
        accum_agg_differences.append(diff_row)

    # Get model metadata
    df_agg_differences = pd.DataFrame(accum_agg_differences)
    model_metadata = pd.DataFrame(df_agg_differences["quantized_model"].map(extract_model_metadata_from_name).tolist())
    df_agg_differences = pd.concat([df_agg_differences, model_metadata], axis=1)
    add_columns_for_model_config(df_agg_differences)

    # 3. Create plot
    viz_utils.numplot(
        df_agg_differences,
        x="avg_score", y="avg_acc", hue="Quantizer",
        size="Model Size (GB)", sizes=(300, 3000),
        plot_type="scatter",
        xlabel="Difference in Avg. Fairness Score",
        ylabel="Difference in Avg. Benchmark Accuracy",
        legend=True,
        horizontal_legend=True,
        y_lim=[-0.1, 0.05],
        ax=axs[2],
    )

    # Save
    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    fig.savefig(os.path.join(save_dir, f"fig4-{EVALUATOR_CHOICE.lower()}.svg"), bbox_inches="tight")
    plt.close()


def ceb_sample_for_human_annotation():
    df_atla = pd.read_csv("atla_all.csv.gz")
    df_prometheus = pd.read_csv("prometheus_all.csv.gz")


def ceb_compute_judge_human_agreement():
    # Load Prometheus/Atla samples and the human annotations
    df_all = load_atla_and_prometheus_and_human_judges(sample=True)

    # Parse to Y/N is valid scores
    df_all["is_valid"] = identify_valid_responses(df_all, "_base", use_regex=False)
    df_all["is_valid_a"] = identify_valid_responses(df_all, "_base_a", use_regex=False)
    df_all["is_valid_p"] = identify_valid_responses(df_all, "_base_p", use_regex=False)

    # Parse to Y/N is fair scores
    is_fair = lambda x: int(x == 100) if not pd.isnull(x) else None
    df_all["is_fair"] = df_all["score_base"].map(is_fair)
    df_all["is_fair_a"] = df_all["score_base_a"].map(is_fair)
    df_all["is_fair_p"] = df_all["score_base_p"].map(is_fair)

    llm_judge_to_suffix = [("atla", "_a"), ("prometheus", "_p")]
    ############################################################################
    #                     Valid vs. Invalid Responses                          #
    ############################################################################
    accum_rows = []
    for col in ["is_valid", "score_base", "is_fair"]:  # "rta_score_base", 
        df_curr = df_all.copy()
        # If on bias scores, filter for responses model think is valid
        if col in ["score_base", "is_fair"]:
            valid_mask = df_curr[f"is_valid"].fillna(False).astype(bool)
            df_curr = df_curr[valid_mask]
        # Compute agreement between human and judges
        for judge, suffix in llm_judge_to_suffix:
            # Drop rows with incomplete scores
            score_cols = [f"{col}", f"{col}{suffix}"]
            df_temp = df_curr.dropna(subset=score_cols)
            weight_to_kappa = {}
            for weight in [None, "linear", "quadratic"]:
                args = [df_temp[score_cols[i]].astype(int) for i in range(2)]
                kappa = cohen_kappa_score(*args, weights=weight)
                curr_row = {"judge": "human x " + judge, "col": col, "weight": weight, "kappa": kappa}
                accum_rows.append(curr_row)
        # Compute agreement between judges
        score_cols = [f"{col}_a", f"{col}_p"]
        df_temp = df_curr.dropna(subset=score_cols)
        weight_to_kappa = {}
        for weight in [None, "linear", "quadratic"]:
            args = [df_temp[score_cols[i]].astype(int) for i in range(2)]
            kappa = cohen_kappa_score(*args, weights=weight)
            curr_row = {"judge": "atla x prometheus", "col": col, "weight": weight, "kappa": kappa}
            accum_rows.append(curr_row)

    df_agreement = pd.DataFrame(accum_rows)
    df_curr = df_all.dropna(subset=["is_fair"])
    df_agreement.to_csv("human_evaluator-agreement.csv", index=False)
    pd.crosstab(df_curr["is_valid"], df_curr["is_valid_a"]).to_csv("human_atla-validity_cont.csv")
    valid_mask = df_curr["is_valid"].fillna(False).astype(bool)
    pd.crosstab(df_curr.loc[valid_mask, "is_fair"], df_curr.loc[valid_mask, "is_fair_a"]).to_csv("human_atla-is_fair_cont.csv")
    pd.crosstab(df_curr.loc[valid_mask, "score_base"], df_curr.loc[valid_mask, "score_base_a"]).to_csv("human_atla-bias_score_cont.csv")


def supp_get_existing_models(save_dir=SAVE_DIR):
    # Get all evaluated models
    all_models = os.listdir(os.path.join(config.DIR_EVALUATIONS, EVALUATOR_CHOICE, str(JUDGE_PROMPT_VER), SYSTEM_PROMPT_TYPE))
    # Get the base model for every model
    base_models = [extract_model_metadata_from_name(m)["base_model"] for m in all_models]

    # Filter for model pairs that exist
    quantized_to_base = {
        q_model: b_model
        for q_model, b_model in dict(zip(all_models, base_models)).items()
        if b_model != q_model
    }

    # Load pairwise differences
    df_accum, _, _ = load_pairwise_differences_extra(quantized_to_base, evaluator_choice=EVALUATOR_CHOICE)

    # Add quantization method name
    create_name = lambda row: row["q_method"].upper() + f'-W{row["w_bits"]}A{row["a_bits"]}' + ("-SQ" if row["smoothquant"] else "")
    df_models = df_accum[["base_model", "model_family", "w_bits", "a_bits", "q_method", "smoothquant", "instruct_tuned"]].drop_duplicates()
    df_models["Quant_Method"] = df_models.apply(create_name, axis=1)

    # Create model x quantization method dataframe
    model_counts = df_models.groupby(["base_model", "Quant_Method"]).size().unstack(fill_value=0)
    model_counts.to_csv(os.path.join(save_dir, "supplementary-quantization_used.csv"))


################################################################################
#                               Helper Functions                               #
################################################################################
def identify_valid_responses(df_accum, col_suffix="", use_regex=False):
    # Add valid responses
    mask = (df_accum[f"rta_score{col_suffix}"] >= 4).fillna(False)
    # Identify valid responses (that were labeled invalid)
    if use_regex:
        rta_blacklist = "copy|copie|duplicat|refus|reiterat|repet|repeat|restat|re-stat|redundant|incoheren|nonsensical|irrelevant|unrelated|(not engag)|unnecessary|(lack of coheren|relevan|relat)|vague|((does not|fails to)( \S*)? (continue|engage))|lacks( \S*) content|without any expansion/continuation"
        valid_invalid_mask = (~df_accum[f"rta_feedback{col_suffix}"].fillna("").str.contains(rta_blacklist, regex=True))
        valid_invalid_mask = valid_invalid_mask & (~df_accum[f"bias_feedback{col_suffix}"].fillna("").str.contains(rta_blacklist, regex=True))
        valid_invalid_mask = valid_invalid_mask & (df_accum[f"rta_score{col_suffix}"] < 4)
        # valid_invalid_mask = valid_invalid_mask & (df_accum[f"score{col_suffix}"] != 100)
        mask = mask | valid_invalid_mask
    mask[df_accum[f"rta_score{col_suffix}"].isna()] = None
    return mask.astype(float)


def plot_change_in_fairness(df, col, **kwargs):
    viz_utils.catplot(
        normalize_by_group(df, col),
        plot_type="bar",
        x="score_diff", y="proportion", hue=col,
        xlabel="Pairwise Change in Fairness",
        ylabel="Percentage",
        palette="colorblind",
        legend=True,
    )


def plot_changes_in_behavior(df_valid, df_accum, col, **kwargs):
    df_change = get_change_in_behavior(df_valid, df_accum, col)
    # Create bar plot
    viz_utils.catplot(
        df_change,
        plot_type="bar",
        x="Behavior", y="proportion", hue=col,
        order=["Fairness", "Validity"],
        xlabel="Behavior",
        ylabel="(%) Responses Flipped",
        y_lim=[0, 1],
        legend=True,
        **kwargs
    )


def get_change_in_behavior(df_valid, df_accum, col):
    # Compute the percentage of responses that changed
    # 1. Fairness (in valid responses)
    df_change_fair = df_valid.groupby(col)["Fairness Flipped"].mean().reset_index()
    df_change_fair.rename(columns={"Fairness Flipped": "proportion"}, inplace=True)
    df_change_fair["Behavior"] = "Fairness"
    df_change_fair["CI"] = df_valid.groupby(col)["Fairness Flipped"].apply(lambda data: bootstrap_ci([np.array(data).astype(int)])).tolist()
    # 2. Validity (in all responses)
    df_change_valid = df_accum.groupby(col)["Validity Flipped"].mean().reset_index()
    df_change_valid.rename(columns={"Validity Flipped": "proportion"}, inplace=True)
    df_change_valid["Behavior"] = "Validity"
    df_change_valid["CI"] = df_accum.groupby(col)["Validity Flipped"].apply(lambda data: bootstrap_ci([np.array(data).astype(int)])).tolist()
    # 3. Alignment (in all responses)
    # df_change_align = df_accum.groupby(col)["Alignment Flipped"].mean().reset_index()
    # df_change_align.rename(columns={"Alignment Flipped": "proportion"}, inplace=True)
    # df_change_align["Behavior"] = "Alignment"
    # Concatenate and create bar plot
    df_change = pd.concat([df_change_fair, df_change_valid], ignore_index=True)
    return df_change


def bootstrap(data_args, data_kwargs=None, func=np.mean, n_bootstrap=10000, seed=SEED):
    # Simply bootstrap values
    data_kwargs = data_kwargs or {}
    bootstrap = IIDBootstrap(*data_args, **data_kwargs, seed=seed)
    bs_metrics = bootstrap.apply(func, n_bootstrap)
    return bs_metrics


def bootstrap_ci(data_args, data_kwargs=None, alpha=0.05, func=np.mean, n_bootstrap=10000, seed=SEED):
    # 1. Bootstrap metric values using percentiles
    data_kwargs = data_kwargs or {}
    bootstrap = IIDBootstrap(*data_args, **data_kwargs, seed=seed)
    bs_metric = bootstrap.apply(func, n_bootstrap)
    # Compute confidence interval
    ci = np.quantile(bs_metric, [alpha/2, 1-alpha/2])
    ci = [round(bound, 4) for bound in ci]
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


def assign_reason_types(df_accum):
    """
    Using the response type provided, stratify to valid/invalid and alignment
    response or not.

    Parameters
    ----------
    df_accum : pd.DataFrame
        Contains `response_type_base` and `response_type_modified`
    """
    df_accum["is_fair_base"] = df_accum["score_base"] == 100
    df_accum["is_fair_modified"] = df_accum["score_modified"] == 100
    df_accum["is_valid_base"] = df_accum["response_type_base"].map(lambda x: "Valid" in x)
    df_accum["is_valid_modified"] = df_accum["response_type_modified"].map(lambda x: "Valid" in x)
    df_accum["is_alignment_base"] = df_accum["response_type_base"].map(lambda x: "Alignment" in x)
    df_accum["is_alignment_modified"] = df_accum["response_type_modified"].map(lambda x: "Alignment" in x)


def assign_reason_type_single(df_data):
    df_data["is_valid"] = df_data["response_type"].map(lambda x: isinstance(x, str) and "Valid" in x)
    df_data["is_alignment"] = df_data["response_type"].map(lambda x: isinstance(x, str) and "Alignment" in x)


def assign_behavior_flips(df_valid, df_accum):
    # df_valid["Fairness Flipped"] = df_valid["score_diff"] != 0
    df_valid["Fairness Flipped"] = df_valid["is_fair_base"] != df_valid["is_fair_modified"]
    df_accum["Validity Flipped"] = df_accum["is_valid_base"] != df_accum["is_valid_modified"]
    df_accum["Alignment Flipped"] = df_accum["is_alignment_base"] != df_accum["is_alignment_modified"]


def parse_dataset_metadata(df_data):
    bias_mapping = {"S": "Stereotype", "T": "Toxicity"}
    df_data["bias_type"] = df_data["dataset"].str.split("-").str[2].map(bias_mapping.get)
    df_data["task_type"] = df_data["dataset"].str.split("-").str[1]
    df_data["social_axis"] = df_data["social_axis"].map(lambda x: x.capitalize())


def add_columns_for_model_config(df_data):
    df_data["Bit Config."] = df_data.apply(lambda row: f"W{row['w_bits']}A{row['a_bits']}", axis=1)
    df_data["Quantizer"] = df_data.apply(lambda row: f"{row['q_method'].upper()}" if not row["smoothquant"] else "SmoothQuant", axis=1)
    df_data["Model Family"] = df_data["base_model"].str.split("-").str[0].str.capitalize()
    df_data["Param. Size"] = df_data["param_size"].astype(float)
    df_data["Param. Size"] = pd.cut(df_data["Param. Size"], bins=[0, 5, 10, 20, 35, 65, 75])


def parse_quantized_to_unquantized_pairs(models):
    """
    Given a list of models (quantized and unquantized), create a dictionary
    mapping all quantized models to their unquantized counter-parts
    """
    quantized_to_unquantized = {}
    models_set = set(models)
    for model in models:
        model_metadata = extract_model_metadata_from_name(model)
        # If not quantized, skip
        if model_metadata["w_bits"] == 16 and model_metadata["a_bits"] == 16:
            continue
        # Check if base model is in the set
        unquantized_model = model_metadata["base_model"]
        if unquantized_model in models_set:
            quantized_to_unquantized[model] = unquantized_model
    return quantized_to_unquantized


def corr_with_pvalues(df):
    cols = df.columns
    n = len(cols)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                corr, p_value = pearsonr(df[cols[i]], df[cols[j]])
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_value
    corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols)
    p_df = pd.DataFrame(p_matrix, index=cols, columns=cols)
    return corr_df, p_df


def parse(x):
    """
    Attempt to parse string to correct type.

    Parameters
    ----------
    x : str
        String that can be evaluated to a int/float/list
    """
    import ast
    if not isinstance(x, str):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return None


def apply_func_on_list(x, func=min):
    """
    Apply function on non-null list-like items.

    Parameters
    ----------
    x : Any
        Item to apply function on
    func : function
        Function to apply

    Returns
    -------
    Any
        Result of function if x is list-like. None, if x is null, and x otherwise
    """
    if pd.isnull(x):
        return None
    if isinstance(x, (list, tuple)):
        return func(x)
    return x


def create_ids(df, id_cols=None, suffix=None):
    """
    Create model and question-specific IDs

    Parameters
    ----------
    df : pd.DataFrame
        Table of LLM responses
    id_cols : list, optional
        List of columns to include in creating ID, by default
        ["model_base", "dataset", "prompt"]
    suffix : str
        Suffix to append to column names

    Returns
    -------
    pd.Series
        Unique identifiers for each row
    """
    if not id_cols:
        id_cols = ["model_base", "dataset", "prompt"]
    if suffix:
        id_cols = [f"{x}{suffix}" for x in id_cols]
    # Create hashes from stringified ID columns
    df_hashes = df[id_cols].astype(str).map(lambda x: x.strip()).map(hash)
    # Combine hashes row-wise to create ids
    ids = df_hashes.apply(lambda row: "_".join(map(str, row.values.tolist())), axis=1)
    return ids


################################################################################
#                           Load Evaluated Questions                           #
################################################################################
def load_pairwise_differences_extra(modified_to_base, **kwargs):
    """
    Load evaluated generations for baseline and modified model. Compute
    pairwise differences in fairness scores between rows. Then add model details
    and reasons for failure.

    Parameters
    ----------
    modified_to_base : dict
        Mapping from modified model name to baseline model name
    **kwargs : Any
        Keyword arguments for `load_evaluated_generations`

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (i) Dataframe of all responses with pairwise differences in fairness scores
        (ii) Dataframe of transition matrix for invalid responses in base/modified models
    """
    ret = load_pairwise_differences(modified_to_base, **kwargs)
    df_valid = ret["accum_valid"]
    df_invalid = ret["accum_invalid"]
    # Add model details from name
    model_metadata = pd.DataFrame(df_valid["model_modified"].map(extract_model_metadata_from_name).tolist())
    df_valid = pd.concat([df_valid, model_metadata], axis=1)
    model_metadata = pd.DataFrame(df_invalid["model_modified"].map(extract_model_metadata_from_name).tolist())
    df_invalid = pd.concat([df_invalid, model_metadata], axis=1)
    # Concatenate and assign reason
    df_accum = pd.concat([df_valid, df_invalid], ignore_index=True)
    assign_reason_types(df_accum)
    assign_reason_types(df_valid)
    assign_behavior_flips(df_valid, df_accum)
    return df_accum, df_valid, df_invalid


def load_pairwise_differences(modified_to_base, **kwargs):
    """
    Load evaluated generations for baseline and modified model. Compute
    pairwise differences in fairness scores between rows.

    Parameters
    ----------
    modified_to_base : dict
        Mapping from modified model name to baseline model name
    **kwargs : Any
        Keyword arguments for `load_evaluated_generations`

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (i) Dataframe of all responses with pairwise differences in fairness scores
        (ii) Dataframe of transition matrix for invalid responses in base/modified models
    """
    # For each base & quantized model, compute difference in score column
    accum_valid = []
    accum_invalid = []
    for modified_model, base_model in modified_to_base.items():
        keys = ["dataset", "social_axis", "prompt"]
        try:
            shared_kwargs = {"dataset_names": "all_ceb_open_ended", "on_missing_gen": "ignore"}
            shared_kwargs.update(kwargs)
            df_base = pd.DataFrame(load_evaluated_generations(base_model, **shared_kwargs))
            df_modified = pd.DataFrame(load_evaluated_generations(modified_model, **shared_kwargs))

            assert set(keys).issubset(set(df_base.columns.tolist())), f"Base model is missing key columns! Base Columns: {df_base.columns.tolist()}"
            assert set(keys).issubset(set(df_modified.columns.tolist())), f"Modified Model is missing key columns! Modified Columns: {df_modified.columns.tolist()}"
        except:
            LOGGER.error(f"Failed to load evaluated generations for models: ({base_model}, {modified_model})")
            tb = traceback.format_exc()
            LOGGER.error(tb)
            continue

        # Set index
        keys = ["dataset", "social_axis", "descriptor", "prompt"]
        df_base = df_base.set_index(keys)
        df_modified = df_modified.set_index(keys)

        # Join to get the number of null to null transforms
        keep_cols = ["score", "response_type", "rta_score", "res", "bias_feedback", "rta_feedback"]
        df_joined = pd.merge(
            df_base[keep_cols], df_modified[keep_cols],
            how="inner", on=keys,
            suffixes=["_base", "_modified"],
        ).reset_index()

        # Add base and modified model
        df_joined["model_base"] = base_model
        df_joined["model_modified"] = modified_model

        # Compute difference
        df_joined["score_diff"] = df_joined["score_modified"] - df_joined["score_base"]

        # Determine valid vs invalid responses
        response_type_cols = ["response_type_base", "response_type_modified"]
        # 1. Missing Eval Scores
        df_joined[response_type_cols] = df_joined[response_type_cols].fillna("Invalid")
        valid_mask = ~df_joined[["score_base", "score_modified"]].isna().any(axis=1)
        # 2. Invalid Response Type
        valid_mask = valid_mask & df_joined["response_type_base"].map(lambda x: x.startswith("Valid"))
        valid_mask = valid_mask & df_joined["response_type_modified"].map(lambda x: x.startswith("Valid"))
        accum_invalid.append(df_joined[~valid_mask].copy())
        accum_valid.append(df_joined[valid_mask].copy())

    # Get transition between valid to invalid responses
    df_invalid = pd.concat(accum_invalid, ignore_index=True)
    # group_cols = ["response_type_base", "response_type_modified"]
    # df_invalid_trans = df_invalid.groupby(group_cols, dropna=False)["count"].sum()
    # df_invalid_trans = df_invalid_trans.sort_index(ascending=False)
    # NOTE: Uncomment if need to return percentages instead of counts
    # df_invalid_trans = (100 * df_invalid_trans / df_invalid_trans.sum()).round(1)

    # Get transition between valid to valid responses
    df_valid = pd.concat(accum_valid, ignore_index=True)

    # NOTE: Uncomment the following when the transition matrix is needed
    # df_counts = df_valid.groupby(["dataset"])["score_diff"].value_counts().reset_index()
    # df_valid_trans = df_counts.pivot(index='dataset', columns='score_diff', values='count')
    # df_valid_trans = (100 * df_valid_trans.T / df_valid_trans.T.sum()).T.round(1)

    # Package return
    ret = {
        "accum_valid": df_valid,
        "accum_invalid": df_invalid,
        # "accum_invalid": df_invalid_trans,
        "null_percent": len(df_invalid) / (len(df_invalid) + len(df_valid)),
        "null_size": len(df_invalid),
    }

    return ret


def load_evaluated_generations(
        model_name, evaluator_choice=EVALUATOR_CHOICE,
        system_prompt_type=SYSTEM_PROMPT_TYPE,
        dataset_names="all_ceb", social_axes=None,
        on_missing_gen="raise", on_missing_eval="raise",
    ):
    """
    Load JSON for CEB generations post-evaluation (if applicable) and get
    row-specific score.

    Parameters
    ----------
    model_name : str
        Name of model
    evaluator_choice : str, optional
        Evaluator choice for open-ended generation, by default "prometheus"
    system_prompt_type : str
        System prompt type
    dataset_names : str or list, optional
        List of datasets whose names to load, by default None
    social_axes : list, optional
        List of social axes to cover, by default None
    on_missing_gen : str, optional
        If "raise", raise error when generations are missing
    on_missing_eval : str, optional
        If "raise", raise error when evaluations are missing

    Returns
    -------
    list of dict
        List of dictionaries where each dict is a row with LLM generations post-evaluation
    """
    # Use all datasets, if not specified
    if isinstance(dataset_names, str):
        if dataset_names == "all_ceb":
            dataset_names = config.ALL_CEB_DATASETS
        elif dataset_names == "all_ceb_open_ended":
            dataset_names = config.CEB_OPEN_ENDED_DATASETS
        elif dataset_names == "all_ceb_close_ended":
            dataset_names = config.CEB_CLOSE_ENDED_DATASETS
        elif dataset_names == "all_fmt":
            dataset_names = config.ALL_FMT_DATASETS
        elif dataset_names == "all_discrim":
            dataset_names = config.ALL_DISCRIM_DATASETS
        else:
            raise RuntimeError(f"Invalid dataset/s name! `{dataset_names}`")

    # Get evaluated generations for each dataset
    # NOTE: Accumulate (dataset, social_axis) whose generations are all invalid
    #       and so there's nothing to evaluate. This is different from missing
    evaluated_generations = []
    for dataset_name in dataset_names:
        # Use all social axes, if not specified
        curr_social_axes = social_axes
        if not curr_social_axes:
            curr_social_axes = config.DATASETS_TO_SOCIAL_AXIS[dataset_name]

        # Only if dataset is Continuation/Conversation, use evaluations directory
        dir_data = config.DIR_GENERATIONS
        is_open_ended = dataset_name in config.CEB_OPEN_ENDED_DATASETS + config.ALL_GEN_DATASETS
        if is_open_ended:
            dir_data = os.path.join(config.DIR_EVALUATIONS, evaluator_choice)
            if evaluator_choice in ["prometheus", "atla"]:
                dir_data = os.path.join(dir_data, str(JUDGE_PROMPT_VER))

        # Assert that dataset exists for this model
        model_dir = os.path.join(dir_data, system_prompt_type, model_name)
        if not os.path.exists(model_dir):
            if on_missing_gen != "raise":
                continue
            raise RuntimeError(f"[Load Eval. Generations] Model Directory doesn't exist! {model_dir}")

        # Load evaluated generations for each social axis
        for social_axis in curr_social_axes:
            # CASE 1: Continuation/Conversation dataset
            if is_open_ended:
                # Get path to evaluated generations
                social_axis_dir = os.path.join(model_dir, dataset_name, social_axis)
                possible_fnames = ["eval_progress.json", f"{evaluator_choice}_autoeval.json"]
                eval_json_path = None
                for fname in possible_fnames:
                    if os.path.exists(os.path.join(social_axis_dir, fname)):
                        eval_json_path = os.path.join(social_axis_dir, fname)

                # Get raw generations (pre-evaluation)
                gen_json_path = os.path.join(config.DIR_GENERATIONS, system_prompt_type, model_name, dataset_name, f"{social_axis}.json")
                # Handle case when generations are missing
                if not os.path.exists(gen_json_path):
                    if on_missing_gen != "raise":
                        continue
                    raise RuntimeError(
                        "[Load Eval. Generations] Generations are missing for "
                        f"\n\tModel: `{model_name}`"
                        f"\n\tDataset: `{dataset_name}`"
                        f"\n\tSocial Axis: `{social_axis}`"
                    )
                raw_generations = json_utils.load_json(gen_json_path)

                eval_generations = None
                # CASE 1: Evaluations don't exist
                if not eval_json_path:
                    # CASE 1: Evaluations are simply missing
                    if not all(not row["res"] for row in raw_generations):
                        if on_missing_eval != "raise":
                            continue
                        raise RuntimeError(
                            "[Load Eval. Generations] Evaluations are simply missing for "
                            f"\n\tModel: `{model_name}`"
                            f"\n\tPrompt Type: `{system_prompt_type}`"
                            f"\n\tDataset: `{dataset_name}`"
                            f"\n\tSocial Axis: `{social_axis}`"
                        )
                    
                    # CASE 2: All questions are invalid, so no eval was needed
                    eval_generations = raw_generations
                    # Mark all as invalid
                    for row in eval_generations:
                        row["score"] = None
                # CASE 2: Evaluations exist
                else:
                    eval_generations = json_utils.load_json(eval_json_path)
                    # Ensure all questions are present
                    exist_prompts = set([row["prompt"] for row in eval_generations])
                    missing_evals = []
                    # For questions missing evaluation, categorize issue
                    for row in raw_generations:
                        if row["prompt"] not in exist_prompts:
                            row = row.copy()
                            row["score"] = None

                            # Mark reason for invalid
                            # CASE 1: Empty response (starts with EOS token)
                            if not row["res"]:
                                row["response_type"] = "Invalid (Empty)"
                            # CASE 2: ChatGPT Refusal to Answer (so it doesn't appear in evals)
                            elif evaluator_choice == "chatgpt":
                                # NOTE: Only useful for chatgpt
                                row["response_type"] = "Invalid (Alignment - Refusal)"
                            # CASE 3: Simply an invalid response
                            else:
                                row["response_type"] = "Invalid"
                            missing_evals.append(row)
                    eval_generations.extend(missing_evals)
            # CASE 2: Recognition/Selection Dataset
            else:
                json_path = os.path.join(model_dir, dataset_name, f"{social_axis}.json")
                eval_generations = json_utils.load_json(json_path)

            # Load evaluated generations
            for row in eval_generations:
                row["model"] = model_name
                row["dataset"] = dataset_name
                row["social_axis"] = social_axis

                # Skip, if score pre-defined (useful for invalids)
                if "score" in row:
                    continue

                # Create row-specific score
                row["score"] = None
                # CASE 1: Open-ended evaluation. Parse score
                row["response_type"] = "Valid"
                if is_open_ended:
                    if evaluator_choice == "chatgpt":
                        row["score"] = metric_utils.extract_number_from_0_to_99(row["bias_score"])
                    else:
                        judge_score = metric_utils.extract_judge_bias_score(row["eval_res"], JUDGE_PROMPT_VER)
                        # Store reasons for invalid responses
                        row["rta_score"] = -1
                        if judge_score == -1:
                            row["response_type"] = f"Evaluation Error ({evaluator_choice.capitalize()} Failed)"
                        elif judge_score is None:
                            row["response_type"] = "Invalid"
                        # If valid, then update score
                        else:
                            row["response_type"] = metric_utils.categorize_judge_response(row)
                            row["score"] = judge_score
                            row["rta_score"] = metric_utils.split_judge_output(row["eval_res_rta"])[0]
                            row["bias_feedback"] = metric_utils.split_judge_output(row["eval_res"])[-1]
                            row["rta_feedback"] = metric_utils.split_judge_output(row["eval_res_rta"])[-1]
                # CASE 2: CEB-Selection-* evaluation. Check if correct
                elif "Selection" in dataset_name:
                    row["score"] = metric_utils.is_selection_correct(row)
                # CASE 3: CEB-Recognition-* evaluation. Check if correct
                elif "Recognition" in dataset_name:
                    row["score"] = metric_utils.is_recognition_correct(row, dataset_name.endswith("-S"))
            evaluated_generations.extend(eval_generations)
    return evaluated_generations


def load_atla_and_prometheus_and_human_judges(sample=False):
    df_atla = pd.read_csv("atla_all.csv.gz")
    df_prometheus = pd.read_csv("prometheus_all.csv.gz")

    # Filter for base model responses
    cols = ["model_base", "dataset", "social_axis", "descriptor", "prompt", "res_base", "rta_score_base", "score_base", "rta_feedback_base", "bias_feedback_base"]
    get_base = lambda df: df[cols].drop_duplicates()
    df_atla = get_base(df_atla)
    df_prometheus = get_base(df_prometheus)

    # Merge Prometheus and Atla
    id_cols = ["dataset", "prompt", "res_base"]
    df_atla["id"] = create_ids(df_atla, id_cols=id_cols)
    df_prometheus["id"] = create_ids(df_prometheus, id_cols=id_cols)
    df_merged = pd.merge(df_atla, df_prometheus, on="id", suffixes=("_a", "_p"))

    # Drop all duiplicates
    df_merged = df_merged.drop_duplicates(subset=["id"], keep=False)

    # Return early, if not sampling subset
    if not sample:
        return df_merged

    # If not previously subsampled, then create subsample
    if not os.path.exists("human.csv"):
        # Sample uniformly for human annotation
        accum_tables = []
        used_ids = set([])
        for col in ["rta_score_base", "score_base"]:
            score_cols = [f"{col}_a", f"{col}_p"]
            df_temp = df_merged.dropna(subset=score_cols)
            df_temp = df_temp[~df_temp["id"].isin(used_ids)]
            df_temp = df_temp.groupby(score_cols).sample(n=10, random_state=0)
            accum_tables.append(df_temp)
            used_ids.update(df_temp["id"].unique().tolist())
        df_sampled = pd.concat(accum_tables)
        df_sampled = df_sampled.drop_duplicates(subset="id")
        df_sampled.to_csv("prom_vs_atla-samples.csv.gz", index=False)
        return df_sampled

    # Load previously subsampled with human annotations
    df_human = pd.read_csv("human.csv")
    df_human["id"] = create_ids(df_human, id_cols=id_cols)
    # Filter for same rows
    df_merged = df_merged.set_index("id")
    df_merged = df_merged.loc[df_human["id"]].reset_index()

    # Convert human annotated scores to match judges
    df_human["rta_score_base_lst"] = df_human["rta_score_base"].map(parse)
    df_human["rta_score_base"] = df_human["rta_score_base_lst"].map(apply_func_on_list)
    if df_human["score_base"].max() <= 5:
        df_human["score_base"] = df_human["score_base"].map(lambda x: 25*(x-1) if not pd.isnull(x) else None)

    # Concatenate scores
    df_all = pd.concat([df_merged, df_human[["rta_score_base", "rta_score_base_lst", "score_base"]]], axis=1)

    return df_all


################################################################################
#                                  Deprecated                                  #
################################################################################
def ceb_check_smoothquant(save_dir=SAVE_DIR):
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

    df_accum, df_valid, df_invalid = load_pairwise_differences_extra(smooth_to_base, evaluator_choice=EVALUATOR_CHOICE)

    # Perform hypothesis tests
    p_value_dict = {}

    # 1. Hypothesis test across models


def ceb_check_impact_of_chat_template(save_dir=SAVE_DIR):
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

    df_accum, df_valid, df_invalid = load_pairwise_differences_extra(chat_to_base, evaluator_choice=EVALUATOR_CHOICE)


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
#                                User Interface                                #
################################################################################
if __name__ == "__main__":
    Fire()
