import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pm4py.objects.log.importer.xes.importer import apply as xes_import
from cortado_core.eventually_follows_pattern_mining.algorithm import (
    generate_eventually_follows_patterns_from_groups,
    Algorithm,
)
from cortado_core.eventually_follows_pattern_mining.util.tree import (
    EventuallyFollowsStrategy,
)
from experiments.eventually_follows_pattern_mining.util import (
    get_support_count,
)
from experiments.subpattern_eval.exit_after import exit_after
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.utils.cvariants import get_concurrency_variants

REL_SUPPORTS = [
    1,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.35,
    0.3,
    0.25,
    0.2,
    0.15,
    0.1,
    0.075,
    0.05,
    0.025,
]
TIMEOUT = int(os.getenv("TIMEOUT", "300"))
LOG_FILE = os.getenv("LOG_FILE", "BPI_Challenge_2017.xes")
EVENT_LOG_DIRECTORY = os.getenv(
    "EVENT_LOG_DIRECTORY", "C:\\sources\\arbeit\\cortado\\event-logs"
)
RESULT_DIRECTORY = os.getenv(
    "RESULT_DIRECTORY",
    "C:\\sources\\arbeit\\cortado\\master_thesis\\Master_Thesis_Results_new",
)
FREQ_COUNT_STRAT = FrequencyCountingStrategy(
    int(
        os.getenv(
            "FREQ_COUNT_STRAT", FrequencyCountingStrategy.VariantTransaction.value
        )
    )
)
N_REPEATS = int(os.getenv("N_REPEATS", 2))
COLUMNS = [
    "Algorithm",
    "Rel_Support",
    "Is_Max_Closed",
    "Runtime",
    "Patterns",
    "Closed_Patterns",
    "Max_Patterns",
]
ALGORITHM_SOFT_EF_NAME = "Advanced Combination Miner Soft EF"
ALGORITHM_INFIX_PATTERNS = "Infix Pattern Miner"
ALGORITHM_ADVANCED_COMBINATION_MINER = "Advanced Combination Miner"
ALGORITHM_BRUTE_FORCE_COMBINATION_MINER = "Brute-Force Combination Miner"
ALGORITHM_RIGHTMOST_EXTENSION_MINER = "Rightmost Extension Miner"
sns_palette = sns.color_palette()
palette = {
    ALGORITHM_RIGHTMOST_EXTENSION_MINER: sns_palette[0],
    ALGORITHM_ADVANCED_COMBINATION_MINER: sns_palette[1],
    ALGORITHM_BRUTE_FORCE_COMBINATION_MINER: sns_palette[2],
    ALGORITHM_INFIX_PATTERNS: sns_palette[3],
    ALGORITHM_SOFT_EF_NAME: sns_palette[4],
}
sns.set(style="ticks", rc={"figure.figsize": (5, 3.5)})

ALGORITHMS = {
    ALGORITHM_RIGHTMOST_EXTENSION_MINER: (
        lambda v, s, f: generate_eventually_follows_patterns_from_groups(
            v, s, f, Algorithm.RightmostExpansion
        ),
        False,
    ),
    ALGORITHM_BRUTE_FORCE_COMBINATION_MINER: (
        lambda v, s, f: generate_eventually_follows_patterns_from_groups(
            v, s, f, Algorithm.InfixPatternCombinationBruteForce
        ),
        False,
    ),
    ALGORITHM_ADVANCED_COMBINATION_MINER: (
        lambda v, s, f: generate_eventually_follows_patterns_from_groups(
            v, s, f, Algorithm.InfixPatternCombinationEnumerationGraph
        ),
        False,
    ),
    ALGORITHM_SOFT_EF_NAME: (
        lambda v, s, f: generate_eventually_follows_patterns_from_groups(
            v,
            s,
            f,
            Algorithm.InfixPatternCombinationEnumerationGraph,
            ef_strategy=EventuallyFollowsStrategy.SoftEventuallyFollows,
        ),
        False,
    ),
    ALGORITHM_INFIX_PATTERNS: (
        lambda v, s, f: generate_eventually_follows_patterns_from_groups(
            v, s, f, Algorithm.RightmostExpansionOnlyInfixPatterns
        ),
        False,
    ),
}


def compare_algorithms():
    log_dir = EVENT_LOG_DIRECTORY
    log_filename = LOG_FILE
    frequency_strategy = FREQ_COUNT_STRAT
    log = xes_import(os.path.join(log_dir, log_filename))
    n_traces = len(log)
    variants = get_concurrency_variants(log)
    n_variants = len(variants)
    results = []
    timeouted_algos = set()
    df = None

    save_dir = os.path.join(
        RESULT_DIRECTORY,
        os.path.splitext(log_filename)[0],
        "algorithm_comparison",
        "data",
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    data_filename = os.path.join(save_dir, f"{frequency_strategy.name}.csv")

    for rel_support in REL_SUPPORTS:
        support_count = get_support_count(
            rel_support, frequency_strategy, n_traces, n_variants
        )
        for algorithm_name, (
            algorithm_func,
            is_max_closed_algorithm,
        ) in ALGORITHMS.items():
            print(algorithm_name, ":", rel_support)
            if algorithm_name in timeouted_algos:
                results.append(
                    get_timeout_result(
                        algorithm_name, rel_support, is_max_closed_algorithm
                    )
                )
                continue
            if is_max_closed_algorithm:
                mean_runtime, (closed, maximal) = measure_time(
                    algorithm_func,
                    get_algorithm_args(variants, support_count),
                    N_REPEATS,
                )
                if mean_runtime == -1:
                    results.append(
                        get_timeout_result(
                            algorithm_name, rel_support, is_max_closed_algorithm
                        )
                    )
                    timeouted_algos.add(algorithm_name)
                    continue
                results.append(
                    get_result_max_closed(
                        algorithm_name, rel_support, mean_runtime, closed, maximal
                    )
                )
            else:
                mean_runtime, patterns = measure_time(
                    algorithm_func,
                    get_algorithm_args(variants, support_count),
                    N_REPEATS,
                )
                if mean_runtime == -1:
                    results.append(
                        get_timeout_result(
                            algorithm_name, rel_support, is_max_closed_algorithm
                        )
                    )
                    timeouted_algos.add(algorithm_name)
                    continue
                results.append(
                    get_result_for_patterns(
                        algorithm_name, rel_support, mean_runtime, patterns
                    )
                )

        print("Save results for rel support", rel_support)
        df = pd.DataFrame(results, columns=COLUMNS)
        df.to_csv(data_filename)
    support_dict = get_lowest_support_dict_without_timeout(df)
    generate_plots(log_filename, data_filename)

    return support_dict


def get_timeout_result(algorithm_name, rel_support, is_max_closed):
    return [algorithm_name, rel_support, is_max_closed, -1, -1, -1, -1]


def get_result_for_patterns(algorithm_name, rel_support, runtime, patterns):
    if isinstance(patterns, dict):
        n_patterns = sum([len(p) for p in patterns.values()])
    else:
        n_patterns = len(patterns)

    return [algorithm_name, rel_support, False, runtime, n_patterns, -1, -1]


def get_result_max_closed(algorithm_name, rel_support, runtime, closed, maximal):
    return [
        algorithm_name,
        rel_support,
        True,
        runtime,
        len(closed) + len(maximal),
        len(closed),
        len(maximal),
    ]


def get_algorithm_args(variants, min_support_count):
    return [variants, min_support_count, FREQ_COUNT_STRAT]


@exit_after(TIMEOUT)
def run_algorithm(alg, args):
    return alg(*args)


def measure_time(alg, args, n_repeats):
    run_times = []
    res = None
    gc.collect()
    for i in range(n_repeats):
        start_time = time.time()
        try:
            res = run_algorithm(alg, args)
        except KeyboardInterrupt:
            return -1, (None, None)

        duration = time.time() - start_time
        gc.collect()
        run_times.append(duration)

    return np.mean(run_times), res


def generate_plots(log_filename, data_filename):
    df = pd.read_csv(data_filename)
    plot_dir = os.path.join(
        RESULT_DIRECTORY,
        os.path.splitext(log_filename)[0],
        "algorithm_comparison",
        "plots",
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    generate_runtime_plots(df, plot_dir)
    generate_number_of_patterns_plots(df, plot_dir)


def generate_runtime_plots(df, plot_dir):
    filtered_df = df[df["Algorithm"] != ALGORITHM_SOFT_EF_NAME]
    filtered_df = filtered_df[filtered_df["Runtime"] != -1].sort_values(
        "Rel_Support", ascending=False
    )
    hue_order = [
        ALGORITHM_RIGHTMOST_EXTENSION_MINER,
        ALGORITHM_BRUTE_FORCE_COMBINATION_MINER,
        ALGORITHM_ADVANCED_COMBINATION_MINER,
        ALGORITHM_INFIX_PATTERNS,
    ]
    plot = sns.lineplot(
        data=filtered_df,
        x="Rel_Support",
        y="Runtime",
        hue="Algorithm",
        markers=True,
        style="Algorithm",
        palette=palette,
        lw=1,
        hue_order=hue_order,
    )

    add_timeout_markers(
        filtered_df,
        "Runtime",
        {
            ALGORITHM_RIGHTMOST_EXTENSION_MINER,
            ALGORITHM_ADVANCED_COMBINATION_MINER,
            ALGORITHM_BRUTE_FORCE_COMBINATION_MINER,
            ALGORITHM_INFIX_PATTERNS,
        },
    )

    legend_handles = plot.axes.get_legend_handles_labels()
    plt.legend([], [], frameon=False)

    plot.set(yscale="log")
    plt.xlabel("Relative Support")
    plt.ylabel("Runtime (in seconds)")
    plot.set_xlim(plot.get_xlim()[::-1])
    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(
        os.path.join(plot_dir, f"runtime_{FREQ_COUNT_STRAT.name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    legend_fig = plt.figure()
    legend_fig.legend(legend_handles[0], legend_handles[1], ncol=4)
    legend_fig.savefig(
        os.path.join(plot_dir, f"runtime_{FREQ_COUNT_STRAT.name}_legend.pdf"),
        bbox_inches="tight",
    )
    plt.close(legend_fig)


def generate_number_of_patterns_plots(df, plot_dir):
    check_validity_of_pattern_numbers(df)
    filtered_df = df[df["Runtime"] != -1]

    patterns_df = filtered_df[
        filtered_df["Algorithm"] == ALGORITHM_ADVANCED_COMBINATION_MINER
    ]
    patterns_df.loc[:, "Algorithm"] = "Real EF Patterns"
    patterns_df = patterns_df.rename(
        columns={"Patterns": "Count", "Algorithm": "Pattern Type"}
    )
    patterns_df = patterns_df[["Pattern Type", "Count", "Rel_Support"]]

    infix_df = filtered_df[filtered_df["Algorithm"] == ALGORITHM_INFIX_PATTERNS]
    infix_df.loc[:, "Algorithm"] = "Infix Patterns"
    infix_df = infix_df.rename(
        columns={"Patterns": "Count", "Algorithm": "Pattern Type"}
    )
    infix_df = infix_df[["Pattern Type", "Count", "Rel_Support"]]

    soft_df = filtered_df[filtered_df["Algorithm"] == ALGORITHM_SOFT_EF_NAME]
    soft_df.loc[:, "Algorithm"] = "Soft EF Patterns"
    soft_df = soft_df.rename(columns={"Patterns": "Count", "Algorithm": "Pattern Type"})
    soft_df = soft_df[["Pattern Type", "Count", "Rel_Support"]]

    plot_df = pd.concat([patterns_df, infix_df, soft_df], ignore_index=True)
    plot_df = plot_df.sort_values("Rel_Support", ascending=False)

    sns_palette = sns.color_palette()
    palette = {
        "Real EF Patterns": sns_palette[0],
        "Soft EF Patterns": sns_palette[1],
        "Infix Patterns": sns_palette[3],
    }
    hue_order = ["Real EF Patterns", "Soft EF Patterns", "Infix Patterns"]
    plot = sns.lineplot(
        data=plot_df,
        x="Rel_Support",
        y="Count",
        hue="Pattern Type",
        markers=True,
        style="Pattern Type",
        lw=1,
        palette=palette,
        hue_order=hue_order,
    )
    add_timeout_markers(
        plot_df,
        "Count",
        {"Real EF Patterns", "Soft EF Patterns", "Infix Patterns"},
        "Pattern Type",
    )

    plot.set(yscale="log")
    plt.xlabel("Relative Support")
    plt.ylabel("Number of Frequent Valid Patterns")
    plot.set_xlim(plot.get_xlim()[::-1])
    legend_handles = plot.axes.get_legend_handles_labels()
    plt.legend([], [], frameon=False)
    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(
        os.path.join(plot_dir, f"n_patterns_{FREQ_COUNT_STRAT.name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    legend_fig = plt.figure()
    legend_fig.legend(legend_handles[0], legend_handles[1], ncol=3)
    legend_fig.savefig(
        os.path.join(plot_dir, f"n_patterns_{FREQ_COUNT_STRAT.name}_legend.pdf"),
        bbox_inches="tight",
    )
    plt.close(legend_fig)


def add_timeout_markers(
    df: pd.DataFrame,
    y_attribute: str,
    alg_selection: set[str],
    id_column: str = "Algorithm",
):
    x_values_timeout, y_values_timeout = get_timeout_markers(
        df, y_attribute, alg_selection, id_column
    )
    if len(x_values_timeout) > 0:
        plt.plot(x_values_timeout, y_values_timeout, "kX")


def get_timeout_markers(
    df: pd.DataFrame, y_attribute: str, alg_selection: set[str], id_column: str
):
    x_values_timeout = []
    y_values_timeout = []

    for algorithm in alg_selection:
        alg_filtered_df = df[df[id_column] == algorithm]
        min_rel_support = min(alg_filtered_df["Rel_Support"])
        y_attribute_min_value = alg_filtered_df[
            alg_filtered_df["Rel_Support"] == min_rel_support
        ][y_attribute].iloc[0]
        if min_rel_support > REL_SUPPORTS[-1]:
            x_values_timeout.append(min_rel_support)
            y_values_timeout.append(y_attribute_min_value)

    return x_values_timeout, y_values_timeout


def get_lowest_support_dict_without_timeout(df: pd.DataFrame) -> dict[str, float]:
    df = df[df["Runtime"] != -1]
    res = {}
    for alg in ALGORITHMS.keys():
        alg_filtered_df = df[df["Algorithm"] == alg]
        min_rel_support = min(alg_filtered_df["Rel_Support"])
        res[alg] = min_rel_support

    return res


def check_validity_of_pattern_numbers(df):
    df = df[df["Algorithm"] != ALGORITHM_SOFT_EF_NAME]

    for rel_support in REL_SUPPORTS:
        filtered_df = df[df["Is_Max_Closed"] == False][
            df["Algorithm"] != ALGORITHM_INFIX_PATTERNS
        ][df["Rel_Support"] == rel_support][df["Patterns"] != -1][
            ["Patterns", "Algorithm"]
        ]
        pattern_set = set(filtered_df["Patterns"])
        if len(pattern_set) > 1:
            raise Exception(
                "Rel support: ",
                rel_support,
                "(count of frequent patterns does not match)",
            )


if __name__ == "__main__":
    # generate_plots('ccc19.xes',
    #                'C:\\sources\\arbeit\\cortado\\master_thesis\\ccc19\\algorithm_comparison\\data\\TraceTransaction.csv')
    compare_algorithms()
