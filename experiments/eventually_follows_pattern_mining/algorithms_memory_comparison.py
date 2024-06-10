import gc
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from pm4py.objects.log.importer.xes.importer import apply as xes_import
import seaborn as sns
from matplotlib import pyplot as plt

from cortado_core.eventually_follows_pattern_mining.algorithm import (
    generate_eventually_follows_patterns_from_groups,
    Algorithm,
)
from cortado_core.eventually_follows_pattern_mining.occurrence_store.occurrence_statistic_tracker import (
    MaxOccurrenceStatisticTracker,
)
from experiments.eventually_follows_pattern_mining.algorithms_comparison import (
    measure_time,
    ALGORITHM_INFIX_PATTERNS,
    ALGORITHM_ADVANCED_COMBINATION_MINER,
    ALGORITHM_BRUTE_FORCE_COMBINATION_MINER,
    ALGORITHM_RIGHTMOST_EXTENSION_MINER,
    add_timeout_markers,
    REL_SUPPORTS,
)
from experiments.eventually_follows_pattern_mining.util import (
    get_support_count,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.utils.cvariants import get_concurrency_variants

LOG_FILE = os.getenv("LOG_FILE", "sepsis_cases.xes")
EVENT_LOG_DIRECTORY = os.getenv(
    "EVENT_LOG_DIRECTORY", "C:\\sources\\arbeit\\cortado\\event-logs"
)
RESULT_DIRECTORY = os.getenv(
    "RESULT_DIRECTORY",
    "C:\\sources\\arbeit\\cortado\\master_thesis\\Master_Thesis_Results_new",
)
FREQ_COUNT_STRAT = FrequencyCountingStrategy(
    int(os.getenv("FREQ_COUNT_STRAT", FrequencyCountingStrategy.TraceTransaction.value))
)
COLUMNS = ["Algorithm", "Rel_Support", "Max_Occ_List_Size"]
sns_palette = sns.color_palette()
palette = {
    ALGORITHM_RIGHTMOST_EXTENSION_MINER: sns_palette[0],
    ALGORITHM_ADVANCED_COMBINATION_MINER: sns_palette[1],
    ALGORITHM_BRUTE_FORCE_COMBINATION_MINER: sns_palette[2],
    ALGORITHM_INFIX_PATTERNS: sns_palette[3],
}
SAVE_DIR = os.path.join(
    RESULT_DIRECTORY, os.path.splitext(LOG_FILE)[0], "algorithm_comparison", "data"
)
sns.set(style="ticks", rc={"figure.figsize": (5, 3.5)})
ALGORITHMS = {
    ALGORITHM_RIGHTMOST_EXTENSION_MINER: lambda v, s, f, tracker: generate_eventually_follows_patterns_from_groups(
        v, s, f, Algorithm.RightmostExpansion, size_tracker=tracker
    ),
    ALGORITHM_BRUTE_FORCE_COMBINATION_MINER: lambda v, s, f, tracker: generate_eventually_follows_patterns_from_groups(
        v, s, f, Algorithm.InfixPatternCombinationBruteForce, size_tracker=tracker
    ),
    ALGORITHM_ADVANCED_COMBINATION_MINER: lambda v, s, f, tracker: generate_eventually_follows_patterns_from_groups(
        v, s, f, Algorithm.InfixPatternCombinationEnumerationGraph, size_tracker=tracker
    ),
    ALGORITHM_INFIX_PATTERNS: lambda v, s, f, tracker: generate_eventually_follows_patterns_from_groups(
        v, s, f, Algorithm.RightmostExpansionOnlyInfixPatterns, size_tracker=tracker
    ),
}


def compare_memory_of_algorithms(support_dict: Optional[dict[str, int]] = None):
    if support_dict is None:
        support_dict = {alg: REL_SUPPORTS[-1] for alg in ALGORITHMS.keys()}

    print("frequency counting strategy", FREQ_COUNT_STRAT)
    log_dir = EVENT_LOG_DIRECTORY
    log_filename = LOG_FILE
    frequency_strategy = FREQ_COUNT_STRAT
    log = xes_import(os.path.join(log_dir, log_filename))

    n_traces = len(log)
    variants = get_concurrency_variants(log)
    n_variants = len(variants)
    results = []
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    data_filename = os.path.join(
        SAVE_DIR, f"{frequency_strategy.name}_memory_allocation.csv"
    )

    for rel_support in REL_SUPPORTS:
        support_count = get_support_count(
            rel_support, frequency_strategy, n_traces, n_variants
        )
        for algorithm_name, algorithm_func in ALGORITHMS.items():
            if support_dict[algorithm_name] > rel_support:
                continue
            tracker = MaxOccurrenceStatisticTracker()
            _ = algorithm_func(variants, support_count, FREQ_COUNT_STRAT, tracker)
            results.append(
                get_result(
                    algorithm_name, rel_support, tracker.get_max_occurrence_size()
                )
            )
            gc.collect()

        print("Save results for rel support", rel_support)
        df = pd.DataFrame(results, columns=COLUMNS)
        df.to_csv(data_filename)
    generate_plots(log_filename, data_filename)


def get_timeout_result(algorithm_name, rel_support):
    return [algorithm_name, rel_support, -1]


def get_result(algorithm_name, rel_support, max_occ_list_size):
    return [algorithm_name, rel_support, max_occ_list_size]


def get_algorithm_args(variants, min_support_count, tracker):
    return [variants, min_support_count, FREQ_COUNT_STRAT, tracker]


def generate_plots(log_filename, data_filename):
    df = pd.read_csv(data_filename)
    plot_dir = os.path.join(
        RESULT_DIRECTORY,
        os.path.splitext(log_filename)[0],
        "algorithm_comparison",
        "plots",
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    generate_memory_plot(df, plot_dir)


def generate_memory_plot(df, plot_dir):
    filtered_df = df[df["Max_Occ_List_Size"] != -1].sort_values(
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
        y="Max_Occ_List_Size",
        hue="Algorithm",
        markers=True,
        style="Algorithm",
        palette=palette,
        lw=1,
        hue_order=hue_order,
    )

    add_timeout_markers(filtered_df, "Max_Occ_List_Size", set(ALGORITHMS.keys()))

    legend_handles = plot.axes.get_legend_handles_labels()
    plt.legend([], [], frameon=False)

    plot.set(yscale="log")
    plt.xlabel("Relative Support")
    plt.ylabel("Maximal Occurrence List Size")
    plot.set_xlim(plot.get_xlim()[::-1])

    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(
        os.path.join(plot_dir, f"memory_allocation_{FREQ_COUNT_STRAT.name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    legend_fig = plt.figure()
    legend_fig.legend(legend_handles[0], legend_handles[1], ncol=4)
    legend_fig.savefig(
        os.path.join(plot_dir, f"memory_allocation_{FREQ_COUNT_STRAT.name}_legend.pdf"),
        bbox_inches="tight",
    )
    plt.close(legend_fig)


if __name__ == "__main__":
    # generate_plots('ccc19.xes',
    #                'C:\\sources\\arbeit\\cortado\\master_thesis\\ccc19\\algorithm_comparison\\data\\TraceTransaction.csv')
    compare_memory_of_algorithms()
