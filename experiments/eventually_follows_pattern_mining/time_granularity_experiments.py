import os
import pickle
from pathlib import Path

import pandas as pd
from pm4py.objects.log.importer.xes.importer import apply as xes_import
import seaborn as sns

from cortado_core.eventually_follows_pattern_mining.algorithm import (
    generate_eventually_follows_patterns_from_groups,
)
from cortado_core.eventually_follows_pattern_mining.util.pattern import flatten_patterns
from experiments.eventually_follows_pattern_mining.algorithms_comparison import (
    measure_time,
)
from cortado_core.utils.split_graph import LeafGroup, ParallelGroup, SequenceGroup
from cortado_core.utils.timestamp_utils import TimeUnit
from matplotlib import pyplot as plt

from experiments.eventually_follows_pattern_mining.util import (
    get_support_count,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.utils.cvariants import get_concurrency_variants

REL_SUPPORTS = [0.95, 0.75, 0.55, 0.35, 0.3, 0.25, 0.2, 0.15]
TIMEOUT = int(os.getenv("TIMEOUT", "50"))
LOG_FILE = os.getenv("LOG_FILE", "BPI_CH_2020_PrepaidTravelCost.xes")
EVENT_LOG_DIRECTORY = os.getenv(
    "EVENT_LOG_DIRECTORY", "C:\\sources\\arbeit\\cortado\\event-logs"
)
RESULT_DIRECTORY = os.getenv(
    "RESULT_DIRECTORY", "C:\\sources\\arbeit\\cortado\\master_thesis"
)
FREQ_COUNT_STRAT = FrequencyCountingStrategy(
    int(os.getenv("FREQ_COUNT_STRAT", FrequencyCountingStrategy.TraceTransaction.value))
)
N_REPEATS = int(os.getenv("N_REPEATS", 1))
COLUMNS = ["Rel_Support", "Time_Granularity", "Runtime", "Patterns"]
COLUMNS_VARIANTS = [
    "Time_Granularity",
    "Variants",
    "Fallthrough Nodes",
    "Sequential Nodes",
    "Parallel Nodes",
]
sns_palette = sns.color_palette("mako", n_colors=6)
palette = {
    "Milliseconds": sns_palette[0],
    "Seconds": sns_palette[1],
    "Minutes": sns_palette[2],
    "Hours": sns_palette[3],
    "Days": sns_palette[4],
    "Month": sns_palette[5],
}


def run_time_granularity_experiments():
    log_dir = EVENT_LOG_DIRECTORY
    log_filename = LOG_FILE
    frequency_strategy = FREQ_COUNT_STRAT
    log = xes_import(os.path.join(log_dir, log_filename))
    n_traces = len(log)
    variants_result = []
    results = []
    save_dir = os.path.join(
        RESULT_DIRECTORY, os.path.splitext(log_filename)[0], "time_granularity", "data"
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for time_granularity in TimeUnit:
        variants = get_concurrency_variants(log, time_granularity=time_granularity)
        (
            n_variants,
            n_fallthrough_nodes,
            n_sequential_nodes,
            n_parallel_nodes,
        ) = get_variants_statistics(variants)
        variants_result.append(
            [
                time_granularity,
                n_variants,
                n_fallthrough_nodes,
                n_sequential_nodes,
                n_parallel_nodes,
            ]
        )
        is_timeout = False

        for rel_support in REL_SUPPORTS:
            support_count = get_support_count(
                rel_support, frequency_strategy, n_traces, n_variants
            )
            if is_timeout:
                results.append(get_timeout_result(rel_support, time_granularity))
                with open(
                    os.path.join(
                        save_dir, f"patterns_{time_granularity}_{rel_support}.pkl"
                    ),
                    "wb",
                ) as file:
                    pickle.dump("timeout", file)
                continue

            mean_runtime, patterns = measure_time(
                generate_eventually_follows_patterns_from_groups,
                (variants, support_count, frequency_strategy),
                N_REPEATS,
            )

            if mean_runtime == -1:
                is_timeout = True
                results.append(get_timeout_result(rel_support, time_granularity))
                with open(
                    os.path.join(
                        save_dir, f"patterns_{time_granularity}_{rel_support}.pkl"
                    ),
                    "wb",
                ) as file:
                    pickle.dump("timeout", file)
                continue

            with open(
                os.path.join(
                    save_dir, f"patterns_{time_granularity}_{rel_support}.pkl"
                ),
                "wb",
            ) as file:
                pickle.dump({str(p) for p in flatten_patterns(patterns)}, file)
            results.append(
                get_result(rel_support, time_granularity, mean_runtime, patterns)
            )

    df = pd.DataFrame(results, columns=COLUMNS)
    data_filename = os.path.join(save_dir, f"{frequency_strategy.name}.csv")
    df.to_csv(data_filename, index=False)

    variants_filename = os.path.join(save_dir, f"variants.csv")
    variants_df = pd.DataFrame(variants_result, columns=COLUMNS_VARIANTS)
    variants_df.to_csv(variants_filename)

    generate_plots(log_filename, data_filename, variants_filename, save_dir)


def get_timeout_result(rel_support, time_granularity):
    return [rel_support, time_granularity, -1, -1]


def get_result(rel_support, time_granularity, mean_runtime, patterns):
    if isinstance(patterns, dict):
        n_patterns = sum([len(p) for p in patterns.values()])
    else:
        n_patterns = len(patterns)

    return [rel_support, time_granularity, mean_runtime, n_patterns]


def get_variants_statistics(variants):
    n_variants = len(variants)
    n_fallthrough_nodes = sum([get_fallthrough_nodes(v) for v in variants])
    n_sequential_nodes = sum([get_sequential_nodes(v) for v in variants])
    n_parallel_nodes = sum([get_parallel_nodes(v) for v in variants])

    return n_variants, n_fallthrough_nodes, n_sequential_nodes, n_parallel_nodes


def get_fallthrough_nodes(variant):
    if isinstance(variant, LeafGroup):
        if len(variant) == 1:
            return 0
        return 1

    return sum([get_fallthrough_nodes(child) for child in variant])


def get_parallel_nodes(variant):
    if isinstance(variant, LeafGroup):
        return 0

    if isinstance(variant, ParallelGroup):
        return 1 + sum([get_parallel_nodes(child) for child in variant])

    return sum([get_parallel_nodes(child) for child in variant])


def get_sequential_nodes(variant):
    if isinstance(variant, LeafGroup):
        return 0

    if isinstance(variant, SequenceGroup):
        return 1 + sum([get_sequential_nodes(child) for child in variant])

    return sum([get_sequential_nodes(child) for child in variant])


def generate_plots(log_filename, data_filename, variants_filename, save_dir):
    df = pd.read_csv(data_filename)
    plot_dir = os.path.join(
        RESULT_DIRECTORY, os.path.splitext(log_filename)[0], "time_granularity", "plots"
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    generate_runtime_plots(df, plot_dir)
    generate_number_of_patterns_plots(df, plot_dir)
    generate_number_of_variants_plot(pd.read_csv(variants_filename), plot_dir)
    generate_variant_operator_distribution_plot(
        pd.read_csv(variants_filename), plot_dir
    )
    generate_time_granularity_heatmap(save_dir, plot_dir)


def generate_runtime_plots(df, plot_dir):
    filtered_df = df[df["Runtime"] != -1].sort_values("Rel_Support", ascending=False)
    plot = sns.lineplot(
        data=filtered_df,
        x="Rel_Support",
        y="Runtime",
        hue="Time_Granularity",
        markers=True,
        style="Time_Granularity",
        palette=palette,
        lw=1,
        estimator=None,
    )
    x_values_timeout = []
    y_values_timeout = []

    for time_granularity in TimeUnit:
        alg_filtered_df = filtered_df[
            filtered_df["Time_Granularity"] == time_granularity
        ]
        min_rel_support = min(alg_filtered_df["Rel_Support"])
        runtime = alg_filtered_df[alg_filtered_df["Rel_Support"] == min_rel_support][
            "Runtime"
        ].iloc[0]
        if min_rel_support > REL_SUPPORTS[-1]:
            x_values_timeout.append(min_rel_support)
            y_values_timeout.append(runtime)

    if len(x_values_timeout) > 0:
        plt.plot(x_values_timeout, y_values_timeout, "kX")
    plot.set(yscale="log")
    plt.title("Runtime for different time granularities (in seconds)")
    plt.xlabel("Relative Support")
    plt.ylabel("Runtime (in seconds)")
    plot.set_xlim(plot.get_xlim()[::-1])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [
        labels.index(l)
        for l in ["Milliseconds", "Seconds", "Minutes", "Hours", "Days", "Month"]
    ]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(
        os.path.join(plot_dir, f"runtime_{FREQ_COUNT_STRAT.name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def generate_number_of_patterns_plots(df, plot_dir):
    filtered_df = df[df["Runtime"] != -1].sort_values("Rel_Support", ascending=False)
    plot = sns.lineplot(
        data=filtered_df,
        x="Rel_Support",
        y="Patterns",
        hue="Time_Granularity",
        markers=True,
        style="Time_Granularity",
        palette=palette,
        lw=1,
        estimator=None,
    )
    x_values_timeout = []
    y_values_timeout = []

    for time_granularity in TimeUnit:
        alg_filtered_df = filtered_df[
            filtered_df["Time_Granularity"] == time_granularity
        ]
        min_rel_support = min(alg_filtered_df["Rel_Support"])
        n_patterns = alg_filtered_df[alg_filtered_df["Rel_Support"] == min_rel_support][
            "Patterns"
        ].iloc[0]
        if min_rel_support > REL_SUPPORTS[-1]:
            x_values_timeout.append(min_rel_support)
            y_values_timeout.append(n_patterns)

    if len(x_values_timeout) > 0:
        plt.plot(x_values_timeout, y_values_timeout, "kX")
    plot.set(yscale="log")
    plt.title("Number of patterns for different time granularities")
    plt.xlabel("Relative Support")
    plt.ylabel("Number of patterns")
    plot.set_xlim(plot.get_xlim()[::-1])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [
        labels.index(l)
        for l in ["Milliseconds", "Seconds", "Minutes", "Hours", "Days", "Month"]
    ]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(
        os.path.join(plot_dir, f"n_patterns_{FREQ_COUNT_STRAT.name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def generate_number_of_variants_plot(df, plot_dir):
    df["Time_Granularity"] = pd.CategoricalIndex(
        df["Time_Granularity"], categories=palette.keys(), ordered=True
    )
    plot = sns.lineplot(
        data=df, x="Time_Granularity", y="Variants", sort=False, estimator=None
    )
    plt.title("Number of variants for different time granularities")
    plt.xlabel("Time granularity")
    plt.ylabel("Number of variants")
    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(os.path.join(plot_dir, f"n_variants.pdf"), bbox_inches="tight")
    plt.close()


def generate_variant_operator_distribution_plot(df, plot_dir):
    df["Time_Granularity"] = pd.CategoricalIndex(
        df["Time_Granularity"], categories=palette.keys(), ordered=True
    )
    plot_df = df.melt(
        id_vars="Time_Granularity",
        var_name="Operator Type",
        value_vars=["Fallthrough Nodes", "Sequential Nodes", "Parallel Nodes"],
    )
    plot = sns.lineplot(
        data=plot_df,
        x="Time_Granularity",
        y="value",
        hue="Operator Type",
        markers=True,
        style="Operator Type",
        lw=1,
        sort=False,
        estimator=None,
    )
    plt.title("Number of operators in variants")
    plt.xlabel("Time granularity")
    plt.ylabel("Number of operators")
    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(
        os.path.join(plot_dir, f"operator_distribution_in_variants.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def generate_time_granularity_heatmap(save_dir, plot_dir):
    res_columns = ["from", "to", "overlap"]

    for rel_support in REL_SUPPORTS:
        res = []

        for tg_from in TimeUnit:
            with open(
                os.path.join(save_dir, f"patterns_{tg_from}_{rel_support}.pkl"), "rb"
            ) as from_file:
                patterns_from = pickle.load(from_file)
            for tg_to in TimeUnit:
                with open(
                    os.path.join(save_dir, f"patterns_{tg_to}_{rel_support}.pkl"), "rb"
                ) as to_file:
                    patterns_to = pickle.load(to_file)

                if patterns_from == "timeout" or patterns_to == "timeout":
                    res.append([tg_from.value, tg_to.value, -1])
                    continue

                res.append(
                    [
                        tg_from.value,
                        tg_to.value,
                        round(
                            len(patterns_from.intersection(patterns_to))
                            / len(patterns_from),
                            2,
                        ),
                    ]
                )

        df = pd.DataFrame(res, columns=res_columns)
        df = df.pivot("from", "to", "overlap")
        df.index = pd.CategoricalIndex(
            df.index,
            categories=["Milliseconds", "Seconds", "Minutes", "Hours", "Days", "Month"],
        )
        df.sort_index(level=0, inplace=True)
        df = df[["Milliseconds", "Seconds", "Minutes", "Hours", "Days", "Month"]]
        sns.heatmap(df, annot=True)
        plt.draw()
        plt.savefig(
            os.path.join(
                plot_dir, f"overlap_{rel_support}_{FREQ_COUNT_STRAT.name}.pdf"
            ),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    # log_filename = LOG_FILE
    # frequency_strategy = FREQ_COUNT_STRAT
    # save_dir = os.path.join(RESULT_DIRECTORY, os.path.splitext(log_filename)[0], 'time_granularity', 'data')
    # data_filename = os.path.join(save_dir, f'{frequency_strategy.name}.csv')
    # variants_filename = os.path.join(save_dir, f'variants.csv')
    # generate_plots(log_filename, data_filename, variants_filename, save_dir)
    run_time_granularity_experiments()
