import os
import pickle
from pathlib import Path
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from cortado_core.eventually_follows_pattern_mining.obj import EventuallyFollowsPattern

from cortado_core.eventually_follows_pattern_mining.util.pattern import flatten_patterns

from cortado_core.eventually_follows_pattern_mining.algorithm import (
    generate_eventually_follows_patterns_from_groups,
)
from experiments.eventually_follows_pattern_mining.algorithms_comparison import (
    REL_SUPPORTS,
)
from experiments.eventually_follows_pattern_mining.overlap_support_count import (
    filter_patterns,
)

from experiments.eventually_follows_pattern_mining.util import (
    get_support_count,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.utils.cvariants import get_concurrency_variants
from cortado_core.utils.timestamp_utils import TimeUnit
from pm4py.objects.log.importer.xes.importer import apply as xes_import

REL_SUPPORT = float(os.getenv("MIN_REL_SUP_OVERLAP", "0.1"))
LOG_FILE = os.getenv("LOG_FILE", "BPI_Ch_2020_PrepaidTravelCost.xes")
EVENT_LOG_DIRECTORY = os.getenv(
    "EVENT_LOG_DIRECTORY", "C:\\sources\\arbeit\\cortado\\event-logs"
)
RESULT_DIRECTORY = os.getenv(
    "RESULT_DIRECTORY",
    "C:\\sources\\arbeit\\cortado\\master_thesis\\results_time_granularity",
)
FREQ_COUNT_STRAT = FrequencyCountingStrategy(
    int(os.getenv("FREQ_COUNT_STRAT", FrequencyCountingStrategy.TraceTransaction.value))
)
COLUMNS = ["Counting Strategy", "Number of Infix Subtrees"]
SAVE_DIR = os.path.join(
    RESULT_DIRECTORY, os.path.splitext(LOG_FILE)[0], "algorithm_comparison", "data"
)
TIME_AGGREGATION_MAP = {
    TimeUnit.HOUR.name: "Hours",
    TimeUnit.DAY.name: "Days",
    TimeUnit.MIN.name: "Minutes",
    TimeUnit.SEC.name: "Seconds",
    TimeUnit.MS.name: "Milliseconds",
}
sns_palette = sns.color_palette("mako", n_colors=6)
palette = {
    "Milliseconds": sns_palette[0],
    "Seconds": sns_palette[1],
    "Minutes": sns_palette[2],
    "Hours": sns_palette[3],
    "Days": sns_palette[4],
    "Month": sns_palette[5],
}
sns.set(style="ticks", rc={"figure.figsize": (5, 3.5)})


def time_granularity_distribution_experiments():
    if FREQ_COUNT_STRAT != FrequencyCountingStrategy.TraceTransaction:
        return

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(SAVE_DIR, "patterns")).mkdir(parents=True, exist_ok=True)

    log_dir = EVENT_LOG_DIRECTORY
    log_filename = LOG_FILE
    log = xes_import(os.path.join(log_dir, log_filename))

    for time_granularity in TimeUnit:
        if time_granularity == TimeUnit.MONTH:
            continue
        n_traces = len(log)
        variants = get_concurrency_variants(log, time_granularity=time_granularity)
        n_variants = len(variants)
        support_count = get_support_count(
            REL_SUPPORT,
            FrequencyCountingStrategy.TraceTransaction,
            n_traces,
            n_variants,
        )
        patterns = generate_eventually_follows_patterns_from_groups(
            variants, support_count, FrequencyCountingStrategy.TraceTransaction
        )
        flat_patterns = flatten_patterns(patterns)
        with open(
            os.path.join(
                SAVE_DIR,
                "patterns",
                f"patterns_{FREQ_COUNT_STRAT}_{time_granularity}_{REL_SUPPORT}.pkl",
            ),
            "wb",
        ) as handle:
            pickle.dump(flat_patterns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    create_plots()


def create_plots():
    create_distribution_plot()
    generate_number_of_patterns_plot()


def create_distribution_plot():
    plot_dir = os.path.join(
        RESULT_DIRECTORY, os.path.splitext(LOG_FILE)[0], "algorithm_comparison", "plots"
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    df = get_time_granularity_distribution_df()
    sns.boxplot(data=df, x=COLUMNS[0], y=COLUMNS[1])
    plt.xlabel("Time Aggregation")
    plt.ylabel("Infix Subtrees per Pattern")
    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(
        os.path.join(plot_dir, "time_granularity_distribution.pdf"), bbox_inches="tight"
    )
    plt.close()


def get_time_granularity_distribution_df():
    results = []

    for time_granularity in TimeUnit:
        if time_granularity == TimeUnit.MONTH:
            continue
        with open(
            os.path.join(
                SAVE_DIR,
                "patterns",
                f"patterns_{FrequencyCountingStrategy.TraceTransaction}_{time_granularity}_{REL_SUPPORT}.pkl",
            ),
            "rb",
        ) as handle:
            patterns: list[EventuallyFollowsPattern] = pickle.load(handle)

            for pattern in patterns:
                results.append(
                    [TIME_AGGREGATION_MAP[time_granularity.name], len(pattern)]
                )

    return pd.DataFrame(results, columns=COLUMNS)


def generate_number_of_patterns_plot():
    df = get_pattern_count_df()
    df = df.sort_values("Relative Support", ascending=False)
    hue_order = ["Milliseconds", "Seconds", "Minutes", "Hours", "Days"]
    plot = sns.lineplot(
        data=df,
        x="Relative Support",
        y="Number of Frequent Valid Patterns",
        hue="Time Granularity",
        markers=True,
        hue_order=hue_order,
        style="Time Granularity",
        palette=palette,
        lw=1,
        estimator=None,
    )

    legend_handles = plot.axes.get_legend_handles_labels()
    plt.legend([], [], frameon=False)

    plot.set(yscale="log")
    plt.xlabel("Relative Support")
    plt.ylabel("Number of Frequent Valid Patterns")
    plot.set_xlim(plot.get_xlim()[::-1])

    # call `draw` to re-render the graph
    plt.draw()
    plot_dir = os.path.join(
        RESULT_DIRECTORY, os.path.splitext(LOG_FILE)[0], "algorithm_comparison", "plots"
    )
    plt.savefig(
        os.path.join(plot_dir, f"n_patterns_time_granularity_trace_transaction.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    legend_fig = plt.figure()
    legend_fig.legend(legend_handles[0], legend_handles[1], ncol=5)
    legend_fig.savefig(
        os.path.join(plot_dir, f"time_granularity_legend.pdf"), bbox_inches="tight"
    )
    plt.close(legend_fig)


def get_pattern_count_df():
    results = []
    log = xes_import(os.path.join(EVENT_LOG_DIRECTORY, LOG_FILE))
    n_traces = len(log)

    for time_granularity in TimeUnit:
        if time_granularity == TimeUnit.MONTH:
            continue
        with open(
            os.path.join(
                SAVE_DIR,
                "patterns",
                f"patterns_{FrequencyCountingStrategy.TraceTransaction}_{time_granularity}_{REL_SUPPORT}.pkl",
            ),
            "rb",
        ) as handle:
            patterns: list[EventuallyFollowsPattern] = pickle.load(handle)

        for rel_support in REL_SUPPORTS:
            if rel_support < REL_SUPPORT:
                break
            support_count = round(n_traces * rel_support)
            filtered_patterns = filter_patterns(patterns, support_count)
            results.append(
                [
                    TIME_AGGREGATION_MAP[time_granularity.name],
                    len(filtered_patterns),
                    rel_support,
                ]
            )

    return pd.DataFrame(
        results,
        columns=[
            "Time Granularity",
            "Number of Frequent Valid Patterns",
            "Relative Support",
        ],
    )


if __name__ == "__main__":
    # time_granularity_distribution_experiments()
    create_plots()
