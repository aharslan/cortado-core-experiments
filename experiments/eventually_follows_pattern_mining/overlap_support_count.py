import os
import pickle
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd
from cortado_core.eventually_follows_pattern_mining.obj import EventuallyFollowsPattern

from cortado_core.eventually_follows_pattern_mining.util.pattern import flatten_patterns
from pm4py.objects.log.importer.xes.importer import apply as xes_import
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.eventually_follows_pattern_mining.algorithms_comparison import (
    REL_SUPPORTS,
)
from cortado_core.eventually_follows_pattern_mining.algorithm import (
    generate_eventually_follows_patterns_from_groups,
)
from experiments.eventually_follows_pattern_mining.util import (
    get_support_count,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.utils.cvariants import get_concurrency_variants

REL_SUPPORT = float(os.getenv("MIN_REL_SUP_OVERLAP", "0.1"))
LOG_FILE = os.getenv("LOG_FILE", "BPI_Ch_2020_PrepaidTravelCost.xes")
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
COLUMNS = ["Rel_Support", "n_patterns_trans", "n_patterns_occ", "overlap"]
SAVE_DIR = os.path.join(
    RESULT_DIRECTORY, os.path.splitext(LOG_FILE)[0], "algorithm_comparison", "data"
)
sns.set(style="ticks", rc={"figure.figsize": (5, 3.5)})


def overlap_experiments():
    # only execute these experiments if they are started with a specific freq count strat from outside
    # ensures that experiments are not executed multiple times
    if FREQ_COUNT_STRAT != FrequencyCountingStrategy.TraceTransaction:
        return
    log_dir = EVENT_LOG_DIRECTORY
    log_filename = LOG_FILE
    log = xes_import(os.path.join(log_dir, log_filename))

    n_traces = len(log)
    variants = get_concurrency_variants(log)
    n_variants = len(variants)
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(SAVE_DIR, "patterns")).mkdir(parents=True, exist_ok=True)

    # for counting_strategy in FrequencyCountingStrategy:
    #     support_count = get_support_count(REL_SUPPORT, counting_strategy, n_traces, n_variants)
    #     print('-------------- generate patterns with counting strategy', counting_strategy, '----------------')
    #     patterns = generate_eventually_follows_patterns_from_groups(variants, support_count, counting_strategy)
    #     flat_platterns = flatten_patterns(patterns)
    #     with open(os.path.join(SAVE_DIR, 'patterns', f'patterns_{counting_strategy}_{REL_SUPPORT}.pkl'),
    #               'wb') as handle:
    #         pickle.dump(flat_platterns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    generate_plots(
        log_filename,
        lambda rel_sup, strat: get_support_count(rel_sup, strat, n_traces, n_variants),
    )


def calculate_overlap(
    p_trans: Iterable[EventuallyFollowsPattern],
    p_occ: Iterable[EventuallyFollowsPattern],
) -> float:
    p_trans_str = {str(p) for p in p_trans}
    p_occ_str = {str(p) for p in p_occ}

    intersection = p_trans_str.intersection(p_occ_str)
    assert len(intersection) == len(p_trans_str)

    return len(intersection) / len(p_occ_str)


def generate_plots(log_filename, support_func):
    plot_dir = os.path.join(
        RESULT_DIRECTORY,
        os.path.splitext(log_filename)[0],
        "algorithm_comparison",
        "plots",
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    generate_overlap_plot(plot_dir, support_func)
    generate_shared_patterns_variant_trace_plot(
        plot_dir,
        support_func,
        FrequencyCountingStrategy.TraceTransaction,
        FrequencyCountingStrategy.VariantTransaction,
    )
    generate_shared_patterns_variant_trace_plot(
        plot_dir,
        support_func,
        FrequencyCountingStrategy.TraceOccurence,
        FrequencyCountingStrategy.VariantOccurence,
    )


def generate_shared_patterns_variant_trace_plot(
    plot_dir, support_func, trace_count_strat, variant_count_strat
):
    df = get_shared_patterns_df(
        support_func, trace_count_strat, variant_count_strat
    ).sort_values("Relative Support", ascending=False)
    hue_order = ["Trace-Based Support", "Variant-Based Support", "Shared Patterns"]
    plot = sns.lineplot(
        data=df,
        x="Relative Support",
        y="Number of Frequent Valid Patterns",
        hue="Strategy",
        style="Strategy",
        markers=True,
        lw=1,
        hue_order=hue_order,
    )
    plot.set(yscale="log")

    plt.xlabel("Relative Support")
    plt.ylabel("Number of Frequent Valid Patterns")
    plot.set_xlim(plot.get_xlim()[::-1])
    # call `draw` to re-render the graph
    plt.draw()
    filename = (
        "shared_patterns_trans.pdf"
        if trace_count_strat == FrequencyCountingStrategy.TraceTransaction
        else "shared_patterns_occ.pdf"
    )
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches="tight")
    plt.close()


def get_shared_patterns_df(
    support_func, trace_counting_strategy, variant_counting_strategy
):
    with open(
        os.path.join(
            SAVE_DIR,
            "patterns",
            f"patterns_{trace_counting_strategy}_{REL_SUPPORT}.pkl",
        ),
        "rb",
    ) as handle:
        patterns_trace = pickle.load(handle)
    with open(
        os.path.join(
            SAVE_DIR,
            "patterns",
            f"patterns_{variant_counting_strategy}_{REL_SUPPORT}.pkl",
        ),
        "rb",
    ) as handle:
        patterns_variant = pickle.load(handle)

    results = []

    for rel_support in REL_SUPPORTS:
        if rel_support < REL_SUPPORT:
            break

        trace_patterns = filter_patterns(
            patterns_trace, support_func(rel_support, trace_counting_strategy)
        )
        variant_patterns = filter_patterns(
            patterns_variant, support_func(rel_support, variant_counting_strategy)
        )

        results.append([rel_support, "Trace-Based Support", len(trace_patterns)])
        results.append([rel_support, "Variant-Based Support", len(variant_patterns)])
        results.append(
            [
                rel_support,
                "Shared Patterns",
                calculate_shared_patterns(trace_patterns, variant_patterns),
            ]
        )

    return pd.DataFrame(
        results,
        columns=["Relative Support", "Strategy", "Number of Frequent Valid Patterns"],
    )


def calculate_shared_patterns(
    p1: Iterable[EventuallyFollowsPattern], p2: Iterable[EventuallyFollowsPattern]
) -> float:
    p1_str = {str(p) for p in p1}
    p2_str = {str(p) for p in p2}

    intersection = p1_str.intersection(p2_str)

    return len(intersection)


def generate_overlap_plot(plot_dir, support_func):
    df = get_overlap_df(support_func).sort_values("Relative Support", ascending=False)
    hue_order = ["Trace-Based Support", "Variant-Based Support"]
    plot = sns.lineplot(
        data=df,
        x="Relative Support",
        y="Overlap",
        hue="Strategy",
        style="Strategy",
        markers=True,
        lw=1,
        hue_order=hue_order,
    )

    plt.xlabel("Relative Support")
    plt.ylabel("Overlap")
    plot.set_xlim(plot.get_xlim()[::-1])
    # call `draw` to re-render the graph
    plt.draw()
    plt.savefig(os.path.join(plot_dir, f"overlap_trans_occ.pdf"), bbox_inches="tight")
    plt.close()


def get_overlap_df(support_func) -> pd.DataFrame:
    patterns = dict()
    for support_strat in FrequencyCountingStrategy:
        with open(
            os.path.join(
                SAVE_DIR, "patterns", f"patterns_{support_strat}_{REL_SUPPORT}.pkl"
            ),
            "rb",
        ) as handle:
            patterns[support_strat] = pickle.load(handle)

    results = []

    for rel_support in REL_SUPPORTS:
        if rel_support < REL_SUPPORT:
            break

        trace_trans_patterns = filter_patterns(
            patterns[FrequencyCountingStrategy.TraceTransaction],
            support_func(rel_support, FrequencyCountingStrategy.TraceTransaction),
        )
        trace_occ_patterns = filter_patterns(
            patterns[FrequencyCountingStrategy.TraceOccurence],
            support_func(rel_support, FrequencyCountingStrategy.TraceOccurence),
        )
        variant_trans_patterns = filter_patterns(
            patterns[FrequencyCountingStrategy.VariantTransaction],
            support_func(rel_support, FrequencyCountingStrategy.VariantTransaction),
        )
        variant_occ_patterns = filter_patterns(
            patterns[FrequencyCountingStrategy.VariantOccurence],
            support_func(rel_support, FrequencyCountingStrategy.VariantOccurence),
        )

        results.append(
            [
                rel_support,
                "Trace-Based Support",
                calculate_overlap(trace_trans_patterns, trace_occ_patterns),
            ]
        )
        results.append(
            [
                rel_support,
                "Variant-Based Support",
                calculate_overlap(variant_trans_patterns, variant_occ_patterns),
            ]
        )

    return pd.DataFrame(results, columns=["Relative Support", "Strategy", "Overlap"])


def filter_patterns(patterns: Iterable[EventuallyFollowsPattern], support_count: int):
    return {p for p in patterns if p.support >= support_count}


if __name__ == "__main__":
    # generate_plots('ccc19.xes',
    #                'C:\\sources\\arbeit\\cortado\\master_thesis\\ccc19\\algorithm_comparison\\data\\TraceTransaction.csv')
    overlap_experiments()
