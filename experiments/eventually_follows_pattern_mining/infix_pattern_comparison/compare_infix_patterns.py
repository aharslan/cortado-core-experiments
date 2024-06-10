from typing import Dict, Set

import pandas as pd
import pm4py
import seaborn as sns
from matplotlib import pyplot as plt

from cortado_core.eventually_follows_pattern_mining.algorithm import (
    generate_eventually_follows_patterns_from_groups,
    Algorithm,
)
from cortado_core.eventually_follows_pattern_mining.obj import EventuallyFollowsPattern
from experiments.eventually_follows_pattern_mining.util import (
    get_support_count,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.utils.cvariants import get_concurrency_variants

REL_SUPPORTS = [0.95, 0.75, 0.55, 0.35, 0.3, 0.25, 0.2, 0.15]
LOG_FILE = "C:\\sources\\arbeit\\cortado\\event-logs\\BPI_Challenge_2012.xes"
PLOT_FILE = "C:\\sources\\arbeit\\cortado\\master_thesis\\plots\\BPI_Challenge_2012.png"


def compare_infix_patterns(
    log_filename: str, frequency_strategy: FrequencyCountingStrategy
):
    log = pm4py.read_xes(log_filename)
    n_traces = len(log)
    variants = get_concurrency_variants(log)
    n_variants = len(variants)
    result = {
        "support": [],
        "infix patterns": [],
        "ef patterns": [],
        "ef maximal patterns": [],
        "ef closed patterns": [],
    }

    for rel_support in REL_SUPPORTS:
        support_count = get_support_count(
            rel_support, frequency_strategy, n_traces, n_variants
        )
        ef_patterns = generate_eventually_follows_patterns_from_groups(
            variants, support_count, frequency_strategy, Algorithm.RightmostExpansion
        )
        infix_patterns = generate_eventually_follows_patterns_from_groups(
            variants,
            support_count,
            frequency_strategy,
            Algorithm.RightmostExpansionOnlyInfixPatterns,
        )
        closed, maximal = generate_eventually_follows_patterns_from_groups(
            variants, support_count, frequency_strategy, Algorithm.BlanketMining
        )
        result["infix patterns"].append(get_number_of_patterns(infix_patterns))
        result["ef patterns"].append(get_number_of_patterns(ef_patterns))
        result["ef maximal patterns"].append(len(maximal))
        result["ef closed patterns"].append(len(closed))
        result["support"].append(rel_support)

    df = pd.DataFrame.from_dict(result).melt(
        id_vars="support", var_name="pattern type", value_name="n_patterns"
    )
    print(df)
    df.to_csv("C:\\sources\\arbeit\\cortado\\master_thesis\\data.csv")

    plot = sns.lineplot(
        data=df,
        x="support",
        y="n_patterns",
        hue="pattern type",
        markers=True,
        sort=False,
        style="pattern type",
    )
    plot.set(yscale="log")
    plt.savefig(PLOT_FILE, bbox_inches="tight")


def get_number_of_patterns(patterns: Dict[int, Set[EventuallyFollowsPattern]]):
    return sum([len(p) for p in patterns.values()])


if __name__ == "__main__":
    compare_infix_patterns(LOG_FILE, FrequencyCountingStrategy.TraceTransaction)
