from experiments.subpattern_eval.exit_after import run_mining_eval
from cortado_core.subprocess_discovery.subtree_mining.maximal_connected_components.maximal_connected_check import (
    check_if_valid_tree,
    set_maximaly_closed_patterns,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.subprocess_discovery.subtree_mining.treebank import (
    create_treebank_from_cv_variants,
)
import pandas as pd
import timeit
from experiments.subpattern_eval.Algos.cm_performance import (
    cm_min_sub_mining_performance,
)
from experiments.subpattern_eval.exit_after import run_mining_eval
from cortado_core.subprocess_discovery.subtree_mining.treebank import (
    create_treebank_from_cv_variants,
)
from cortado_core.utils.cvariants import get_concurrency_variants
from pm4py.objects.log.importer.xes.importer import apply as xes_import

from cortado_core.utils.timestamp_utils import TimeUnit

if __name__ == "__main__":
    log_path = ""
    log = log_path + "BPI Challenge 2017" + ".xes"

    print("Loading Log...")
    log = xes_import(log)

    timeout = 600
    strategy_name = "TraceTransaction"
    strategy = FrequencyCountingStrategy.TraceTransaction

    df_dicts = []
    reps = 2
    artStart = True

    print("Mining K Patterns")

    for time_gran in [
        TimeUnit.MS,
        TimeUnit.SEC,
        TimeUnit.MIN,
        TimeUnit.HOUR,
        TimeUnit.DAY,
    ]:
        print("Creating Variants...")
        variants = get_concurrency_variants(log, False, time_gran)

        print("Creating  Treebank...")
        treeBank = create_treebank_from_cv_variants(variants, artStart)

        rel_sup = 0.01
        k = 100

        tTraces = sum([treeBank[tid].nTraces for tid in treeBank])
        tTrees = len(treeBank)

        if (
            strategy == FrequencyCountingStrategy.TraceOccurence
            or strategy == FrequencyCountingStrategy.TraceTransaction
        ):
            abs_sup = round(tTraces * rel_sup)
        else:
            abs_sup = round(tTrees * rel_sup)

        print("Mining...")

        rTimes, k_patterns, nC, bailOut = run_mining_eval(
            cm_min_sub_mining_performance,
            timeout,
            prms={
                "frequency_counting_strat": strategy,
                "treebank": treeBank,
                "k_it": k,
                "min_sup": abs_sup,
            },
            repeats=1,
        )

        treePatterns = set()
        pattern_set = set()

        nRes = 0
        nClosed = 0
        nMax = 0
        nValid = 0

        if not bailOut:
            set_maximaly_closed_patterns(k_patterns)

            for _, patterns in k_patterns.items():
                nRes += len(patterns)
                nClosed += len(
                    [
                        pattern
                        for pattern in patterns
                        if pattern.closed and check_if_valid_tree(pattern.tree)
                    ]
                )
                nMax += len(
                    [
                        pattern
                        for pattern in patterns
                        if pattern.maximal and check_if_valid_tree(pattern.tree)
                    ]
                )
                nValid += len(
                    [
                        pattern
                        for pattern in patterns
                        if check_if_valid_tree(pattern.tree)
                    ]
                )

                for pattern in patterns:
                    if check_if_valid_tree(pattern.tree):
                        treePatterns.add(repr(pattern.tree))
                        pattern_set.add(str(pattern))
