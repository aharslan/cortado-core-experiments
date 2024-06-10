import pandas as pd
import timeit
from experiments.subpattern_eval.Algos.cm_performance import (
    cm_min_sub_mining_performance,
)
from experiments.subpattern_eval.Algos.valid_2_pattern_performance import (
    min_sub_mining_performance_2_pattern,
)
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
from cortado_core.utils.cvariants import get_concurrency_variants
from pm4py.objects.log.importer.xes.importer import apply as xes_import

from cortado_core.utils.timestamp_utils import TimeUnit


def compare_time_granularity(
    log, log_name, strategy, strategy_name, timeout, artStart=False
):
    print("Loading Log...")
    start_time = timeit.default_timer()
    log = xes_import(log)
    load_time = timeit.default_timer() - start_time

    print("Load Time:", load_time)

    df_dicts = []
    reps = 2

    print("Mining K Patterns")

    for time_gran in [
        TimeUnit.MS,
        TimeUnit.SEC,
        TimeUnit.MIN,
        TimeUnit.HOUR,
        TimeUnit.DAY,
    ]:
        print("Creating Variants...")
        start_time = timeit.default_timer()
        variants = get_concurrency_variants(log, False, time_gran)
        cutting_time = timeit.default_timer() - start_time

        print("Cutting Time:", cutting_time)

        print("Creating  Treebank...")
        start_time = timeit.default_timer()
        treeBank = create_treebank_from_cv_variants(variants, artStart)
        treeBank_time = timeit.default_timer() - start_time

        print("Treebank Creation Time:", treeBank_time)

        min_sups = [0.5, 0.35, 0.2, 0.15, 0.1, 0.05, 0.0375, 0.025, 0.0175, 0.01]
        k_stops = [100]

        tTraces = sum([treeBank[tid].nTraces for tid in treeBank])
        tTrees = len(treeBank)

        for k in k_stops:
            bailouted = {}

            for rel_sup in min_sups:
                df_dict = {}

                if (
                    strategy == FrequencyCountingStrategy.TraceOccurence
                    or strategy == FrequencyCountingStrategy.TraceTransaction
                ):
                    abs_sup = round(tTraces * rel_sup)
                else:
                    abs_sup = round(tTrees * rel_sup)

                print()
                print("Current Sup Level:", rel_sup, "K_max:", k)
                print("Abs Sup", abs_sup)
                print("Current Strat", strategy)
                print()

                df_dict["k_max"] = k
                df_dict["rel_sup"] = rel_sup
                df_dict["abs_sup"] = abs_sup
                df_dict["tTraces"] = tTraces
                df_dict["tTrees"] = tTrees
                df_dict["strat"] = strategy
                df_dict["time_granularity"] = time_gran
                df_dict["treeBankTime"] = treeBank_time
                df_dict["cuttingTime"] = cutting_time
                df_dict["logLoadTime"] = load_time

                performanceTest = [
                    (
                        min_sub_mining_performance_2_pattern,
                        {
                            "frequency_counting_strat": strategy,
                            "treebank": treeBank,
                            "k_it": k,
                            "min_sup": abs_sup,
                        },
                        "Valid",
                    ),
                    (
                        cm_min_sub_mining_performance,
                        {
                            "frequency_counting_strat": strategy,
                            "treebank": treeBank,
                            "k_it": k,
                            "min_sup": abs_sup,
                        },
                        "Blanket",
                    ),
                ]

                for algo, prms, name in performanceTest:
                    if not name in bailouted:
                        rTimes, k_patterns, nC, bailOut = run_mining_eval(
                            algo, timeout, prms=prms, repeats=reps
                        )

                    else:
                        bailouted[name] = True
                        nC = 0
                        k_patterns = {}
                        rTimes = []
                        bailOut = True

                    if bailOut:
                        bailouted[name] = True

                    start_time = timeit.default_timer()
                    set_maximaly_closed_patterns(k_patterns)

                    df_dict["rSetClosed" + name] = timeit.default_timer() - start_time
                    print("Post Pruning Time:", df_dict["rSetClosed" + name])

                    treePatterns = set()
                    pattern_set = set()

                    nRes = 0
                    nClosed = 0
                    nMax = 0
                    nValid = 0

                    treePatterns = set()
                    pattern_set = set()

                    closed_treePatterns = set()
                    closed_pattern_set = set()

                    maximal_treePatterns = set()
                    maximal_pattern_set = set()

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
                                    if pattern.closed
                                    and check_if_valid_tree(pattern.tree)
                                ]
                            )
                            nMax += len(
                                [
                                    pattern
                                    for pattern in patterns
                                    if pattern.maximal
                                    and check_if_valid_tree(pattern.tree)
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

                                    if pattern.closed:
                                        closed_treePatterns.add(repr(pattern.tree))
                                        closed_pattern_set.add(str(pattern))

                                        if pattern.maximal:
                                            maximal_treePatterns.add(repr(pattern.tree))
                                            maximal_pattern_set.add(str(pattern))

                    df_dict["rTime" + name] = min(rTimes, default=0)
                    df_dict["rTimes" + name] = rTimes
                    df_dict["nRes" + name] = nRes
                    df_dict["nValid" + name] = nValid
                    df_dict["nClosed" + name] = nClosed
                    df_dict["nMax" + name] = nMax
                    df_dict["nCandidates" + name] = nC
                    df_dict["bailOut" + name] = bailOut

                    if name == "Valid":  # Don't duplicate the Data
                        df_dict["tree_list Valid"] = list(treePatterns)
                        df_dict["patter_list Valid"] = list(pattern_set)

                        df_dict["tree_list Closed"] = list(closed_treePatterns)
                        df_dict["patter_list Closed"] = list(closed_pattern_set)

                        df_dict["tree_list Maximal"] = list(maximal_treePatterns)
                        df_dict["patter_list Maximal"] = list(maximal_pattern_set)

                    print("Max K" + name, max(k_patterns.keys(), default=0)),
                    print("Closed " + name, nClosed)
                    print("Maximal " + name, nMax)
                    print("nRes " + name, nRes)
                    print("nC " + name, nC)
                    print("Runtime " + name, min(rTimes, default=0))
                    print("Bailout" + name, bailOut)
                    print()

                df_dicts.append(df_dict)

            print("Writing Results to File ... ")
            pd.DataFrame.from_dict(df_dicts).to_csv(
                ".\\cortado_core\\experiments\\subpattern_eval\\Eval_Res\\"
                + log_name
                + "_"
                + strategy_name
                + "_Pattern_Mining_Timegranularities_Performance.csv"
            )


if __name__ == "__main__":
    files = ["Sepsis Cases - Event Log"]

    for file in files:
        log_path = ""
        log = log_path + file + ".xes"

        log_name = file
        strategy_name = "VariantTransaction"
        strategy = FrequencyCountingStrategy.VariantTransaction

        timeout = 600  # 5 Minute Timeout
        compare_time_granularity(log, log_name, strategy, strategy_name, timeout, False)
