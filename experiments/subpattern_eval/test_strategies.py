import itertools
from statistics import mean, median
import pandas as pd
import timeit
from experiments.subpattern_eval.Algos.valid_2_pattern_performance import (
    min_sub_mining_memory_2_pattern,
)
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


def compare_strategy_valid(log, log_name, art_start=False):
    print("Loading Log...")
    start_time = timeit.default_timer()
    log = xes_import(log)
    load_time = timeit.default_timer() - start_time

    print("Load Time:", load_time)

    print("Creating Variants...")
    start_time = timeit.default_timer()
    variants = get_concurrency_variants(log, False)
    cutting_time = timeit.default_timer() - start_time

    print("Cutting Time:", cutting_time)

    print("Creating  Treebank...")
    start_time = timeit.default_timer()
    treeBank = create_treebank_from_cv_variants(variants, art_start)
    treeBank_time = timeit.default_timer() - start_time

    print("Treebank Creation Time:", treeBank_time)

    min_sups = [
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.05,
        0.0375,
        0.025,
        0.01,
    ]
    k_stops = [100]

    tTraces = sum([treeBank[tid].nTraces for tid in treeBank])
    tTrees = len(treeBank)

    df_dicts = []
    heat_map_dicts_trees = []
    heat_map_dicts_patterns = []
    print("Mining K Patterns")

    for k in k_stops:
        for rel_sup in min_sups:
            treePatternList = []
            patternList = []

            for strategy in FrequencyCountingStrategy:
                df_dict = {}

                if (
                    strategy == FrequencyCountingStrategy.TraceOccurence
                    or strategy == FrequencyCountingStrategy.TraceTransaction
                ):
                    abs_sup = round(tTraces * rel_sup)
                else:
                    abs_sup = round(tTrees * rel_sup)

                print()
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
                df_dict["treeBankTime"] = treeBank_time
                df_dict["cuttingTime"] = cutting_time
                df_dict["logLoadTime"] = load_time

                k_patterns, nC, _ = min_sub_mining_memory_2_pattern(
                    treeBank,
                    frequency_counting_strat=strategy,
                    k_it=k,
                    min_sup=abs_sup,
                    delete_rmo=False,
                )

                set_maximaly_closed_patterns(k_patterns)

                variantCoverage = []
                traceCoverage = []
                totalSup = 0

                treePatterns = set()
                pattern_set = set()

                nRes = 0
                nClosed = 0
                nMax = 0
                nValid = 0

                lRes = []
                lValid = []
                lClosed = []
                lMaximal = []

                for _, patterns in k_patterns.items():
                    iRes = len(patterns)
                    nRes += iRes
                    lRes.append(iRes)

                    iClosed = len(
                        [
                            pattern
                            for pattern in patterns
                            if pattern.closed and check_if_valid_tree(pattern.tree)
                        ]
                    )
                    nClosed += iClosed
                    lClosed.append(iClosed)

                    iMax = len(
                        [
                            pattern
                            for pattern in patterns
                            if pattern.maximal and check_if_valid_tree(pattern.tree)
                        ]
                    )
                    nMax += iMax
                    lMaximal.append(iMax)

                    iValid = len(
                        [
                            pattern
                            for pattern in patterns
                            if check_if_valid_tree(pattern.tree)
                        ]
                    )
                    nValid += iValid
                    lValid.append(iValid)

                    for pattern in patterns:
                        if check_if_valid_tree(pattern.tree):
                            variantsCovered = set(pattern.rmo.keys())
                            variantCoverage.append(len(variantsCovered) / tTrees)
                            traceCoverage.append(
                                sum([treeBank[tid].nTraces for tid in variantsCovered])
                                / tTraces
                            )
                            treePatterns.add(repr(pattern.tree))
                            pattern_set.add(str(pattern))

                    totalSup += sum(
                        [
                            pattern.support
                            for pattern in patterns
                            if check_if_valid_tree(pattern.tree)
                        ]
                    )

                df_dict["nRes"] = nRes
                df_dict["nValid"] = nValid
                df_dict["nClosed"] = nClosed
                df_dict["nMax"] = nMax
                df_dict["nCandidates"] = nC

                df_dict["lRes"] = lRes
                df_dict["lValid"] = lValid
                df_dict["lClosed"] = lClosed
                df_dict["lMax"] = lMaximal

                df_dict["medianVariantCoverage"] = median(variantCoverage)
                df_dict["medianTraceCoverage"] = median(traceCoverage)
                df_dict["meanVariantCoverage"] = mean(variantCoverage)
                df_dict["meanTraceCoverage"] = mean(traceCoverage)
                df_dict["totalSupport"] = totalSup

                print("Max K Valid", max(k_patterns.keys(), default=0)),
                print("nRes Valid", nRes)
                print("nC Valid:", nC)
                print("Average Variant Coverage", df_dict["meanVariantCoverage"])
                print("Average Trace Coverage", df_dict["meanTraceCoverage"])
                print("Median Variant Coverage", df_dict["medianVariantCoverage"])
                print("Median Trace Coverage", df_dict["medianTraceCoverage"])
                print("Total Support", df_dict["totalSupport"])
                print()

                df_dicts.append(df_dict)

                treePatternList.append((strategy, treePatterns))
                patternList.append((strategy, pattern_set))

            for (strategy_l, patterns_l), (
                strategy_r,
                patterns_r,
            ) in itertools.combinations(treePatternList, 2):
                heat_map_dict = {}

                heat_map_dict["l"] = strategy_l
                heat_map_dict["r"] = strategy_r
                heat_map_dict["l_size"] = len(patterns_l)
                heat_map_dict["r_size"] = len(patterns_r)
                heat_map_dict["l_diff_r"] = len(patterns_l.difference(patterns_r))
                heat_map_dict["r_diff_l"] = len(patterns_r.difference(patterns_l))
                heat_map_dict["r_sub_l"] = patterns_r.issubset(patterns_l)
                heat_map_dict["l_sub_r"] = patterns_l.issubset(patterns_l)
                heat_map_dict["k"] = k
                heat_map_dict["sup"] = rel_sup
                heat_map_dict["type"] = "trees"

                shared_patterns = len(patterns_l.intersection(patterns_r))
                heat_map_dict["nShared"] = shared_patterns
                heat_map_dict["Jaccard"] = shared_patterns / (
                    len(patterns_l) + len(patterns_r) - shared_patterns
                )

                print(
                    "Trees:",
                    "Strategy L:",
                    strategy_l,
                    "Strategy R:",
                    strategy_r,
                    "Similarity",
                    heat_map_dict["Jaccard"],
                )

                heat_map_dicts_trees.append(heat_map_dict)

            for (strategy_l, patterns_l), (
                strategy_r,
                patterns_r,
            ) in itertools.combinations(patternList, 2):
                heat_map_dict = {}

                heat_map_dict["l"] = strategy_l
                heat_map_dict["l_size"] = len(patterns_l)
                heat_map_dict["r_size"] = len(patterns_r)
                heat_map_dict["l_diff_r"] = len(patterns_l.difference(patterns_r))
                heat_map_dict["r_diff_l"] = len(patterns_r.difference(patterns_l))
                heat_map_dict["r_sub_l"] = patterns_r.issubset(patterns_l)
                heat_map_dict["l_sub_r"] = patterns_l.issubset(patterns_l)
                heat_map_dict["r"] = strategy_r
                heat_map_dict["k"] = k
                heat_map_dict["sup"] = rel_sup
                heat_map_dict["type"] = "patterns"

                shared_patterns = len(patterns_l.intersection(patterns_r))

                heat_map_dict["nShared"] = shared_patterns
                heat_map_dict["Jaccard"] = shared_patterns / (
                    len(patterns_l) + len(patterns_r) - shared_patterns
                )

                print(
                    "Patterns:",
                    "Strategy L:",
                    strategy_l,
                    "Strategy R:",
                    strategy_r,
                    "Similarity",
                    heat_map_dict["Jaccard"],
                )

                heat_map_dicts_patterns.append(heat_map_dict)

            print("Writing Results to File ... ")
            pd.DataFrame.from_dict(df_dicts).to_csv(
                ".\\cortado_core\\experiments\\subpattern_eval\\Eval_Res\\"
                + log_name
                + "_Pattern_Mining_Strategies_Performance.csv"
            )
            pd.DataFrame.from_dict(
                heat_map_dicts_trees + heat_map_dicts_patterns
            ).to_csv(
                ".\\cortado_core\\experiments\\subpattern_eval\\Eval_Res\\"
                + log_name
                + "_Strategy_Pairwise_Comparsion.csv"
            )
