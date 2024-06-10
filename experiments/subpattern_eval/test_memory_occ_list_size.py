import pandas as pd
import timeit
from experiments.subpattern_eval.Algos.asai_performance import (
    min_sub_mining_asai_memory,
)
from experiments.subpattern_eval.Algos.valid_performance import (
    min_sub_mining_memory,
)
from experiments.subpattern_eval.exit_after import run_mining_memory_eval

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
from cortado_core.subprocess_discovery.subtree_mining.utilities import (
    compute_occurence_list_size,
)
from cortado_core.utils.cvariants import get_concurrency_variants
from pm4py.objects.log.importer.xes.importer import apply as xes_import


def compare_occurence_list_size(
    log, log_name, strategy, strategy_name, timeout, artStart
):
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
    treeBank = create_treebank_from_cv_variants(variants, artStart)
    treeBank_time = timeit.default_timer() - start_time

    print("Treebank Creation Time:", treeBank_time)

    min_sups = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0375, 0.025, 0.01]
    tTraces = sum([len(variants[variant]) for variant in variants])
    tTrees = len(variants)
    df_dicts = []

    print("Mining K Patterns")

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
        print("Current Sup Level:", rel_sup, "K_max:", 100)
        print("Abs Sup", abs_sup)
        print()

        df_dict["k_max"] = 100
        df_dict["rel_sup"] = rel_sup
        df_dict["abs_sup"] = abs_sup
        df_dict["tTraces"] = tTraces
        df_dict["tTrees"] = tTrees

        df_dict["treeBankTime"] = treeBank_time
        df_dict["cuttingTime"] = cutting_time
        df_dict["logLoadTime"] = load_time

        performanceTest = [
            (
                min_sub_mining_memory,
                {
                    "frequency_counting_strat": strategy,
                    "treebank": treeBank,
                    "k_it": 100,
                    "min_sup": abs_sup,
                    "bfs_traversal": True,
                },
                "Valid (BFS)",
            ),
            (
                min_sub_mining_asai_memory,
                {
                    "frequency_counting_strat": strategy,
                    "treebank": treeBank,
                    "k_it": 100,
                    "min_sup": abs_sup,
                    "no_pruning": False,
                },
                "Asai",
            ),
        ]

        for algo, prms, name in performanceTest:
            if not name in bailouted:
                rTimes, k_patterns, nC, bailOut = run_mining_memory_eval(
                    algo, timeout, prms=prms
                )

            else:
                bailouted[name] = True
                nC = 0
                k_patterns = {}
                rTimes = []
                bailOut = True

            if bailOut:
                bailouted[name] = True

            nRes = 0
            nClosed = 0
            nMax = 0
            nValid = 0

            lClosed = []
            lValid = []
            lMax = []
            lRes = []

            occ_lists = []

            for k, patterns in k_patterns.items():
                try:
                    occ_lists.append(
                        sum([compute_occurence_list_size(f.rmo) for f in patterns])
                    )

                except:
                    for f in patterns:
                        print(f)
                        print(f.rmo)

            start_time = timeit.default_timer()
            set_maximaly_closed_patterns(k_patterns)

            df_dict["rSetClosed" + name] = timeit.default_timer() - start_time
            print("Post Pruning Time:", df_dict["rSetClosed" + name])

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

                lClosed.append(
                    len(
                        [
                            pattern
                            for pattern in patterns
                            if pattern.closed and check_if_valid_tree(pattern.tree)
                        ]
                    )
                )
                lValid.append(
                    len(
                        [
                            pattern
                            for pattern in patterns
                            if check_if_valid_tree(pattern.tree)
                        ]
                    )
                )
                lMax.append(
                    len(
                        [
                            pattern
                            for pattern in patterns
                            if pattern.maximal and check_if_valid_tree(pattern.tree)
                        ]
                    )
                )

                lRes.append(len(patterns))

            df_dict["rTime" + name] = min(rTimes, default=0)
            df_dict["rTimes" + name] = rTimes
            df_dict["nRes" + name] = nRes
            df_dict["nValid" + name] = nValid
            df_dict["nClosed" + name] = nClosed
            df_dict["nMax" + name] = nMax

            df_dict["lRes" + name] = lRes
            df_dict["lValid" + name] = lValid
            df_dict["lClosed" + name] = lClosed
            df_dict["lMax" + name] = lMax

            df_dict["nCandidates" + name] = nC
            df_dict["bailOut" + name] = bailOut
            df_dict["maxOccurenceList" + name] = max(occ_lists, default=0)
            df_dict["OccurenceList" + name] = occ_lists
            df_dict["TotalList" + name] = sum(occ_lists)

            print("Max K " + name, max(k_patterns.keys(), default=0)),
            print("Valid " + name, nValid)
            print("Closed " + name, nClosed)
            print("Maximal " + name, nMax)
            print("nRes " + name, nRes)
            print("nC " + name, nC)
            print("Runtime " + name, min(rTimes, default=0))
            print("Bailout" + name, bailOut)
            print(
                "OccList" + name, max(occ_lists, default=0), sum(occ_lists), occ_lists
            )
            print()

        df_dicts.append(df_dict)

        print("Writing Results to File ... ")
        pd.DataFrame.from_dict(df_dicts).to_csv(
            ".\\cortado_core\\experiments\\subpattern_eval\\Eval_Res\\"
            + log_name
            + "_"
            + strategy_name
            + "_Pattern_Mining_Memory.csv"
        )


if __name__ == "__main__":
    for strat, strat_name in [
        (FrequencyCountingStrategy.TraceOccurence, "TraceOccurence"),
        (FrequencyCountingStrategy.VariantOccurence, "VariantOccurence"),
    ]:
        files = ["Sepsis Cases - Event Log"]

        for file in files:
            log_path = ""
            log = log_path + file + ".xes"
            log_name = file

            timeout = 300  # 5 Minute Timeout

            compare_occurence_list_size(
                log, log_name, strat, strat_name, timeout, False
            )
