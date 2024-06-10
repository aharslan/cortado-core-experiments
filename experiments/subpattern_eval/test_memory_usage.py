import pandas as pd
import timeit
from experiments.subpattern_eval.Algos.asai_performance import (
    min_sub_mining_asai,
)
from experiments.subpattern_eval.Algos.cm_performance import (
    cm_min_sub_mining_performance,
)
from experiments.subpattern_eval.Algos.valid_2_pattern_performance import (
    min_sub_mining_performance_2_pattern,
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
import memory_profiler as mp
import gc


def compare_memory(log, log_name, strategy, strategy_name, artStart):
    pre_load_mem = mp.memory_usage(max_usage=True)

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
    treeBank = create_treebank_from_cv_variants(variants, True)
    treeBank_time = timeit.default_timer() - start_time

    print("Treebank Creation Time:", treeBank_time)

    min_sups = [0.3, 0.2, 0.1, 0.05, 0.01]
    k_stops = [50]

    tTraces = sum([len(variants[variant]) for variant in variants])
    tTrees = len(variants)
    df_dicts = []

    post_init_mem = mp.memory_usage(max_usage=True)

    initial_mem = post_init_mem - pre_load_mem

    print("Inital Mem Usage", initial_mem)

    print("Mining K Patterns")

    for k in k_stops:
        for rel_sup in min_sups:
            df_dict = {}

            if (
                strategy == FrequencyCountingStrategy.TraceOccurence
                or strategy == FrequencyCountingStrategy.TraceTransaction
            ):
                abs_sup = round(tTraces * rel_sup)
            else:
                abs_sup = round(tTrees * rel_sup)

            reps = 2

            print()
            print("Current Sup Level:", rel_sup, "K_max:", k)
            print("Abs Sup", abs_sup)
            print()

            gc.collect()

            memProfile, k_patterns, nC = run_mining_memory_eval(
                min_sub_mining_performance_2_pattern,
                prms={
                    "artifical_start": artStart,
                    "frequency_counting_strat": strategy,
                    "variants": variants,
                    "treebank": treeBank,
                    "k_it": k,
                    "min_sup": abs_sup,
                },
                repeats=reps,
            )

            nRes = 0
            nValid = 0

            set_maximaly_closed_patterns(k_patterns)

            for _, patterns in k_patterns.items():
                nRes += len(patterns)
                nValid += len(
                    [
                        pattern
                        for pattern in patterns
                        if check_if_valid_tree(pattern.tree)
                    ]
                )

            df_dict["nResValid"] = nRes
            df_dict["nValidValid"] = nValid
            df_dict["nCandidatesValid"] = nC

            print("memProfile Valid", memProfile)
            for key, value in memProfile.items():
                df_dict[key + "_Valid"] = value

            print("nRes Valid", nRes)
            print("nC Valid:", nC)
            print("nValid Valid:", nValid)
            print()

            del k_patterns

            gc.collect()

            memProfile, k_patterns, nC = run_mining_memory_eval(
                min_sub_mining_asai,
                prms={
                    "artifical_start": artStart,
                    "frequency_counting_strat": strategy,
                    "variants": variants,
                    "treebank": treeBank,
                    "k_it": k,
                    "min_sup": abs_sup,
                    "only_valid": False,
                    "no_pruning": True,
                },
                repeats=reps,
            )

            set_maximaly_closed_patterns(k_patterns)

            nRes = 0
            nValid = 0
            for _, patterns in k_patterns.items():
                nRes += len(patterns)
                nValid += len(
                    [
                        pattern
                        for pattern in patterns
                        if check_if_valid_tree(pattern.tree)
                    ]
                )

            df_dict["nResNoPruning"] = nRes
            df_dict["nValidNoPruning"] = nValid
            df_dict["nCandidatesNoPruning"] = nC

            print("memProfile No Pruning", memProfile)
            for key, value in memProfile.items():
                df_dict[key + "_NoPruning"] = value

            print("nRes NoPruning", nRes)
            print("nC NoPruning:", nC)
            print("nValid NoPruning:", nValid)
            print()

            del k_patterns

            gc.collect()

            memProfile, k_patterns, nC = run_mining_memory_eval(
                min_sub_mining_asai,
                prms={
                    "artifical_start": artStart,
                    "frequency_counting_strat": strategy,
                    "variants": variants,
                    "treebank": treeBank,
                    "k_it": k,
                    "min_sup": abs_sup,
                    "only_valid": False,
                    "no_pruning": False,
                },
                repeats=reps,
            )

            set_maximaly_closed_patterns(k_patterns)

            nRes = 0
            nValid = 0
            for _, patterns in k_patterns.items():
                nRes += len(patterns)
                nValid += len(
                    [
                        pattern
                        for pattern in patterns
                        if check_if_valid_tree(pattern.tree)
                    ]
                )

            df_dict["nResAsai"] = nRes
            df_dict["nValidAsai"] = nValid
            df_dict["nCandidatesAsai"] = nC

            print("memProfile Asai", memProfile)
            for key, value in memProfile.items():
                df_dict[key + "_Asai"] = value

            print("nRes Asai:", nRes)
            print("nC Asai:", nC)
            print("nValid Asai:", nValid)
            print()

            del k_patterns

            gc.collect()

            memProfile, k_patterns, nC = run_mining_memory_eval(
                cm_min_sub_mining_performance,
                prms={
                    "artifical_start": artStart,
                    "frequency_counting_strat": strategy,
                    "variants": variants,
                    "treebank": treeBank,
                    "k_it": k,
                    "min_sup": abs_sup,
                },
                repeats=reps,
            )

            set_maximaly_closed_patterns(k_patterns)

            nRes = 0
            nValid = 0
            for _, patterns in k_patterns.items():
                nRes += len(patterns)
                nValid += len(
                    [
                        pattern
                        for pattern in patterns
                        if check_if_valid_tree(pattern.tree)
                    ]
                )

            df_dict["nResCM"] = nRes
            df_dict["nValidCM"] = nValid
            df_dict["nCandidatesCM"] = nC

            print("memProfile CM", memProfile)
            for key, value in memProfile.items():
                df_dict[key + "_CM"] = value

            print("nRes CM:", nRes)
            print("nC CM:", nC)
            print("nValid CM:", nValid)
            print()

            df_dict["k_max"] = k
            df_dict["rel_sup"] = rel_sup
            df_dict["abs_sup"] = abs_sup
            df_dict["tTraces"] = tTraces
            df_dict["tTrees"] = tTrees
            df_dict["log_mem_usage"] = initial_mem

            df_dicts.append(df_dict)

            del k_patterns

            gc.collect()

            print("Writing Results to File ... ")
            pd.DataFrame.from_dict(df_dicts).to_csv(
                ".\\cortado_core\\experiments\\subpattern_eval\\Eval_Res\\"
                + log_name
                + "_"
                + strategy_name
                + "_Pattern_Mining_MemoryUsage.csv"
            )
            print()

            gc.collect()


def run_mining_memory_eval(fnc, prms, repeats):
    gc.collect()

    @exit_if_memory_usage_above(5e9)
    def run_algo(fnc, prms):
        return fnc(**prms)

    max_runtime_mem = []
    memory_profile = {}
    k_patterns = None
    start_mem = mp.memory_usage(max_usage=True)

    try:
        max_mem, (k_patterns, nC) = mp.memory_usage(
            (run_algo, [fnc, prms]), interval=0.001, max_usage=True, retval=True
        )
        max_runtime_mem.append(max_mem - start_mem)

    except Exception as e:
        print("Exception caught during Mining", e)

        k_patterns = {}
        nC = 0
        max_runtime_mem = [5e9]
        gc.collect()

    except KeyboardInterrupt as k:
        k_patterns = {}
        nC = 0
        max_runtime_mem = [5e9]
        gc.collect()

    memory_profile["max_runtime_mem"] = max(max_runtime_mem)

    return memory_profile, k_patterns, nC


import os
import psutil
import threading
import sys
import threading
import timeit
import _thread


def quit_function():
    sys.stderr.flush()
    _thread.interrupt_main()  # raises KeyboardInterrupt


def checkMemory(mem, stop):
    process = psutil.Process(os.getpid())

    while True:
        memoryUsage = process.memory_info().rss  # in bytes
        if memoryUsage > mem:
            quit_function()
            break

        if stop():
            break


def exit_if_memory_usage_above(mem):
    """
    use as decorator to exit process if
    function takes longer than s seconds
    """

    def outer(fn):
        def inner(*args, **kwargs):
            stop_threads = False

            mem_watcher = threading.Thread(
                target=checkMemory, args=(mem, lambda: stop_threads)
            )
            mem_watcher.start()

            try:
                result = fn(*args, **kwargs)
            finally:
                stop_threads = True
                mem_watcher.join()

            return result

        return inner

    return outer


if __name__ == "__main__":
    files = [
        "Sepsis Cases - Event Log",
        "BPI_CH_2020_PrepaidTravelCost",
        "RoadTrafficFineManagement",
        "BPI Challenge 2017",
        "receipt",
        "BPI_Challenge_2012",
    ]
    # files = ['reduced1000_BPI_2017']

    for file in files:
        log_path = ""
        log = log_path + file + ".xes"

        log_name = file
        strategy_name = "TraceTransaction"
        strategy = FrequencyCountingStrategy.TraceTransaction

        compare_memory(log, log_name, strategy, strategy_name, False)
