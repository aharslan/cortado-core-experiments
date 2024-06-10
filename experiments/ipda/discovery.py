from collections import Counter
from copy import deepcopy
from enum import Enum
from multiprocessing import Pool
import sys
from time import sleep
import time
import os
import random
from pm4py.objects.petri_net.exporter.exporter import serialize as serialize_pn
from experiments.ipda.evaluation import evaluate_model

from cortado_core.lca_approach import add_trace_to_pt_language

import pickle
import pm4py
from tqdm import tqdm

from pm4py.objects.log.obj import Event, Trace, EventLog
from pm4py.read import read_xes
from pm4py.convert import convert_to_petri_net
from pm4py.discovery import discover_process_tree_inductive
from pm4py.filtering import filter_variants_top_k, filter_variants
from pm4py.stats import get_variants_as_tuples, get_variants
from pm4py.vis import view_process_tree

from math import ceil
from cortado_core.model_repair.algorithm import repair_petri_net_with_log
from cortado_core.naive_approach import ensure_trace_replayable_on_process_tree

from cortado_core.process_tree_utils.reduction import apply_reduction_rules
from cortado_core.utils import process_tree

CONFIG = {"TIMEOUT": 7 * 24 * 60 * 60}

sys.setrecursionlimit(3000)


class InitialModelMethod(Enum):
    TOP_1 = 0
    TOP_1_PERCENT = 1
    TOP_2_PERCENT = 2
    TOP_5_PERCENT = 3
    TOP_10_PERCENT = 4


class Approach(Enum):
    INCREMENTAL_LCA_TRUE = 0
    INCREMENTAL_LCA_FALSE = 1
    INDUCTIVE_MINER = 2
    MODEL_REPAIR = 3
    NAIVE = 4


class OrderingStrategy(Enum):
    FREQUENCY_SORTED = 0
    SHUFFLED = 1


def run_approach_evaluation(
    log,
    log_name,
    approach,
    initial_method=None,
    ordering_strategy=OrderingStrategy.FREQUENCY_SORTED,
):
    match approach:
        case Approach.INCREMENTAL_LCA_TRUE:
            incremental_approach(log, log_name, True, initial_method, ordering_strategy)
        case Approach.INCREMENTAL_LCA_FALSE:
            incremental_approach(
                log, log_name, False, initial_method, ordering_strategy
            )
        case Approach.INDUCTIVE_MINER:
            inductive_miner_approach(log, log_name, ordering_strategy)
        case Approach.MODEL_REPAIR:
            model_repair_approach(log, log_name, initial_method, ordering_strategy)
        case Approach.NAIVE:
            naive_approach(log, log_name, initial_method, ordering_strategy)
    print("Experiments done.")


def incremental_approach(log, log_name, lca, initial_method, ordering_strategy):
    initial_log, remaining_log = split_initial_log(log, initial_method)
    initial_log = EventLog(
        list(map(lambda x: create_trace_for_variant(x), get_variants(initial_log)))
    )
    input_tree = discover_process_tree_inductive(initial_log)
    input_pn, input_im, input_fm = convert_to_petri_net(input_tree)
    remaining_variants = get_ordered_variants(remaining_log, ordering_strategy)
    results = []
    start_time = time.time()

    with Pool() as multiprocessing_pool:
        for variant, frequency in tqdm(
            remaining_variants, desc="Incrementally adding variants", leave=True
        ):
            if time.time() - start_time > CONFIG["TIMEOUT"]:
                raise TimeoutError()
            trace_to_be_added = create_trace_for_variant(variant)
            iteration_start_time = time.time()
            output_tree = add_trace_to_pt_language(
                pt=input_tree,
                log=initial_log,
                trace=trace_to_be_added,
                try_pulling_lca_down=lca,
                pool=multiprocessing_pool,
            )
            iteration_duration = time.time() - iteration_start_time
            apply_reduction_rules(output_tree)
            output_pn, output_im, output_fm = convert_to_petri_net(output_tree)

            fitness, precision = evaluate_model(output_pn, output_im, output_fm, log)

            iteration_dict = {}
            iteration_dict["input_model"] = serialize_pn(input_pn, input_im, input_fm)
            iteration_dict["added_variant"] = variant
            iteration_dict["added_variant_frequency"] = frequency
            iteration_dict["output_model"] = serialize_pn(
                output_pn, output_im, output_fm
            )
            iteration_dict["input_tree"] = deepcopy(input_tree)
            iteration_dict["output_tree"] = deepcopy(output_tree)
            iteration_dict["duration"] = iteration_duration
            iteration_dict["fitness"] = fitness
            iteration_dict["precision"] = precision
            iteration_dict["f-measure"] = (
                2 * fitness * precision / (fitness + precision)
            )  # harmonic mean

            results.append(iteration_dict)
            save_experiment(
                f"{log_name}__incremental_lca_{lca}__{initial_method}__preliminary",
                results,
            )

            initial_log.append(trace_to_be_added)
            input_tree = output_tree
            input_pn, input_im, input_fm = output_pn, output_im, output_fm
    save_experiment(
        f"{log_name}__incremental_lca_{lca}__{initial_method}__final", results, False
    )
    return results


def inductive_miner_approach(log, log_name, ordering_strategy):
    variants = get_ordered_variants(log, ordering_strategy)
    discovery_log = EventLog()
    results = []
    start_time = time.time()
    for variant, frequency in tqdm(
        variants, desc="Discovering models for growing event-log", leave=True
    ):
        if time.time() - start_time > CONFIG["TIMEOUT"]:
            raise TimeoutError()

        discovery_log.append(create_trace_for_variant(variant))
        iteration_start_time = time.time()
        model = discover_process_tree_inductive(discovery_log, multi_processing=True)
        iteration_duration = time.time() - iteration_start_time

        apply_reduction_rules(model)
        pn, im, fm = convert_to_petri_net(model)

        fitness, precision = evaluate_model(pn, im, fm, log)

        iteration_dict = {}
        iteration_dict["input_model"] = None
        iteration_dict["added_variant"] = variant
        iteration_dict["added_variant_frequency"] = frequency
        iteration_dict["output_model"] = serialize_pn(pn, im, fm)
        iteration_dict["input_tree"] = None
        iteration_dict["output_tree"] = model
        iteration_dict["duration"] = iteration_duration
        iteration_dict["fitness"] = fitness
        iteration_dict["precision"] = precision
        iteration_dict["f-measure"] = (
            2 * fitness * precision / (fitness + precision)
        )  # harmonic mean

        results.append(iteration_dict)
        save_experiment(f"{log_name}__inductive_miner____preliminary", results)

    save_experiment(f"{log_name}__inductive_miner____final", results, False)
    return results


def model_repair_approach(log, log_name, initial_method, ordering_strategy):
    initial_log, remaining_log = split_initial_log(log, initial_method)
    input_tree = discover_process_tree_inductive(initial_log)
    pn, im, fm = convert_to_petri_net(input_tree)
    variants = get_ordered_variants(remaining_log, ordering_strategy)
    discovery_log = EventLog()
    results = []
    start_time = time.time()

    with Pool() as multiprocessing_pool:
        for variant, frequency in tqdm(
            variants, desc="Incrementally adding variants", leave=True
        ):
            if time.time() - start_time > CONFIG["TIMEOUT"]:
                raise TimeoutError()
            discovery_log.append(create_trace_for_variant(variant))

            iteration_dict = {}
            iteration_dict["input_model"] = serialize_pn(pn, im, fm)

            iteration_start_time = time.time()
            repair_petri_net_with_log(
                pn, im, fm, discovery_log, pool=multiprocessing_pool
            )
            iteration_duration = time.time() - iteration_start_time

            fitness, precision = evaluate_model(pn, im, fm, log)

            iteration_dict["added_variant"] = variant
            iteration_dict["added_variant_frequency"] = frequency
            iteration_dict["output_model"] = serialize_pn(pn, im, fm)
            iteration_dict["input_tree"] = None
            iteration_dict["output_tree"] = None
            iteration_dict["duration"] = iteration_duration
            iteration_dict["fitness"] = fitness
            iteration_dict["precision"] = precision
            iteration_dict["f-measure"] = (
                2 * fitness * precision / (fitness + precision)
            )  # harmonic mean

            results.append(iteration_dict)
            save_experiment(
                f"{log_name}__model_repair__{initial_method}__preliminary",
                results,
            )

    save_experiment(
        f"{log_name}__model_repair__{initial_method}__final", results, False
    )
    return results


def naive_approach(log, log_name, initial_method, ordering_strategy):
    initial_log, remaining_log = split_initial_log(log, initial_method)
    input_tree = discover_process_tree_inductive(initial_log)
    input_pn, input_im, input_fm = convert_to_petri_net(input_tree)
    variants = get_ordered_variants(remaining_log, ordering_strategy)
    results = []
    start_time = time.time()

    for variant, frequency in tqdm(
        variants, desc="Incrementally adding variants", leave=True
    ):
        if time.time() - start_time > CONFIG["TIMEOUT"]:
            raise TimeoutError()
        trace_to_be_added = create_trace_for_variant(variant)
        iteration_start_time = time.time()
        output_tree = ensure_trace_replayable_on_process_tree(
            trace_to_be_added, input_tree
        )
        iteration_duration = time.time() - iteration_start_time
        apply_reduction_rules(output_tree)
        output_pn, output_im, output_fm = convert_to_petri_net(output_tree)

        fitness, precision = evaluate_model(output_pn, output_im, output_fm, log)

        iteration_dict = {}
        iteration_dict["input_model"] = serialize_pn(input_pn, input_im, input_fm)
        iteration_dict["added_variant"] = variant
        iteration_dict["added_variant_frequency"] = frequency
        iteration_dict["output_model"] = serialize_pn(output_pn, output_im, output_fm)
        iteration_dict["input_tree"] = deepcopy(input_tree)
        iteration_dict["output_tree"] = deepcopy(output_tree)
        iteration_dict["duration"] = iteration_duration
        iteration_dict["fitness"] = fitness
        iteration_dict["precision"] = precision
        iteration_dict["f-measure"] = (
            2 * fitness * precision / (fitness + precision)
        )  # harmonic mean

        results.append(iteration_dict)
        save_experiment(
            f"{log_name}__naive__{initial_method}__preliminary",
            results,
        )

        input_tree = output_tree
        input_pn, input_im, input_fm = output_pn, output_im, output_fm
    save_experiment(f"{log_name}__naive__{initial_method}__final", results, False)
    return results


def split_initial_log(log, initial_method):
    log_for_initial_model = None
    match initial_method:
        case InitialModelMethod.TOP_1:
            log_for_initial_model = filter_variants_top_k(log, 1)
        case InitialModelMethod.TOP_1_PERCENT:
            log_for_initial_model = get_top_variants_for_threshold(log, 0.01)
        case InitialModelMethod.TOP_2_PERCENT:
            log_for_initial_model = get_top_variants_for_threshold(log, 0.02)
        case InitialModelMethod.TOP_5_PERCENT:
            log_for_initial_model = get_top_variants_for_threshold(log, 0.05)
        case InitialModelMethod.TOP_10_PERCENT:
            log_for_initial_model = get_top_variants_for_threshold(log, 0.1)

    initial_variants = get_variants(log_for_initial_model)
    remaining_log = filter_variants(log, initial_variants, retain=False)
    log_for_initial_model = filter_variants(log, initial_variants, retain=True)
    return log_for_initial_model, remaining_log


def save_experiment(name, data, overwrite=True):
    if not overwrite:
        base_name = name
        suffix = 1
        while os.path.exists(f"{name}.pickle"):
            name = f"{base_name}_{suffix}"
            suffix += 1

    with open(f"{name}.pickle", "wb") as file:
        pickle.dump(data, file)


def get_top_variants_for_threshold(log, threshold):
    variants = get_ordered_variants(log, OrderingStrategy.FREQUENCY_SORTED)
    filtered_variants = variants[: ceil(len(variants) * threshold)]
    filtered_variants = list(map(lambda item: item[0], filtered_variants))
    return filter_variants(log, filtered_variants)


def get_frequency_sorted_variants(log):
    variants = get_variants_as_tuples(log)
    sorted_variants = sorted(
        variants.items(),
        key=lambda item: item[1] if isinstance(item[1], int) else len(item[1]),
        reverse=True,
    )
    return sorted_variants


def get_shuffled_variants(log, seed=None):
    if seed != None:
        random.seed(seed)

    variants = list(get_variants_as_tuples(log).items())
    random.shuffle(variants)

    return variants


def get_ordered_variants(log, ordering_strategy):
    match ordering_strategy:
        case OrderingStrategy.SHUFFLED:
            return get_shuffled_variants(log)
        case OrderingStrategy.FREQUENCY_SORTED:
            return get_frequency_sorted_variants(log)
        case _:
            if isinstance(ordering_strategy, OrderingStrategy):
                raise NotImplementedError
            else:
                raise ValueError("Invalid value for ordering_strategy.")


def create_trace_for_variant(variant):
    trace = Trace()
    for act_name in variant:
        trace.append(Event({"concept:name": act_name}))
    return trace


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        raise TypeError(
            "Input has to be the following args: `logpath approach initial_model_method [ordering_strategy]`"
        )
    log_path = sys.argv[1]
    path_head, path_tail = os.path.split(log_path)
    log_name = path_tail.split(".")[-2]

    if log_path.endswith(".xes"):
        log = read_xes(log_path)
        pass
    else:
        raise NotImplementedError

    approach = Approach(int(sys.argv[2]))
    initial_method = InitialModelMethod(int(sys.argv[3]))
    ordering_strategy = (
        OrderingStrategy(int(sys.argv[4]))
        if len(sys.argv) == 5
        else OrderingStrategy.FREQUENCY_SORTED
    )

    print(
        f"Executing {approach.name} with initial method {initial_method.name} and ordering_strategy {ordering_strategy.name}"
    )

    run_approach_evaluation(log, log_name, approach, initial_method, ordering_strategy)
