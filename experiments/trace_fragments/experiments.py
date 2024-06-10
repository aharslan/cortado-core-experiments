import glob
import math
import os
from enum import Enum
from pathlib import Path

import pandas as pd
import pm4py

from pm4py.objects.log.importer.xes.importer import apply as xes_import
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.util.interval_lifecycle import to_interval
from pm4py.util.variants_util import variant_to_trace
from pm4py.util.lp import solver as lp_solver

from experiments.action_oriented_process_mining.experiment import (
    flatten_patterns,
)
from experiments.trace_fragments.create_plots import create_plots
from cortado_core.lca_approach import add_trace_to_pt_language
from cortado_core.models.infix_type import InfixType
from cortado_core.process_tree_utils.reduction import apply_reduction_rules
from cortado_core.subprocess_discovery.concurrency_trees.cTrees import cTreeOperator
from cortado_core.subprocess_discovery.subtree_mining.maximal_connected_components.maximal_connected_check import (
    set_maximaly_closed_patterns,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.subprocess_discovery.subtree_mining.right_most_path_extension.min_sub_mining import (
    min_sub_mining,
)
from cortado_core.subprocess_discovery.subtree_mining.tree_pattern import TreePattern
from cortado_core.subprocess_discovery.subtree_mining.treebank import (
    create_treebank_from_cv_variants,
)
from cortado_core.trace_ordering.utils.f_mesaure import calculate_f_measure
from cortado_core.utils.constants import ARTIFICAL_START_NAME, ARTIFICAL_END_NAME
from cortado_core.utils.cvariants import get_concurrency_variants
from cortado_core.utils.sequentializations import generate_sequentializations
from cortado_core.utils.split_graph import SequenceGroup, LeafGroup
from cortado_core.utils.trace import TypedTrace


class PatternType(Enum):
    All = 1
    Maximal = 2
    Closed = 3


class Sorting(Enum):
    Support = 1
    NumberChevrons = 2


LOG_FILE = os.getenv("LOG_FILE", "BPI_Challenge_2012_short.xes")
EVENT_LOG_DIRECTORY = os.getenv(
    "EVENT_LOG_DIRECTORY", "C:\\sources\\arbeit\\cortado\\event-logs"
)
RESULTS_DIRECTORY = os.getenv(
    "RESULT_DIRECTORY", "C:\\sources\\arbeit\\cortado\\trace_fragments"
)
FREQ_COUNT_STRAT = FrequencyCountingStrategy(
    int(
        os.getenv(
            "FREQ_COUNT_STRAT", FrequencyCountingStrategy.VariantTransaction.value
        )
    )
)
MIN_SUPPORT_REL = float(os.getenv("MIN_SUPPORT", "0.02"))
PATTERN_TYPE = PatternType(int(os.getenv("PATTERN_TYPE", PatternType.All.value)))
SORTING = Sorting(int(os.getenv("SORTING", Sorting.Support.value)))
REL_PERCENTAGE_INITIAL_MODEL = float(os.getenv("REL_SUPP_INIT_MODEL", "0.1"))
NOISE_THRESHOLD = float(os.getenv("NOISE_THRESHOLD_INIT_MODEL", "0"))
PERFORM_ONLY_CLASSICAL = os.getenv("PERFORM_ONLY_CLASSICAL", "0") == "1"
USE_SEQUENTIAL_VARIANTS = os.getenv("SEQUENTIAL_VARIANTS", "1") == "1"

COLUMNS = ["processed_variants", "f_measure", "fitness", "precision", "no_model_change"]


def run_experiments(
    event_log_filename,
    counting_strategy,
    max_size,
    min_support_rel,
    pattern_type,
    sorting,
    rel_percentage_init_model,
    noise_threshold,
    is_trace_fragment_experiments: bool,
    use_seq_variants: bool,
):
    event_log = xes_import(event_log_filename)
    variants = get_variants(event_log, use_seq_variants)
    min_support = get_support_count(
        min_support_rel, counting_strategy, len(event_log), len(variants)
    )
    initial_model, already_added, remaining_variants = get_initial_model(
        variants, rel_percentage_init_model, noise_threshold
    )

    if is_trace_fragment_experiments:
        infixes_to_add = get_infixes_to_add(
            variants, counting_strategy, max_size, min_support, pattern_type
        )
        infixes_to_add = apply_sorting(infixes_to_add, sorting)
        infixes_to_add = [
            (p.to_concurrency_group(), get_infix_type(p)) for p in infixes_to_add
        ]
    else:
        infixes_to_add = [(v, InfixType.NOT_AN_INFIX) for v in remaining_variants]

    print("Start adding", len(infixes_to_add), "infixes")

    results = add_infixes_to_initial_model(
        initial_model,
        infixes_to_add,
        already_added,
        eval=lambda model: eval_function(event_log, model),
    )
    results_df = pd.DataFrame(results, columns=COLUMNS)
    results_path = os.path.join(
        RESULTS_DIRECTORY,
        f'results_{LOG_FILE}_{"infix" if is_trace_fragment_experiments else "classical"}_{"cvariants" if not use_seq_variants else "seqvariants"}_{PATTERN_TYPE}_{SORTING}_{FREQ_COUNT_STRAT}_{REL_PERCENTAGE_INITIAL_MODEL}_{NOISE_THRESHOLD}_minsup_{MIN_SUPPORT_REL}',
    )
    Path(results_path).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(os.path.join(results_path, "data.csv"))
    create_plots(results_path, is_trace_fragment_experiments)


def add_infixes_to_initial_model(initial_model, infixes_to_add, already_added, eval):
    evaluation_result = eval(initial_model)
    results = [[0] + evaluation_result]
    model = initial_model
    previous_evaluation = evaluation_result
    for idx, (infix_to_add, infix_type) in enumerate(infixes_to_add):
        traces = get_typed_traces_for_variant(infix_to_add, infix_type)
        for trace in traces:
            previous_model = model
            model = add_trace_to_pt_language(
                model,
                already_added,
                trace,
                try_pulling_lca_down=True,
                add_artificial_start_end=True,
            )
            already_added.append(trace)

        if model == previous_model:
            results.append([idx + 1] + previous_evaluation + [True])
        else:
            evaluation_result = eval(model)
            results.append([idx + 1] + evaluation_result + [False])
            previous_evaluation = evaluation_result

    return results


def eval_function(log, model):
    f, fitness, precision = calculate_f_measure(model, log)
    return [f, fitness, precision]


def get_infix_type(pattern: TreePattern):
    tree = pattern.tree

    if tree.op != cTreeOperator.Sequential:
        return InfixType.PROPER_INFIX

    if (
        tree.children[0].label == ARTIFICAL_START_NAME
        and tree.children[-1] == ARTIFICAL_END_NAME
    ):
        return InfixType.NOT_AN_INFIX

    if tree.children[0].label == ARTIFICAL_START_NAME:
        return InfixType.PREFIX

    if tree.children[-1].label == ARTIFICAL_END_NAME:
        return InfixType.POSTFIX

    return InfixType.PROPER_INFIX


def get_initial_model(variants, rel_percentage, noise_threshold):
    absolute_n_variants = math.ceil(rel_percentage * len(variants))
    variants_to_add = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)[
        :absolute_n_variants
    ]
    log = EventLog(get_traces_from_variants(variants_to_add))
    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=noise_threshold)
    apply_reduction_rules(tree)

    remaining_variants = sorted(
        variants.items(), key=lambda x: len(x[1]), reverse=True
    )[absolute_n_variants:]

    return (
        tree,
        [TypedTrace(t, InfixType.NOT_AN_INFIX) for t in log],
        [v for v, _ in remaining_variants],
    )


def get_infixes_to_add(
    variants, counting_strategy, max_size, min_support, pattern_type
):
    tree_bank = create_treebank_from_cv_variants(
        variants, artifical_start=True, add_traces=True
    )
    patterns = min_sub_mining(
        tree_bank,
        frequency_counting_strat=counting_strategy,
        k_it=max_size,
        min_sup=min_support,
    )

    if pattern_type == PatternType.All:
        return flatten_patterns(patterns)

    set_maximaly_closed_patterns(patterns)
    flat_patterns = flatten_patterns(patterns)

    if pattern_type == PatternType.Maximal:
        return [p for p in flat_patterns if p.maximal]

    if pattern_type == PatternType.Closed:
        return [p for p in flat_patterns if p.to_concurrency_group()]

    raise ValueError("Unknown pattern type")


def apply_sorting(infixes_to_add, sorting):
    infixes_to_add.sort(reverse=True, key=get_sorting_function(sorting))
    return infixes_to_add


def get_sorting_function(sorting):
    if sorting == Sorting.Support:
        return lambda p: p.support

    if sorting == Sorting.NumberChevrons:
        return lambda p: p.to_concurrency_group().number_of_activities()

    raise ValueError("Unknown sorting value")


def get_traces_from_variants(variants):
    traces = []

    for variant, t in variants:
        sequentializations = generate_sequentializations(variant)
        traces += [variant_to_trace(seq) for seq in sequentializations]

    return traces


def get_typed_traces_for_variant(variant, infix_type):
    sequentializations = generate_sequentializations(variant)
    traces = [
        TypedTrace(variant_to_trace(seq), infix_type) for seq in sequentializations
    ]

    return remove_artificial_events(traces)


def remove_artificial_events(traces):
    for trace in traces:
        trace.trace = Trace(
            [
                e
                for e in trace.trace
                if e["concept:name"] not in {ARTIFICAL_END_NAME, ARTIFICAL_START_NAME}
            ]
        )

    return traces


def get_support_count(
    rel_support: float,
    frequency_strategy: FrequencyCountingStrategy,
    n_traces: int,
    n_variants: int,
) -> int:
    if (
        frequency_strategy == FrequencyCountingStrategy.TraceOccurence
        or frequency_strategy == FrequencyCountingStrategy.TraceTransaction
    ):
        return round(n_traces * rel_support)

    return round(n_variants * rel_support)


def run_experiments_for_classical_discovery_techniques(event_log_filename: str):
    event_log = xes_import(event_log_filename)
    partitioned_filenames = partition_log(event_log)

    for noise_threshold in [0, 0.5, 0.7, 0.9]:
        discover_models_im(noise_threshold, partitioned_filenames)

    evaluate_models(event_log)


def partition_log(log: EventLog) -> list[str]:
    partition_paths = []
    results_path = os.path.join(RESULTS_DIRECTORY, f"partitioned_logs_{LOG_FILE}")
    Path(results_path).mkdir(parents=True, exist_ok=True)

    partitions = [0.2, 0.4, 0.6, 0.8, 1]
    new_log = EventLog()
    variants = pm4py.get_variants(log)
    variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)

    for partition in partitions:
        while len(new_log) < len(log) * partition:
            traces = variants.pop(0)[1]
            for trace in traces:
                new_log.append(trace)

        file = os.path.join(results_path, f"{partition}.xes")
        pm4py.write_xes(new_log, file)
        partition_paths.append(file)

    return partition_paths


def discover_models_im(
    noise_threshold: float,
    logs: list[str],
    res_path=os.path.join(RESULTS_DIRECTORY, LOG_FILE, f"classical"),
):
    for log_name in logs:
        log = xes_import(log_name)
        model = pm4py.discover_process_tree_inductive(
            log, noise_threshold=noise_threshold
        )
        filename = Path(log_name).stem
        results_path = os.path.join(res_path, f"im_{noise_threshold}")
        Path(results_path).mkdir(parents=True, exist_ok=True)
        pm4py.write_ptml(model, os.path.join(results_path, f"{filename}.ptml"))


def evaluate_models(
    log, results_path=os.path.join(RESULTS_DIRECTORY, LOG_FILE, f"classical")
):
    for folder, _, _ in os.walk(results_path):
        if folder == results_path:
            continue

        results = []
        for model_file in glob.glob(os.path.join(folder, "*.ptml")):
            percentage_added_traces = Path(model_file).stem
            model = pm4py.read_ptml(model_file)
            results.append(
                [percentage_added_traces] + eval_function(log, model) + [None]
            )

        results_df = pd.DataFrame(results, columns=COLUMNS)
        results_path = os.path.join(folder, "results.csv")
        results_df.to_csv(results_path)


def get_variants(log: EventLog, use_sequential_variants: bool):
    if not use_sequential_variants:
        return get_concurrency_variants(log)
    else:
        variants = pm4py.get_variants(log)
        return {
            __pm4py_variant_to_concurrency_variant(variant): traces
            for variant, traces in variants.items()
        }


def __pm4py_variant_to_concurrency_variant(variant: tuple[str]):
    return SequenceGroup([LeafGroup([a]) for a in variant])


if __name__ == "__main__":
    print("LP Solver:", lp_solver.DEFAULT_LP_SOLVER_VARIANT)

    if PERFORM_ONLY_CLASSICAL:
        run_experiments_for_classical_discovery_techniques(
            os.path.join(EVENT_LOG_DIRECTORY, LOG_FILE)
        )
    else:
        print("freq strat:", FREQ_COUNT_STRAT)
        run_experiments(
            os.path.join(EVENT_LOG_DIRECTORY, LOG_FILE),
            FREQ_COUNT_STRAT,
            1000,
            MIN_SUPPORT_REL,
            PATTERN_TYPE,
            SORTING,
            REL_PERCENTAGE_INITIAL_MODEL,
            NOISE_THRESHOLD,
            True,
            USE_SEQUENTIAL_VARIANTS,
        )
        run_experiments(
            os.path.join(EVENT_LOG_DIRECTORY, LOG_FILE),
            FREQ_COUNT_STRAT,
            1000,
            MIN_SUPPORT_REL,
            PATTERN_TYPE,
            SORTING,
            REL_PERCENTAGE_INITIAL_MODEL,
            NOISE_THRESHOLD,
            False,
            USE_SEQUENTIAL_VARIANTS,
        )
