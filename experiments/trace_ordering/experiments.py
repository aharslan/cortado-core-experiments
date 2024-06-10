import copy
from multiprocessing import Pool
import os
import sys
from typing import List, Tuple, Dict
import time

import pandas as pd
import pm4py
from tqdm import tqdm


from experiments.ipda.discovery import (
    InitialModelMethod,
    create_trace_for_variant,
    split_initial_log,
    save_experiment,
)
from experiments.ipda.evaluation import evaluate_model

from experiments.trace_ordering.create_plots import create_plots
from cortado_core.lca_approach import add_trace_to_pt_language
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.util.variants_util import variant_to_trace
from pm4py.discovery import discover_process_tree_inductive
from pm4py.objects.petri_net.exporter.exporter import serialize as serialize_pn
from pm4py.stats import get_variants

from cortado_core.trace_ordering.filter.rate_filter import RateFilter
from cortado_core.trace_ordering.scoring.alignment_trace_scorer import (
    AlignmentTraceScorer,
)
from cortado_core.trace_ordering.scoring.brute_force_trace_scorer import (
    BruteForceTraceScorer,
)
from cortado_core.trace_ordering.scoring.duplicates_trace_scorer import (
    DuplicatesTraceScorer,
)
from cortado_core.trace_ordering.scoring.lca_height_trace_scorer import (
    LCAHeightTraceScorer,
)
from cortado_core.trace_ordering.scoring.levenshtein_trace_scorer import (
    LevenshteinTraceScorer,
)
from cortado_core.trace_ordering.scoring.missing_activities_scorer import (
    MissingActivitiesScorer,
)
from cortado_core.trace_ordering.scoring.random_scorer import RandomScorer
from cortado_core.trace_ordering.scoring.trace_scorer_adapter import TraceScorerAdapter
from cortado_core.trace_ordering.strategy.strategy import Strategy
from cortado_core.trace_ordering.strategy_component.highest_best_strategy_component import (
    HighestBestStrategyComponent,
)
from cortado_core.trace_ordering.strategy_component.lowest_best_strategy_component import (
    LowestBestStrategyComponent,
)


def run_experiments_for_strategy(
    strategy: Strategy,
    strategy_name: str,
    log: EventLog,
    log_name: str,
    initial_pt: ProcessTree,
    initial_log: EventLog,
):
    variants_log, variant_count = __create_event_log_for_variants(log)
    candidate_traces = [trace for trace in variants_log]
    initial_log = list(
        map(lambda x: create_trace_for_variant(x), get_variants(initial_log))
    )
    input_model = copy.deepcopy(initial_pt)
    input_pn, input_im, input_fm = pm4py.convert.convert_to_petri_net(input_model)

    results = []

    with Pool() as multiprocessing_pool:
        with tqdm(total=len(candidate_traces)) as pbar:
            while len(candidate_traces) > 0:
                trace_selection_start_time = time.time()
                trace_to_be_added = strategy.apply_trace(
                    log, initial_log, input_model, candidate_traces
                )
                trace_selection_duration = time.time() - trace_selection_start_time
                output_model = add_trace_to_pt_language(
                    input_model,
                    EventLog(initial_log),
                    trace_to_be_added,
                    try_pulling_lca_down=True,
                    pool=multiprocessing_pool,
                )

                output_pn, output_im, output_fm = pm4py.convert.convert_to_petri_net(
                    output_model
                )

                fitness, precision = evaluate_model(
                    output_pn, output_im, output_fm, log
                )

                iteration_dict = {}
                iteration_dict["input_model"] = serialize_pn(
                    input_pn, input_im, input_fm
                )
                iteration_dict["added_variant"] = tuple(
                    map(lambda x: x["concept:name"], trace_to_be_added)
                )
                iteration_dict["added_variant_frequency"] = variant_count[
                    trace_to_be_added
                ]
                iteration_dict["output_model"] = serialize_pn(
                    output_pn, output_im, output_fm
                )
                iteration_dict["input_tree"] = copy.deepcopy(input_model)
                iteration_dict["output_tree"] = copy.deepcopy(output_model)
                iteration_dict["duration"] = trace_selection_duration
                iteration_dict["fitness"] = fitness
                iteration_dict["precision"] = precision
                iteration_dict["f-measure"] = (
                    2 * fitness * precision / (fitness + precision)
                )  # harmonic mean

                results.append(iteration_dict)
                input_model = output_model
                input_pn, input_im, input_fm = output_pn, output_im, output_fm
                save_experiment(f"{log_name}__{strategy_name}__preliminary", results)

                candidate_traces.remove(trace_to_be_added)
                initial_log.append(trace_to_be_added)
                pbar.update(1)

    save_experiment(f"{log_name}__{strategy_name}__final", results, False)
    return results


def __create_event_log_for_variants(log: EventLog) -> Tuple[EventLog, Dict[Trace, int]]:
    variants = pm4py.get_variants_as_tuples(log)
    new_log = EventLog()
    trace_to_count = dict()

    for variant, traces in variants.items():
        trace = variant_to_trace(variant)
        new_log.append(trace)
        trace_to_count[trace] = traces if isinstance(traces, int) else len(traces)

    return new_log, trace_to_count


def get_strategy(strategy_name: str, filter_rate: int) -> List[Tuple[str, Strategy]]:
    # C Alignment Costs
    # M Missing Activities
    # L Levenshtein Distance
    # B Brute Force
    # D Duplicates
    # H LCA Height
    # R Random
    strategy_components = {
        "C": LowestBestStrategyComponent(TraceScorerAdapter(AlignmentTraceScorer())),
        "M": LowestBestStrategyComponent(MissingActivitiesScorer()),
        "L": LowestBestStrategyComponent(TraceScorerAdapter(LevenshteinTraceScorer())),
        "B": HighestBestStrategyComponent(TraceScorerAdapter(BruteForceTraceScorer())),
        "D": LowestBestStrategyComponent(
            TraceScorerAdapter(DuplicatesTraceScorer(try_pulling_lca_down=True))
        ),
        "H": HighestBestStrategyComponent(
            TraceScorerAdapter(LCAHeightTraceScorer(try_pulling_lca_down=True))
        ),
        "R": LowestBestStrategyComponent(RandomScorer()),
    }

    strategy = []
    for i, strategy_component in enumerate(strategy_name):
        if strategy_component.upper() not in strategy_components:
            raise ValueError(f"Strategy-Component {strategy_component} not found.")
        if i + 1 == len(strategy_name):
            filter_rate = 0
        strategy.append(
            (strategy_components[strategy_component.upper()], RateFilter(filter_rate))
        )

    return Strategy(strategy)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise TypeError(
            "Input has to be the following args: `logpath strategy filter_rate`"
        )
    log_path = sys.argv[1]
    path_head, path_tail = os.path.split(log_path)
    log_name = path_tail.split(".")[-2]

    if log_path.endswith(".xes"):
        log = pm4py.read.read_xes(log_path)
        pass
    else:
        raise NotImplementedError

    strategy_name = sys.argv[2].upper()
    filter_rate = float(sys.argv[3])

    print(f"Executing strategy {strategy_name} with filter rate {filter_rate}")

    strategy = get_strategy(strategy_name, filter_rate)

    initial_log, remaining_log = split_initial_log(log, InitialModelMethod.TOP_1)
    initial_pt = discover_process_tree_inductive(initial_log)

    run_experiments_for_strategy(
        strategy,
        f"{strategy_name}_{filter_rate}",
        remaining_log,
        log_name,
        initial_pt,
        initial_log,
    )
