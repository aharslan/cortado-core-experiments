import random
from typing import List, Callable, Dict, Iterable, Tuple

from pm4py.objects.log.importer.xes.importer import apply as xes_import
from pm4py.objects.log.obj import EventLog, Trace

from cortado_core.subprocess_discovery.concurrency_trees.cTrees import cTreeFromcGroup
from cortado_core.subprocess_discovery.subtree_mining.maximal_connected_components.maximal_connected_check import (
    check_if_valid_tree,
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
from cortado_core.utils.cvariants import get_concurrency_variants


def randomly_select_half(patterns):
    return random.sample(patterns, k=len(patterns) // 2)


def apply(
    event_log_filename: str,
    min_support: int,
    max_size: int,
    counting_strategy: FrequencyCountingStrategy,
    mine_prefixes_suffixes: bool,
    selection_function: Callable[
        [List[TreePattern]], List[TreePattern]
    ] = randomly_select_half,
) -> Tuple[Dict[Trace, List[bool]], List[TreePattern]]:
    """
    Takes an event log filename, pattern mining parameters and a selection function, and returns (a subset of) infix
    patterns and a dictionary, indicating the presence of patterns in each trace.

    An example call looks as follows:
    res_dict, patterns = apply('C:\\event-logs\\exported.xes', 20, 100, FrequencyCountingStrategy.TraceTransaction,
                               True, selection_function=randomly_select_half)
    """
    event_log = xes_import(event_log_filename)
    variants = get_concurrency_variants(event_log)
    tree_bank = create_treebank_from_cv_variants(
        variants, mine_prefixes_suffixes, add_traces=True
    )
    patterns = min_sub_mining(
        tree_bank,
        frequency_counting_strat=counting_strategy,
        k_it=max_size,
        min_sup=min_support,
    )
    flat_patterns = flatten_patterns(patterns)
    flat_patterns = selection_function(flat_patterns)

    return (
        calculate_presence_of_patterns_in_traces(variants, flat_patterns, tree_bank),
        flat_patterns,
    )


def calculate_presence_of_patterns_in_traces(
    variants, flat_patterns: List[TreePattern], tree_bank
) -> Dict[Trace, List[bool]]:
    results = initialize_results(variants)
    for pattern in flat_patterns:
        for tree_bank_entry in tree_bank.values():
            is_present = tree_bank_entry.uid in pattern.rmo
            for trace in tree_bank_entry.traces:
                results[trace].append(is_present)

    return results


def initialize_results(variants):
    results = dict()
    for variant, traces in variants.items():
        for trace in traces:
            results[trace] = []
    return results


def flatten_patterns(patterns: Dict[int, Iterable[TreePattern]]):
    flat_patterns = []

    for it, it_patterns in patterns.items():
        for pattern in it_patterns:
            if check_if_valid_tree(pattern.tree):
                flat_patterns.append(pattern)

    return flat_patterns


def verify_results(res: Dict[Trace, List[bool]], patterns: List[TreePattern]):
    """
    For each pattern in patterns, verify_results list the present and not present traces by printing their concurrency
    tree representation. Useful for checking if the results look reasonable.
    Parameters
    ----------
    res
    patterns

    Returns
    -------

    """
    for i, pattern in enumerate(patterns):
        print("------------ pattern ---------------")
        print(pattern.tree)

        present_variants = []
        not_present_variants = []
        for trace, indicators in res.items():
            variant = list(get_concurrency_variants(EventLog([trace])).keys())[0]
            if indicators[i]:
                present_variants.append(cTreeFromcGroup(variant))
            else:
                not_present_variants.append(cTreeFromcGroup(variant))

        print("------------ present traces ---------------")

        for v in present_variants:
            print(v)

        print("------------ not present traces ---------------")

        for v in not_present_variants:
            print(v)


if __name__ == "__main__":
    r, pts = apply(
        "C:\\sources\\arbeit\\cortado\\event-logs\\exported.xes",
        20,
        100,
        FrequencyCountingStrategy.TraceTransaction,
        True,
        selection_function=randomly_select_half,
    )
    verify_results(r, pts)
    print("finished")
