from collections import Counter
from typing import List, Mapping
from pm4py.objects.log.obj import Trace
from cortado_core.utils.split_graph import Group


def compute_frequent_activity_sets(variants: Mapping[Group, List[Trace]]):
    """
    Computes the sets of frequent activites in a single pass over the variants, takes into account the different frequency counting strategies

    Args:
        variants (Mapping[Group, List[Trace]]): The variants object as created in cvariants
        freq_strat FrequencyCountingStrategy: Frequency Counting Strategy
        min_sup int: The minimal support

    Returns:
        _type_: _description_
    """

    directly_follows_counter = Counter()
    follows_counter = Counter()
    concurrent_counter = Counter()
    start_activities = Counter()
    end_activities = Counter()
    activities = Counter()

    df_C: Counter
    c_C: Counter
    sa_C: Counter
    ea_C: Counter
    a_C: Counter

    count = lambda x: {k: len(v) for k, v in x.items()}

    def add(lCounter: Mapping, rCounter: Mapping):
        for k, v in rCounter.items():
            nVal = lCounter.get(k, 0) + v
            lCounter[k] = nVal

        return lCounter

    for variant in variants:
        nT = len(variants[variant])

        df_C = count(variant.graph.directly_follows)
        c_C = count(variant.graph.concurrency_pairs)
        f_C = count(variant.graph.follows)
        a_C = count(variant.graph.events)
        sa_C = count(variant.graph.start_activities)
        ea_C = count(variant.graph.end_activities)
        directly_follows_counter = add(directly_follows_counter, df_C)
        concurrent_counter = add(concurrent_counter, c_C)
        follows_counter = add(follows_counter, f_C)
        start_activities = add(start_activities, sa_C)
        end_activities = add(end_activities, ea_C)
        activities = add(activities, a_C)

    # Check if the Activites are above a certain support
    frequent_df_pairs = set(directly_follows_counter)
    frequent_f_pairs = set(follows_counter)
    frequent_cc_pairs = set(concurrent_counter)
    frequent_end_activity = set(end_activities)
    frequent_start_activity = set(start_activities)
    frequent_activities = set(activities)

    def flatten_pairs(pairs):
        """
        Flatten a pair into a {l : rs} dict and returns lists of the right and left elements
        """

        freq_dict = {}
        ls = []
        rs = []

        for l, r in pairs:
            freq_dict[l] = freq_dict.get(l, []) + [r]
            ls += [l]
            rs += [r]

        return freq_dict, ls, rs

    frequent_f_relations, frequent_f_starter, _ = flatten_pairs(frequent_f_pairs)
    frequent_df_relations, frequent_df_starter, _ = flatten_pairs(frequent_df_pairs)
    frequent_cc_relations, cc_ls, cc_rs = flatten_pairs(frequent_cc_pairs)
    frequent_cc_activities = set(cc_ls + cc_rs)

    return (
        frequent_activities,
        frequent_start_activity,
        frequent_end_activity,
        frequent_df_relations,
        set(frequent_df_starter),
        frequent_f_relations,
        set(frequent_f_starter),
        frequent_cc_relations,
        frequent_cc_activities,
    )
