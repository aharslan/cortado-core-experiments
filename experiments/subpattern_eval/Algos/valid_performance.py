from collections import defaultdict, deque
from typing import Mapping

from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)

from cortado_core.subprocess_discovery.subtree_mining.three_pattern_candiate_generation import (
    compute_freq3,
)

from cortado_core.subprocess_discovery.subtree_mining.tree_pattern import TreePattern

from cortado_core.subprocess_discovery.subtree_mining.treebank import TreeBankEntry

from cortado_core.subprocess_discovery.subtree_mining.tree_pruning import (
    compute_f3_pruned_set,
)
from cortado_core.subprocess_discovery.subtree_mining.folding_label import fold_loops


def min_sub_mining_performance(
    treebank: Mapping[int, TreeBankEntry],
    frequency_counting_strat: FrequencyCountingStrategy,
    k_it,
    min_sup,
    loop=False,
    bfs_traversal=False,
):
    """
    Args:
        treebank (Mapping[int, TreeBankEntry]): _description_
        frequency_counting_strat (FrequencyCountingStrategy): _description_
        k_it (_type_): _description_
        min_sup (_type_): _description_
        loop (bool, optional): _description_. Defaults to False.
        bfs_traversal (bool, optional): _description_. Defaults to False.

    Comment:
    Valid Tree Miner (that further returns the number of candidate patterns created)
    """

    if loop:
        fold_loops(treebank, loop)

    fSets, F3 = compute_freq3(treebank, frequency_counting_strat, min_sup)

    pSets, F3 = compute_f3_pruned_set(fSets, F3)
    k_patterns = defaultdict(set)
    k_patterns[3] = F3
    nC = 0

    Q: deque[TreePattern] = deque(F3)

    while len(Q) > 0:
        tp = Q.pop()

        if tp.size >= k_it:
            continue  # Skip after reaching k cut off

        # Compute the right most path extension of all k-1 pattern
        tps = tp.right_most_path_extension(pSets)
        sup_to_gain = tp.support
        nC += len(tps)

        for c in tps:
            if f := c.update_rmo_list(
                treebank, min_sup, frequency_counting_strat, sup_to_gain
            ):
                if bfs_traversal:
                    Q.appendleft(f)
                else:
                    Q.append(f)

                k_patterns[tp.size + 1].add(f)

        del tp.rmo

    return k_patterns, nC


def min_sub_mining_memory(
    treebank: Mapping[int, TreeBankEntry],
    frequency_counting_strat: FrequencyCountingStrategy,
    k_it,
    min_sup,
    loop=False,
    bfs_traversal=True,
):
    """_summary_

    Args:
        treebank (Mapping[int, TreeBankEntry]): _description_
        frequency_counting_strat (FrequencyCountingStrategy): _description_
        k_it (_type_): _description_
        min_sup (_type_): _description_
        loop (bool, optional): _description_. Defaults to False.
        bfs_traversal (bool, optional): _description_. Defaults to False.

    Comment:
    Valid Tree Miner (without RMO deletion after each iteration for Memory Usage Eval)
    """
    if loop:
        fold_loops(treebank, loop)

    fSets, F3 = compute_freq3(treebank, frequency_counting_strat, min_sup)

    pSets, F3 = compute_f3_pruned_set(fSets, F3)
    k_patterns = defaultdict(set)
    k_patterns[3] = F3

    Q: deque[TreePattern] = deque(F3)
    nC = 0

    while len(Q) > 0:
        tp = Q.pop()

        if tp.size >= k_it:
            continue  # Skip after reaching k cut off

        # Compute the right most path extension of all k-1 pattern
        tps = tp.right_most_path_extension(pSets)
        sup_to_gain = tp.support

        for c in tps:
            if f := c.update_rmo_list(
                treebank, min_sup, frequency_counting_strat, sup_to_gain
            ):
                if bfs_traversal:
                    Q.appendleft(f)
                else:
                    Q.append(f)

                k_patterns[tp.size + 1].add(f)

        nC += len(tps)
        # del c.rmo

    return k_patterns, nC
