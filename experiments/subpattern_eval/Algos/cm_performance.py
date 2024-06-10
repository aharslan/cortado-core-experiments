from typing import Mapping

from cortado_core.subprocess_discovery.subtree_mining.blanket_mining.cm_tree import (
    CMConcurrencyTree,
)
from cortado_core.subprocess_discovery.subtree_mining.blanket_mining.cm_tree_pattern import (
    CMTreePattern,
)
from cortado_core.subprocess_discovery.subtree_mining.blanket_mining.compute_frequency_blanket import (
    check_frequency_blanket,
)
from cortado_core.subprocess_discovery.subtree_mining.blanket_mining.compute_occurence_blanket import (
    check_occ_blanket,
)
from cortado_core.subprocess_discovery.subtree_mining.blanket_mining.compute_root_occurence_blanket import (
    check_root_occurence_blanket,
)
from cortado_core.subprocess_discovery.subtree_mining.blanket_mining.compute_transaction_blanket import (
    check_transaction_blanket,
)

from cortado_core.subprocess_discovery.subtree_mining.blanket_mining.create_initial_candidates import (
    generate_initial_candidates as generate_initial_candidates_cm,
)
from cortado_core.subprocess_discovery.subtree_mining.ct_frequency_counting import (
    ct_compute_frequent_activity_sets,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
    PruningSets,
)
from cortado_core.subprocess_discovery.subtree_mining.tree_pruning import (
    _get_prune_sets,
    compute_f3_pruned_set,
)
from cortado_core.subprocess_discovery.subtree_mining.treebank import TreeBankEntry
from cortado_core.subprocess_discovery.subtree_mining.utilities import (
    _contains_fallthrough,
    compute_occurence_list_size,
)
from cortado_core.utils.split_graph import Group
from cortado_core.subprocess_discovery.subtree_mining.maximal_connected_components.maximal_connected_check import (
    check_if_valid_tree,
)


def cm_grow_performance(
    tp: CMTreePattern,
    treebank,
    min_sup: int,
    frequency_counting_strat: FrequencyCountingStrategy,
    skipPrune: bool,
    pSets: PruningSets,
    has_fallthroughs: bool,
):
    nC = 0

    occurenceBased = (
        frequency_counting_strat == FrequencyCountingStrategy.TraceOccurence
        or frequency_counting_strat == FrequencyCountingStrategy.VariantOccurence
    )
    E = []
    B_left_occ_not_empty, B_occ_not_empty = check_occ_blanket(tp)

    if (not skipPrune) and B_left_occ_not_empty:
        return None, 0

    else:
        patterns = tp.right_most_path_extension(pSets, skipPrune, has_fallthroughs)
        nC = len(patterns)
        sup_to_gain = tp.support

        for e in patterns:
            if p := e.update_rmo_list(
                treebank, min_sup, frequency_counting_strat, sup_to_gain
            ):
                E.append(p)

    if not B_occ_not_empty and check_if_valid_tree(tp.tree):
        if (occurenceBased and not check_root_occurence_blanket(tp)) or (
            not occurenceBased and not check_transaction_blanket(tp)
        ):
            tp.closed = True

            if not any([p.rml.label for p in E]):
                B_freq_not_empty = check_frequency_blanket(
                    tp=tp,
                    min_sup=min_sup,
                    treeBank=treebank,
                    strategy=frequency_counting_strat,
                )

                if not B_freq_not_empty:
                    tp.maximal = True

    del tp.rmo
    return E, nC


def cm_min_sub_mining_performance(
    treebank: Mapping[int, TreeBankEntry],
    frequency_counting_strat: FrequencyCountingStrategy,
    k_it: int,
    min_sup: int,
):
    """ """

    fSets = ct_compute_frequent_activity_sets(
        treebank, frequency_counting_strat, min_sup
    )

    C = generate_initial_candidates_cm(
        treebank, min_sup, frequency_counting_strat, fSets
    )
    k_pattern = {2: C}

    nCandidatesGenerated = len(C)
    has_fallthroughs = any([_contains_fallthrough(f.tree) for f in C])

    # Define the initial Prune Sets
    pSets = _get_prune_sets(fSets, C)

    # Skip the initial pruning step to compute the full set of F3 Patterns
    skipPrune = True

    for k in range(k_it):
        E = []

        for c in C:
            res, nC = cm_grow_performance(
                c,
                treebank,
                min_sup,
                frequency_counting_strat,
                skipPrune,
                pSets,
                has_fallthroughs,
            )

            nCandidatesGenerated += nC

            c.tree.clean_occurence_list()

            if res:
                E.extend(res)

        C = E

        if len(E) > 0:
            k_pattern[k + 3] = E
        else:
            break

        if k == 0:
            pSets, C = compute_f3_pruned_set(pSets, C)
            skipPrune = False

    return k_pattern, nCandidatesGenerated


def cm_min_sub_mining_memory(
    treebank: Mapping[int, TreeBankEntry],
    frequency_counting_strat: FrequencyCountingStrategy,
    k_it: int,
    min_sup: int,
    delete_rmo: bool = True,
):
    """ """

    fSets = ct_compute_frequent_activity_sets(
        treebank, frequency_counting_strat, min_sup
    )

    C = generate_initial_candidates_cm(
        treebank, min_sup, frequency_counting_strat, fSets
    )

    k_pattern = {2: C}

    nCandidatesGenerated = len(C)
    has_fallthroughs = any([_contains_fallthrough(f.tree) for f in C])
    occurence_list_sizes = []

    pSets = _get_prune_sets(fSets, C)

    # Skip the initial pruning step to compute the full set of F3 Patterns
    skipPrune = True

    for k in range(k_it):
        E = []

        occurence_list_sizes.append(
            sum([compute_all_occurence_list_entries(f.tree) for f in C])
        )

        for c in C:
            res, nC = cm_grow_performance(
                c,
                treebank,
                min_sup,
                frequency_counting_strat,
                skipPrune,
                pSets,
                has_fallthroughs,
            )

            nCandidatesGenerated += nC

            c.tree.clean_occurence_list()

            if res:
                E.extend(res)

        C = E

        if len(E) > 0:
            k_pattern[k + 3] = E
        else:
            occurence_list_sizes.append(
                sum([compute_all_occurence_list_entries(f.tree) for f in C])
            )
            break

        if k == 0:
            pSets, C = compute_f3_pruned_set(pSets, C)
            skipPrune = False

    return k_pattern, nCandidatesGenerated, occurence_list_sizes


def compute_all_occurence_list_entries(tree: CMConcurrencyTree):
    occ_list_size = compute_occurence_list_size(tree.occList)

    for child in tree.children:
        occ_list_size += compute_all_occurence_list_entries(child)

    return occ_list_size
