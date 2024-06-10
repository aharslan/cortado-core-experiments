from cortado_core.tests.pattern_mining.util.initial_candidate_generation_2_patterns import (
    generate_initial_candidates,
)
from cortado_core.subprocess_discovery.subtree_mining.ct_frequency_counting import (
    ct_compute_frequent_activity_sets,
)
from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)
from cortado_core.subprocess_discovery.subtree_mining.tree_pruning import (
    _get_prune_sets,
    compute_f3_pruned_set_2_patterns,
)
from cortado_core.subprocess_discovery.subtree_mining.utilities import (
    _contains_fallthrough,
    compute_occurence_list_size,
)


def min_sub_mining_performance_2_pattern(
    treebank,
    frequency_counting_strat: FrequencyCountingStrategy,
    k_it,
    min_sup,
):
    """ """
    fSets = ct_compute_frequent_activity_sets(
        treebank, frequency_counting_strat, min_sup
    )

    F = generate_initial_candidates(treebank, min_sup, frequency_counting_strat, fSets)

    has_fallthroughs = any([_contains_fallthrough(f.tree) for f in F])
    nC = len(F)

    # Store the results
    k_pattern = {2: F}

    pSets = _get_prune_sets(fSets, F)

    skipPrune = True

    # For every k > 2 create the k pattern from the frequent k-1 pattern
    for k in range(k_it):
        newF = []

        for tp in F:
            # Compute the right most path extension of all k-1 pattern
            tps = tp.right_most_path_extension(pSets, skipPrune, has_fallthroughs)
            nC += len(tps)

            sup_to_gain = tp.support
            del tp.rmo
            # Add the extended patterns to the k Candidate set

            for c in tps:
                if f := c.update_rmo_list(
                    treebank, min_sup, frequency_counting_strat, sup_to_gain
                ):
                    newF.append(f)

        # For each candidate update the rmo and through this compute the support

        F = newF

        # Break early, if there is no frequent pattern lefts
        if len(F) > 0:
            k_pattern[3 + k] = F
        else:
            break

        if k == 0:
            pSets, F = compute_f3_pruned_set_2_patterns(pSets, F)
            skipPrune = False

    return k_pattern, nC


def min_sub_mining_memory_2_pattern(
    treebank,
    frequency_counting_strat: FrequencyCountingStrategy,
    k_it,
    min_sup,
    delete_rmo: bool = True,
):
    """ """

    fSets = ct_compute_frequent_activity_sets(
        treebank, frequency_counting_strat, min_sup
    )

    F = generate_initial_candidates(treebank, min_sup, frequency_counting_strat, fSets)

    has_fallthroughs = any([_contains_fallthrough(f.tree) for f in F])
    nC = len(F)

    # Store the results
    k_pattern = {2: F}
    occurence_list_sizes = []

    pSets = _get_prune_sets(fSets, F)

    skipPrune = True

    # For every k > 2 create the k pattern from the frequent k-1 pattern
    for k in range(k_it):
        occurence_list_sizes.append(
            sum([compute_occurence_list_size(f.rmo) for f in F])
        )

        newF = []

        for tp in F:
            # Compute the right most path extension of all k-1 pattern
            tps = tp.right_most_path_extension(pSets, skipPrune, has_fallthroughs)
            nC += len(tps)

            sup_to_gain = tp.support

            if delete_rmo:
                del tp.rmo
            # Add the extended patterns to the k Candidate set

            for c in tps:
                if f := c.update_rmo_list(
                    treebank, min_sup, frequency_counting_strat, sup_to_gain
                ):
                    newF.append(f)

        # For each candidate update the rmo and through this compute the support

        F = newF

        # Break early, if there is no frequent pattern lefts
        if len(F) > 0:
            k_pattern[3 + k] = F
        else:
            occurence_list_sizes.append(
                sum([compute_occurence_list_size(f.rmo) for f in F])
            )
            break

        if k == 0:
            pSets, F = compute_f3_pruned_set_2_patterns(pSets, F)
            skipPrune = False

    return k_pattern, nC, occurence_list_sizes
