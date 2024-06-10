from collections import Counter
from typing import Mapping, Set
from cortado_core.subprocess_discovery.concurrency_trees.cTrees import (
    ConcurrencyTree,
    cTreeOperator,
)

from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)

from cortado_core.subprocess_discovery.subtree_mining.right_most_path_extension.support_counting import (
    check_min_sup,
)

from cortado_core.subprocess_discovery.subtree_mining.tree_pattern import (
    TreePattern,
    extend_motif_on_operator_node,
)


def count_activites_in_tree(tree: ConcurrencyTree):
    seq = Counter()
    act = Counter()
    con = Counter()

    seq_set = set()
    act_set = set()
    con_set = set()

    for child in tree.children:
        if child.label:
            act_set.add(child.label)

            if tree.op == cTreeOperator.Sequential:
                seq_set.add(child.label)

            elif tree.op == cTreeOperator.Concurrent:
                con_set.add(child.label)

        else:
            seq_c, act_c, con_c = count_activites_in_tree(child)

            seq += seq_c
            act += act_c
            con += con_c

    seq += Counter(seq_set)
    act += Counter(act_set)
    con += Counter(con_set)

    return seq, act, con


def compute_frequent_activity_sets_asai(treebank, freq_strat, min_sup):
    df_counter = Counter()
    con_counter = Counter()
    act_counter = Counter()

    for _, tree_entry in treebank.items():
        df, act, con = count_activites_in_tree(tree_entry.tree)

        # If it is an Trace-based Strategy, multiply with the number of traces
        if (
            freq_strat == FrequencyCountingStrategy.TraceTransaction
            or freq_strat == FrequencyCountingStrategy.TraceOccurence
        ):
            nT = tree_entry.nTraces

        else:
            nT = 1

        if (
            freq_strat == FrequencyCountingStrategy.TraceTransaction
            or freq_strat == FrequencyCountingStrategy.VariantTransaction
        ):
            df_counter.update({key: nT for key, _ in df.items()})
            con_counter.update({key: nT for key, _ in con.items()})
            act_counter.update({key: nT for key, _ in act.items()})

        else:
            df_counter.update({key: nT * count for key, count in df.items()})
            con_counter.update({key: nT * count for key, count in con.items()})
            act_counter.update({key: nT * count for key, count in act.items()})

    frequent_df = set([pair for pair in df_counter if df_counter[pair] > min_sup])
    frequent_act = set([pair for pair in act_counter if act_counter[pair] > min_sup])
    frequent_cc = set([pair for pair in con_counter if con_counter[pair] > min_sup])

    return frequent_act, frequent_df, frequent_cc


def min_sub_mining_asai(
    treebank,
    frequency_counting_strat: FrequencyCountingStrategy,
    k_it,
    min_sup,
    no_pruning=False,
):
    """_summary_

    Returns:
        _type_: _description_
    """

    # We do not pre-prune the activity set
    if no_pruning:
        freq_sup = 0

    else:
        freq_sup = min_sup

    aActivities, dfActivites, ccActivites = compute_frequent_activity_sets_asai(
        treebank, frequency_counting_strat, freq_sup
    )

    if no_pruning:
        ccActivites = aActivities
        dfActivites = aActivities

    F = generate_initial_candidates_asai(
        treebank,
        min_sup,
        frequency_counting_strat,
        aActivities,
        dfActivites,
        ccActivites,
    )

    # Store the results
    k_pattern = {2: F}

    nCandidatesGenerated = len(F)

    # For every k > 2 create the k pattern from the frequent k-1 pattern
    for k in range(k_it):
        newF = []

        for tp in F:
            # Compute the right most path extension of all k-1 pattern
            tps = right_most_path_extension(
                tp, dfActivites, ccActivites, aActivities, no_pruning
            )
            nCandidatesGenerated += len(tps)
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

    return k_pattern, nCandidatesGenerated


def min_sub_mining_asai_memory(
    treebank,
    frequency_counting_strat: FrequencyCountingStrategy,
    k_it,
    min_sup,
    no_pruning=False,
):
    # Fo not pre-prune the activity set, simply compute all activites
    if no_pruning:
        freq_sup = 0

    else:
        freq_sup = min_sup

    aActivities, dfActivites, ccActivites = compute_frequent_activity_sets_asai(
        treebank, frequency_counting_strat, freq_sup
    )

    if no_pruning:
        ccActivites = aActivities
        dfActivites = aActivities

    F = generate_initial_candidates_asai(
        treebank,
        min_sup,
        frequency_counting_strat,
        aActivities,
        dfActivites,
        ccActivites,
    )

    # Store the results
    k_pattern = {2: F}

    nC = 0

    # For every k > 2 create the k pattern from the frequent k-1 pattern
    for k in range(k_it):
        newF = []

        for tp in F:
            # Compute the right most path extension of all k-1 pattern
            tps = right_most_path_extension(
                tp, dfActivites, ccActivites, aActivities, no_pruning
            )
            nC += len(tps)
            sup_to_gain = tp.support

            # del tp.rmo
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

    return k_pattern, nC


def right_most_path_extension(tp, dfActivites, ccActivites, aActivites, no_pruning):
    """ """
    extended_motifes = []

    # The extension_position as an offset in height from the rml/rmo Node in the pattern / tree
    currentNode = tp.rml
    extensionOffset = 0

    while currentNode is not None:
        if currentNode.op:
            extended_motifes.extend(
                extend_node_asai(
                    tp,
                    currentNode,
                    extensionOffset,
                    dfActivites,
                    ccActivites,
                    aActivites,
                    no_pruning,
                )
            )

        currentNode = currentNode.parent
        extensionOffset += 1

    return extended_motifes


def generate_initial_candidates_asai(
    treebank: Mapping[int, TreeBankEntry],
    min_sup: int,
    frequency_counting_strat: FrequencyCountingStrategy,
    aActivities,
    dfActivites,
    ccActivites,
) -> Set[TreePattern]:
    def create_operator_leaf_2_pattern(activity, operator):
        leaf = ConcurrencyTree(None, None, None, activity, None)
        parent = ConcurrencyTree([leaf], None, None, None, operator)

        leaf.parent = parent

        tp = TreePattern(parent, leaf, 0)

        tp.size = 2

        return tp

    C2 = dict()

    # Create the "P" -> "S" and "S" -> "P" pattern
    leaf = ConcurrencyTree(None, None, None, None, cTreeOperator.Concurrent)
    parent = ConcurrencyTree([leaf], None, None, None, cTreeOperator.Sequential)
    leaf.parent = parent
    tp = TreePattern(parent, leaf, 0)
    tp.size = 2

    C2[(cTreeOperator.Sequential, cTreeOperator.Concurrent)] = tp

    leaf = ConcurrencyTree(None, None, None, None, cTreeOperator.Sequential)
    parent = ConcurrencyTree([leaf], None, None, None, cTreeOperator.Concurrent)
    leaf.parent = parent
    tp = TreePattern(parent, leaf, 0)
    tp.size = 2

    C2[(cTreeOperator.Concurrent, cTreeOperator.Sequential)] = tp

    leaf = ConcurrencyTree(None, None, None, None, cTreeOperator.Fallthrough)
    parent = ConcurrencyTree([leaf], None, None, None, cTreeOperator.Concurrent)
    leaf.parent = parent
    tp = TreePattern(parent, leaf, 0)
    tp.size = 2

    C2[(cTreeOperator.Concurrent, cTreeOperator.Fallthrough)] = tp

    leaf = ConcurrencyTree(None, None, None, None, cTreeOperator.Fallthrough)
    parent = ConcurrencyTree([leaf], None, None, None, cTreeOperator.Sequential)
    leaf.parent = parent
    tp = TreePattern(parent, leaf, 0)
    tp.size = 2

    C2[(cTreeOperator.Sequential, cTreeOperator.Fallthrough)] = tp

    # Create the frequent "S" -> Activity pattern
    for activity in dfActivites:
        C2[(cTreeOperator.Sequential, activity)] = create_operator_leaf_2_pattern(
            activity, cTreeOperator.Sequential
        )

    # Create all frequent "P" -> Activity patterns
    for activity in ccActivites:
        C2[(cTreeOperator.Concurrent, activity)] = create_operator_leaf_2_pattern(
            activity, cTreeOperator.Concurrent
        )

    for activity in aActivities:
        C2[(cTreeOperator.Fallthrough, activity)] = create_operator_leaf_2_pattern(
            activity, cTreeOperator.Fallthrough
        )

    def create_rmo_list_entries(
        tid: int, tree: ConcurrencyTree, C2: Mapping[any, TreePattern]
    ):
        if pOp := tree.op:
            # FOr every child
            for child in tree.children:
                # If it is an operator node, set the label accordingly and recurse over its children
                if cT := child.op:
                    create_rmo_list_entries(tid, child, C2)

                # Else simply use the child label in the next step
                else:
                    cT = child.label

                # If the entry exists, add the child to the rmo list as a place of occurence
                if (pOp, cT) in C2:
                    C2[(pOp, cT)].add_rmo((tid, tree.id, child))

    # Parallelizable

    # Compute the RMO list for the 2-Patterns
    for i in treebank:
        tree = treebank[i].tree
        create_rmo_list_entries(i, tree, C2)

    F2 = set()

    # Compute the set of frequent 2-Patterns
    for c in C2:
        pattern = C2[c]
        if check_min_sup(pattern, frequency_counting_strat, treebank, min_sup=min_sup):
            F2.add(pattern)
        else:
            # Clean up
            del pattern

    # Clean up
    del C2

    return F2


def extend_node_asai(
    tp: TreePattern,
    eNode: ConcurrencyTree,
    eOffset: int,
    dfActivites,
    ccActivites,
    aActivites,
    no_pruning,
):
    """
    [summary]

    Args:
        tp (TreePattern): [description]
        eNode (ConcurrencyTree): [description]
        eOffset (int): [description]
        aActivities

    Returns:
        [type]: [description]
    """

    extended_motifes = []

    if no_pruning:
        for activity in aActivites:
            extended_motifes.append(
                extend_motif_on_operator_node(
                    tp, eNode, eOffset, op=None, label=activity
                )
            )

        extended_motifes.append(
            extend_motif_on_operator_node(
                tp, eNode, eOffset, op=cTreeOperator.Concurrent, label=None
            )
        )
        extended_motifes.append(
            extend_motif_on_operator_node(
                tp, eNode, eOffset, op=cTreeOperator.Sequential, label=None
            )
        )
        extended_motifes.append(
            extend_motif_on_operator_node(
                tp, eNode, eOffset, op=cTreeOperator.Fallthrough, label=None
            )
        )

    else:
        if eNode.op == cTreeOperator.Sequential:
            for activity in dfActivites:
                extended_motifes.append(
                    extend_motif_on_operator_node(
                        tp, eNode, eOffset, op=None, label=activity
                    )
                )

            extended_motifes.append(
                extend_motif_on_operator_node(
                    tp, eNode, eOffset, op=cTreeOperator.Concurrent, label=None
                )
            )
            extended_motifes.append(
                extend_motif_on_operator_node(
                    tp, eNode, eOffset, op=cTreeOperator.Fallthrough, label=None
                )
            )

        elif eNode.op == cTreeOperator.Concurrent:
            for activity in ccActivites:
                extended_motifes.append(
                    extend_motif_on_operator_node(
                        tp, eNode, eOffset, op=None, label=activity
                    )
                )

            extended_motifes.append(
                extend_motif_on_operator_node(
                    tp, eNode, eOffset, op=cTreeOperator.Sequential, label=None
                )
            )
            extended_motifes.append(
                extend_motif_on_operator_node(
                    tp, eNode, eOffset, op=cTreeOperator.Fallthrough, label=None
                )
            )

        elif eNode.op == cTreeOperator.Fallthrough:
            for activity in aActivites:
                extended_motifes.append(
                    extend_motif_on_operator_node(
                        tp, eNode, eOffset, op=None, label=activity
                    )
                )

    return extended_motifes
