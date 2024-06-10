import multiprocessing
from typing import List
from pm4py import ProcessTree
from pm4py.objects.conversion.process_tree.variants.to_petri_net import (
    apply as pt_to_pn_converter,
)
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness.variants.alignment_based import (
    apply as fitness_evaluator,
)
import copy
import itertools

from experiments.freezing_subtrees.PoolFactory import PoolFactory
from experiments.freezing_subtrees.utils import (
    add_variants_to_process_model,
)


def get_fitting_and_non_fitting_traces(
    variants_added_so_far, variant_to_be_added, process_tree
):
    fitting_traces = set()
    traces_to_add = set()

    # may be unnecessary, consider simplifying by removing computations
    # concatenate_variants = variants_added_so_far[:]
    # concatenate_variants.append(variant_to_be_added)
    # for selected_variant in concatenate_variants:
    #     if typed_trace_fits_process_tree(selected_variant['typed_trace'], process_tree):
    #         fitting_traces.add(selected_variant['typed_trace'])
    #     else:
    #         traces_to_add.add(selected_variant['typed_trace'])

    for selected_variant in variants_added_so_far:
        fitting_traces.add(selected_variant["typed_trace"])
    traces_to_add.add(variant_to_be_added["typed_trace"])

    return fitting_traces, traces_to_add


def compute_frozen_sub_trees(
    process_tree: ProcessTree,
    variants_added_so_far,
    variant_to_be_added,
    event_log,
    evaluation_criteria,
    max_subtrees,
):
    frozen_subtrees: list[ProcessTree] = []
    resulting_tree: ProcessTree = copy.deepcopy(process_tree)

    fitting_traces, traces_to_add = get_fitting_and_non_fitting_traces(
        variants_added_so_far, variant_to_be_added, process_tree
    )

    if len(variants_added_so_far) > 0:
        (
            performance_compare_value,
            proc_tree_baseline,
        ) = calculate_precision_compare_value(
            process_tree, fitting_traces, traces_to_add, event_log, evaluation_criteria
        )

        freezing_candidates = calculate_set_of_freezing_candidates(
            process_tree,
            fitting_traces,
            traces_to_add,
            event_log,
            performance_compare_value,
            evaluation_criteria,
        )

        if len(freezing_candidates) > 0:
            frozen_subtrees, resulting_tree = apply_sub_tree_rating(
                process_tree,
                freezing_candidates,
                fitting_traces,
                traces_to_add,
                event_log,
                performance_compare_value,
                evaluation_criteria,
                max_subtrees,
            )
        elif len(freezing_candidates) == 0:
            resulting_tree = proc_tree_baseline

    elif len(variants_added_so_far) == 0:
        resulting_tree = add_variants_to_process_model(
            copy.deepcopy(process_tree),
            copy.deepcopy(frozen_subtrees),
            list(fitting_traces),
            list(traces_to_add),
            PoolFactory.instance().get_pool(),
        )

    return (
        frozen_subtrees,
        resulting_tree,
        get_performance(resulting_tree, event_log, "f_measure"),
    )


def calculate_precision_compare_value(
    process_tree: ProcessTree,
    fitting_traces,
    traces_to_add,
    event_log,
    evaluation_criteria,
):
    pt = add_variants_to_process_model(
        copy.deepcopy(process_tree),
        [],
        list(fitting_traces),
        list(traces_to_add),
        PoolFactory.instance().get_pool(),
    )

    return get_performance(pt, event_log, evaluation_criteria), pt


def get_performance(process_tree, event_log, evaluation_criteria):
    fitness = None
    f_measure = None

    pn, initial_marking, final_marking = pt_to_pn_converter(process_tree)

    precision = precision_evaluator.apply(
        event_log,
        pn,
        initial_marking,
        final_marking,
        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE,
    )

    if evaluation_criteria != "precision":
        fitness = fitness_evaluator(event_log, pn, initial_marking, final_marking)
        fitness = fitness["log_fitness"]
        f_measure = 2 * (precision * fitness) / (precision + fitness)

    return {"fitness": fitness, "precision": precision, "f_measure": f_measure}


def calculate_set_of_freezing_candidates(
    process_tree: ProcessTree,
    fitting_traces,
    traces_to_add,
    event_log,
    performance_compare_value,
    evaluation_criteria,
):
    freezing_candidates: List[ProcessTree] = []
    sub_trees = get_all_subtrees(process_tree)

    # calculate precision for each discovered tree
    for potentially_frozen_sub_tree in sub_trees:
        pt = add_variants_to_process_model(
            copy.deepcopy(process_tree),
            copy.deepcopy([potentially_frozen_sub_tree]),
            list(fitting_traces),
            list(traces_to_add),
            PoolFactory.instance().get_pool(),
        )
        current_performance = get_performance(pt, event_log, evaluation_criteria)

        if (
            current_performance[evaluation_criteria]
            > performance_compare_value[evaluation_criteria]
        ):
            freezing_candidates.append(potentially_frozen_sub_tree)

    return freezing_candidates


def calculate_set_of_freezing_candidates_parallel(
    process_tree: ProcessTree,
    fitting_traces,
    traces_to_add,
    event_log,
    performance_compare_value,
    evaluation_criteria,
):
    freezing_candidates: List[ProcessTree] = []
    sub_trees = get_all_subtrees(process_tree)

    pool = multiprocessing.Pool(2)
    second_arg = {
        "process_tree": process_tree,
        "fitting_traces": fitting_traces,
        "traces_to_add": traces_to_add,
        "event_log": event_log,
        "evaluation_criteria": evaluation_criteria,
    }

    if len(sub_trees) > 0:
        frozen_sub_trees = pool.starmap(
            subtree_rating_parallel, zip(sub_trees, itertools.repeat(second_arg))
        )

    # calculate precision for each discovered tree

    return freezing_candidates


def subtree_rating_parallel(sub_tree, second_arg):
    frozen_sub_trees = []

    pt = add_variants_to_process_model(
        copy.deepcopy(second_arg["process_tree"]),
        copy.deepcopy([sub_tree]),
        list(second_arg["fitting_traces)"]),
        list(second_arg["traces_to_add)"]),
        PoolFactory.instance().get_pool(),
    )
    current_performance = get_performance(
        pt, second_arg["event_log"], second_arg["evaluation_criteria"]
    )
    return {
        "frozen_sub_tree": copy.deepcopy(sub_tree),
        "performance": current_performance,
    }

    return frozen_sub_trees


def get_all_subtrees(pt: ProcessTree, sub_trees=None) -> list[ProcessTree]:
    if sub_trees is None:
        sub_trees = []

    if pt.parent is None:
        for c in pt.children:
            if c.operator is not None:
                get_all_subtrees(c, sub_trees)
    elif pt.operator is not None:
        sub_trees.append(pt)
        for c in pt.children:
            if c.operator is not None:
                get_all_subtrees(c, sub_trees)

    return sub_trees


def apply_sub_tree_rating(
    input_process_tree: ProcessTree,
    input_freezing_candidates: List[ProcessTree],
    fitting_traces,
    traces_to_add,
    event_log,
    performance_compare_value,
    evaluation_criteria,
    max_subtrees,
):
    best_freezing_combination = []
    resulting_process_tree = copy.deepcopy(input_process_tree)

    if max_subtrees == -1:
        max_subtrees = len(input_freezing_candidates)

    combinations: List[List[ProcessTree]] = powerset(
        input_freezing_candidates, max_subtrees
    )

    subtree_combinations: List[List[ProcessTree]] = remove_subtree_combination(
        combinations, max_subtrees
    )

    if len(subtree_combinations) > 0:
        max_performance = -100

        # calculate precision for each combination
        for frozen_subtrees_combination in subtree_combinations:
            pt = add_variants_to_process_model(
                copy.deepcopy(input_process_tree),
                copy.deepcopy(frozen_subtrees_combination),
                list(fitting_traces),
                list(traces_to_add),
                PoolFactory.instance().get_pool(),
            )

            current_performance = get_performance(pt, event_log, evaluation_criteria)

            if (
                current_performance[evaluation_criteria]
                > performance_compare_value[evaluation_criteria]
                and current_performance[evaluation_criteria] > max_performance
            ):
                max_performance = current_performance[evaluation_criteria]
                best_freezing_combination = copy.deepcopy(frozen_subtrees_combination)
                resulting_process_tree = copy.deepcopy(pt)

    return best_freezing_combination, resulting_process_tree


def powerset(s, max_length=0):
    combinations = []
    for i in range(max_length):
        for comb in itertools.combinations(s, i + 1):
            combinations.append(list(comb))
    return combinations


# removes combinations that has a higher length then max_length
# or where one subtree is subtree of another subtree in the combination
def remove_subtree_combination(combinations, max_length):
    clean_combs = []

    for combination in combinations:
        del_subtrees = []
        new_comb = []
        for i, subtree in enumerate(combination):
            for j, subtree2 in enumerate(combination):
                if i == j:
                    continue
                if is_subtree(subtree2, subtree):
                    del_subtrees.append(id(subtree))
                    break
        for subtree in combination:
            if not id(subtree) in del_subtrees:
                new_comb.append(subtree)

        clean_combs.append(new_comb)
    unique_combs = []
    for combination in clean_combs:
        if max_length >= len(combination) > 0 and not combination in unique_combs:
            unique_combs.append(combination)
    return unique_combs


def is_subtree(pt, subtree):
    if id(pt) == id(subtree):
        return True
    else:
        for child in pt.children:
            if is_subtree(child, subtree):
                return True
        return False
