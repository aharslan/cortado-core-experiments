import argparse
import copy
from asyncio import sleep

import pm4py
from pm4py import ProcessTree
from tqdm import tqdm
import pickle
from pm4py.visualization.process_tree import visualizer as pt_visualizer

from experiments.freezing_subtrees.freezing_strategies.local_brute_force import (
    compute_frozen_sub_trees,
)
from experiments.freezing_subtrees.setup import (
    import_initial_model,
    get_experiment_variants,
    import_event_log,
)


def execute_experiment(
    process_tree, variants, event_log, output_file, evaluation_criteria, max_subtrees
):
    resulting_tree: ProcessTree = process_tree
    frozen_subtrees: list[ProcessTree] = []
    variants_added_so_far = []
    trace_counter = 1
    total_traces = len(variants)

    pbar = tqdm(variants)

    with open(output_file, "wb") as f:
        for variant in pbar:
            sleep(0.1)
            pbar.set_description(
                "\n\n\n\n**************************************************** ADDING TRACE: "
                + str(trace_counter)
                + " of "
                + str(total_traces)
                + " ****************************************************\n\n\n\n"
            )

            try:
                log_object = {
                    "input_proc_tree": copy.deepcopy(resulting_tree),
                    "frozen_subtrees_in_input_pt": [],
                    "added_variant": variant,
                    "variant_occurrences": variant["count"],
                    "output_proc_tree": {},
                    "f_measure": 0,
                    "fitness": 0,
                    "precision": 0,
                }

                frozen_subtrees, resulting_tree, performance = compute_frozen_sub_trees(
                    resulting_tree,
                    variants_added_so_far,
                    variant,
                    event_log,
                    evaluation_criteria,
                    max_subtrees,
                )
                variants_added_so_far.append(variant)

                log_object["output_proc_tree"] = resulting_tree
                log_object["frozen_subtrees_in_input_pt"] = (
                    copy.deepcopy(frozen_subtrees),
                )
                log_object["fitness"] = performance["fitness"]
                log_object["precision"] = performance["precision"]
                log_object["f_measure"] = performance["f_measure"]
                pickle.dump(log_object, f)

            except:
                print(
                    "\n\n\n\n**************************************************** something went wrong computing "
                    "the frozen subtrees  ****************************************************\n\n\n\n"
                )

            # p_t = pt_visualizer.apply(log_object["input_proc_tree"])
            # pt_visualizer.view(p_t)
            #
            # if len(log_object["frozen_subtrees_in_input_pt"]) > 0:
            #     f_t = pt_visualizer.apply(log_object["frozen_subtrees_in_input_pt"][0])
            #     pt_visualizer.view(f_t)

            trace_counter = trace_counter + 1


parser = argparse.ArgumentParser()
parser.add_argument(
    "--tree",
    default=r"C:\Users\ahmad\OneDrive\Desktop\MS DS\Cortado HiWi\Tasks Artifacts\2. Experiments For Freezing Sub-Trees\experiments\bpi_ch_20_pre_tra_cos\process_tree.ptml",
    type=str,
    help="This is the input process tree",
)
parser.add_argument(
    "--log",
    default=r"C:\Users\ahmad\OneDrive\Desktop\MS DS\Cortado HiWi\Tasks Artifacts\2. Experiments For Freezing Sub-Trees\experiments\bpi_ch_20_pre_tra_cos\PrepaidTravelCost.xes",
    type=str,
    help="This is the input event log",
)
parser.add_argument(
    "--output",
    default=r"C:\Users\ahmad\OneDrive\Desktop\MS DS\Cortado HiWi\Tasks Artifacts\2. Experiments For Freezing Sub-Trees\experiments\bpi_ch_20_pre_tra_cos\pickle.dat",
    type=str,
    help="This is the output log file",
)
parser.add_argument(
    "--evaluation",
    default="precision",
    type=str,
    help="This is the evaluation criteria: fitness, precision or f_measure",
)
parser.add_argument(
    "--limit",
    default="-1",
    type=int,
    help="Limit of trace variants to go through in the iteration -1 = no limit",
)
parser.add_argument(
    "--max_subtrees",
    default="-1",
    type=int,
    help="Limit of maximum subtrees to consider -1 = no limit",
)

if __name__ == "__main__":
    args = parser.parse_args()

    print(pm4py.util.lp.solver.DEFAULT_LP_SOLVER_VARIANT)

    if args.tree:
        process_tree = import_initial_model(args.tree)

    if args.log:
        event_log = import_event_log(args.log)

    if args.output:
        output_file = args.output

    # 'fitness', 'precision' or 'f_measure'
    if args.evaluation:
        evaluation_criteria = args.evaluation

    # get variants sorted by frequency of traces
    variants = get_experiment_variants(event_log)

    if args.limit and args.limit != -1:
        variants = variants[0 : args.limit]

    if args.max_subtrees:
        max_subtrees = args.max_subtrees

    execute_experiment(
        process_tree,
        variants,
        event_log,
        output_file,
        evaluation_criteria,
        max_subtrees,
    )
