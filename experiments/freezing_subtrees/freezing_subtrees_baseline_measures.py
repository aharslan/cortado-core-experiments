import argparse
import copy
import pickle
import pm4py
from pm4py import ProcessTree

from experiments.freezing_subtrees.PoolFactory import PoolFactory
from experiments.freezing_subtrees.check_output_log import (
    get_experiment_output_log,
)
from experiments.freezing_subtrees.freezing_strategies.local_brute_force import (
    get_fitting_and_non_fitting_traces,
    get_performance,
)
from experiments.freezing_subtrees.setup import import_event_log
from experiments.freezing_subtrees.utils import (
    add_variants_to_process_model,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_log_file",
    default=r"C:\Users\ahmad\OneDrive\Desktop\pickle.dat",
    type=str,
    help="This is the input pickle log file of main experiment",
)

parser.add_argument(
    "--log",
    default=r"C:\Users\ahmad\OneDrive\Desktop\experiments backup after first finish iter\rtfm\log.xes",
    type=str,
    help="This is the input event log",
)
parser.add_argument(
    "--output",
    default=r"C:\Users\ahmad\OneDrive\Desktop\baseline.dat",
    type=str,
    help="This is the output log file",
)


def is_equal(variant1, variant1_count, log_variants, index):
    if type(variant1) is not tuple:
        return variant1

    found = False
    while not found:
        if (
            variant1 == log_variants[index]["variant"]
            and variant1_count == log_variants[index]["count"]
        ):
            found = True
            break
        else:
            index += 1

    return log_variants[index]


def get_traces_and_variants(variants):
    total_variants = len(variants)
    total_traces = 0
    for v in variants:
        total_traces += v["count"]

    return total_variants, total_traces


if __name__ == "__main__":
    print(pm4py.util.lp.solver.DEFAULT_LP_SOLVER_VARIANT)

    args = parser.parse_args()
    input_log_file = r"./experiments/rtfm/pickle.dat"

    if args.input_log_file:
        input_log_file = args.input_log_file

    if args.log:
        event_log = import_event_log(args.log)
        # log_variants = get_experiment_variants(event_log)
        # get_traces_and_variants(log_variants)

    if args.output:
        output = args.output

    data = get_experiment_output_log(input_log_file)
    input_proc_tree: ProcessTree = data[0]["input_proc_tree"]
    resulting_tree = copy.deepcopy(input_proc_tree)
    fitting_traces = []

    with open(output, "wb") as f:
        i = 0
        for row in data:
            log_object = {
                "input_proc_tree": copy.deepcopy(resulting_tree),
                "frozen_subtrees_in_input_pt": [],
                "added_variant": row["added_variant"],
                "variant_occurrences": row["variant_occurrences"],
                "output_proc_tree": {},
                "f_measure": 0,
                "fitness": 0,
                "precision": 0,
            }

            variant = data[i]["added_variant"]
            # variant = is_equal(row['added_variant'], row['variant_occurrences'], log_variants, i)

            f_traces, traces_to_add = get_fitting_and_non_fitting_traces(
                fitting_traces, variant, resulting_tree
            )

            resulting_tree = add_variants_to_process_model(
                copy.deepcopy(resulting_tree),
                [],
                list(f_traces),
                list(traces_to_add),
                PoolFactory.instance().get_pool(),
            )

            fitting_traces.append(variant)

            performance = get_performance(resulting_tree, event_log, "f_measure")

            log_object["output_proc_tree"] = resulting_tree
            log_object["fitness"] = performance["fitness"]
            log_object["precision"] = performance["precision"]
            log_object["f_measure"] = performance["f_measure"]
            pickle.dump(log_object, f)

            i = i + 1
