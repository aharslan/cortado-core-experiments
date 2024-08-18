import argparse
import copy
from asyncio import sleep
import random

import pm4py
from cortado_core.negative_process_model_repair.trace_remover import \
    apply_frequency_rating_based_negative_process_model_repair, \
    apply_sub_tree_rating_based_negative_process_model_repair, \
    apply_complete_brute_force_based_negative_process_model_repair, \
    apply_heuristic_brute_force_based_negative_process_model_repair, apply_negative_process_model_repair

from cortado_core.process_tree_utils.reduction import apply_reduction_rules
from cortado_core.models.infix_type import InfixType
from pm4py import ProcessTree, discover_process_tree_inductive
from pm4py.objects.log.obj import EventLog
from tqdm import tqdm
import pickle
from cortado_core.negative_process_model_repair.constants import Constants

from experiments.negative_process_model_repair.Setup import import_event_log, get_experiment_variants
from experiments.negative_process_model_repair.utils import get_traces_from_variant


def transform(variant):
    return get_traces_from_variant(variant[3])[0]


def transform_1(variant):
    return [variant[0], variant[1], variant[2]]


def execute_experiment_for_all_approaches(
        event_log_title,
        variants,
        output_path,
        total_iterations,
        iteration_sample_size,
        iteration_sampling_method,
        iteration_sampling_method_neg_variant,
        reduction_approach,
        experiment_identifier
):
    log_object = {
        "proc_tree_input": None,
        "proc_tree_post_repair": None,
        "tree_updated": None,
        "sampled_variants_positive": None,
        "sampled_variant_negative": None,
        "approach_used": None,
        "applied_rules": None,
        "positive_variants_conformance_post_repair": None,
        "resulting_tree_edit_distance": None,
        "failed_updates": None,
        "update_operations": None
    }

    output_file_complete_brute_force = \
        (output_path + '\\' + str(total_iterations) + '+' + 'complete_brute_force' + '+' + iteration_sampling_method +
         '+' + str(iteration_sample_size) + '+' + iteration_sampling_method_neg_variant + '+' + event_log_title + '+' +
         str(Constants.MIN_THRESHOLD_POSITIVE_FITTING_VARIANTS) + '+' + str(Constants.MAX_THRESHOLD_TREE_EDIT_DISTANCE)
         + '+' + str(Constants.STOP_WHEN_AN_UPDATE_MEETS_THRESHOLD) + '.dat')
    output_file_heuristic_brute_force = \
        (output_path + '\\' + str(total_iterations) + '+' + 'heuristic_brute_force' + '+' + iteration_sampling_method +
         '+' + str(iteration_sample_size) + '+' + iteration_sampling_method_neg_variant + '+' + event_log_title + '+' +
         str(Constants.MIN_THRESHOLD_POSITIVE_FITTING_VARIANTS) + '+' + str(Constants.MAX_THRESHOLD_TREE_EDIT_DISTANCE)
         + '+' + str(Constants.STOP_WHEN_AN_UPDATE_MEETS_THRESHOLD) + '.dat')
    output_file_sub_tree_rating = \
        (output_path + '\\' + str(total_iterations) + '+' + 'sub_tree_rating' + '+' + iteration_sampling_method +
         '+' + str(iteration_sample_size) + '+' + iteration_sampling_method_neg_variant + '+' + event_log_title + '+' +
         str(Constants.MIN_THRESHOLD_POSITIVE_FITTING_VARIANTS) + '+' + str(Constants.MAX_THRESHOLD_TREE_EDIT_DISTANCE)
         + '+' + str(Constants.STOP_WHEN_AN_UPDATE_MEETS_THRESHOLD) + '.dat')
    output_file_negative_process_model_repair = \
        (output_path + '\\' + str(
            total_iterations) + '+' + 'negative_process_model_repair' + '+' + iteration_sampling_method +
         '+' + str(iteration_sample_size) + '+' + iteration_sampling_method_neg_variant + '+' + event_log_title + '+' +
         str(Constants.MIN_THRESHOLD_POSITIVE_FITTING_VARIANTS) + '+' + str(Constants.MAX_THRESHOLD_TREE_EDIT_DISTANCE)
         + '+' + str(Constants.STOP_WHEN_AN_UPDATE_MEETS_THRESHOLD) + '.dat')

    pbar = tqdm(range(total_iterations))

    sampling_offset = 0
    with (open(output_file_complete_brute_force, "wb") as cbf, open(output_file_heuristic_brute_force, "wb") as hbf,
          open(output_file_sub_tree_rating, "wb") as sr, open(output_file_negative_process_model_repair,
                                                                "wb") as npmr):

        for iteration in pbar:
            try:
                sleep(0.1)

                pbar.set_description(
                    "\n\n\n\n**************************************************** (" + str(
                        experiment_identifier) + ")Executing Iteration: "
                    + str(iteration + 1)
                    + " of "
                    + str(total_iterations)
                    + " ****************************************************\n\n\n\n"
                )

                experiment_variants = []  # [0:(math.floor(0.2 * len(variants)))]

                if iteration_sampling_method == 'random':
                    experiment_variants = random.sample(variants, iteration_sample_size)
                elif iteration_sampling_method == 'shifting_window':
                    experiment_variants = variants[0 + sampling_offset:iteration_sample_size + sampling_offset]
                positive_typed_traces = list(map(transform, experiment_variants))

                experiment_variants = list(map(transform_1, experiment_variants))

                log = EventLog([t.trace for t in positive_typed_traces if t.infix_type == InfixType.NOT_AN_INFIX])
                pt: ProcessTree = discover_process_tree_inductive(log)
                apply_reduction_rules(pt)

                negative_variant = None
                if iteration_sampling_method_neg_variant == 'random':
                    negative_variant = [experiment_variants.pop(random.randint(0, len(experiment_variants) - 1))]
                elif iteration_sampling_method_neg_variant == 'most_frequent':
                    negative_variant = [
                        experiment_variants.pop(max(enumerate(experiment_variants), key=lambda x: x[1][2])[0])]
                elif iteration_sampling_method_neg_variant == 'least_frequent':
                    negative_variant = [
                        experiment_variants.pop(max(enumerate(experiment_variants), key=lambda x: -x[1][2])[0])]

                log_object['proc_tree_input'] = copy.deepcopy(pt)
                log_object['sampled_variants_positive'] = experiment_variants
                log_object['sampled_variant_negative'] = negative_variant

                if reduction_approach == 'complete_brute_force' or reduction_approach == 'all':
                    print(
                        "\n\n\n\n**************************************************** doing complete_brute_force "
                        "  ****************************************************\n\n\n\n"
                    )
                    (log_object['tree_updated'], log_object['proc_tree_post_repair'], log_object['approach_used'],
                     log_object['positive_variants_conformance_post_repair'],
                     log_object['resulting_tree_edit_distance'], log_object['applied_rules'],
                     log_object['failed_updates'], log_object['update_operations']) = (
                        apply_complete_brute_force_based_negative_process_model_repair(copy.deepcopy(pt), negative_variant,
                                                                                       experiment_variants)
                    )
                    pickle.dump(log_object, cbf)

                if reduction_approach == 'heuristic_brute_force' or reduction_approach == 'all':
                    print(
                        "\n\n\n\n**************************************************** doing heuristic_brute_force "
                        "  ****************************************************\n\n\n\n"
                    )
                    (log_object['tree_updated'], log_object['proc_tree_post_repair'], log_object['approach_used'],
                     log_object['positive_variants_conformance_post_repair'],
                     log_object['resulting_tree_edit_distance'], log_object['applied_rules'],
                     log_object['failed_updates'], log_object['update_operations']) = (
                        apply_heuristic_brute_force_based_negative_process_model_repair(copy.deepcopy(pt), negative_variant,
                                                                                        experiment_variants)
                    )
                    pickle.dump(log_object, hbf)

                if reduction_approach == 'sub_tree_rating' or reduction_approach == 'all':
                    print(
                        "\n\n\n\n**************************************************** doing sub_tree_rating "
                        "  ****************************************************\n\n\n\n"
                    )
                    (log_object['tree_updated'], log_object['proc_tree_post_repair'], log_object['approach_used'],
                     log_object['positive_variants_conformance_post_repair'],
                     log_object['resulting_tree_edit_distance'], log_object['applied_rules'],
                     log_object['failed_updates'], log_object['update_operations']) = (
                        apply_sub_tree_rating_based_negative_process_model_repair(copy.deepcopy(pt), negative_variant,
                                                                                  experiment_variants)
                    )
                    pickle.dump(log_object, sr)

                if reduction_approach == 'negative_process_model_repair' or reduction_approach == 'all':
                    print(
                        "\n\n\n\n**************************************************** doing "
                        "negative_process_model_repair"
                        "  ****************************************************\n\n\n\n"
                    )
                    (log_object['tree_updated'], log_object['proc_tree_post_repair'], log_object['approach_used'],
                     log_object['positive_variants_conformance_post_repair'],
                     log_object['resulting_tree_edit_distance'], log_object['applied_rules'],
                     log_object['failed_updates'], log_object['update_operations']) = (
                        apply_negative_process_model_repair(copy.deepcopy(pt), negative_variant, experiment_variants)
                    )
                    pickle.dump(log_object, npmr)

                sampling_offset += 1

            except Exception as e:
                print(
                    "\n\n\n\n**************************************************** something went wrong " + str(e) +
                    "  ****************************************************\n\n\n\n"
                )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--log",
    default=r"C:\Users\ahmad\OneDrive\Desktop\MS_DS\Cortado_HiWi\Tasks_Artifacts\Processes_Event_Logs\bpi_ch_20\PermitLog.xes",
    type=str,
    help="Input event log",
)
parser.add_argument(
    "--output",
    default=r"C:\Users\ahmad\OneDrive\Desktop\Sem6\Thesis\experiments",
    type=str,
    help="Output log save location",
)
parser.add_argument(
    "--total_iterations",
    default="20",
    type=int,
    help="Total number of iterations",
)

parser.add_argument(
    "--iteration_sample_size",
    default="20",
    type=int,
    help="Number of fitting variants in each iteration, minimum value: 2",
)
parser.add_argument(
    "--iteration_sampling_method",
    default="shifting_window",
    type=str,
    help="Sampling method of fitting variants in each iteration: random, shifting_window",
)
parser.add_argument(
    "--iteration_sampling_method_neg_variant",
    default="random",
    type=str,
    help="Sampling method of negative variant from the positive variants sample: random, most_frequent, least_frequent",
)
parser.add_argument(
    "--reduction_approach",
    default="all",
    type=str,
    help="Reduction method to be used in the experiment: all, complete_brute_force, heuristic_brute_force, "
         "sub_tree_rating, negative_process_model_repair",
)
parser.add_argument(
    "--experiment_identifier",
    default="1",
    type=int,
    help="To identify and distinguish between different experiments",
)

if __name__ == "__main__":
    args = parser.parse_args()

    print(pm4py.util.lp.solver.DEFAULT_LP_SOLVER_VARIANT)

    if args.log:
        event_log = import_event_log(args.log)

    if args.output:
        output_file = args.output

    if args.total_iterations:
        total_iterations = args.total_iterations

    if args.iteration_sample_size:
        iteration_sample_size = args.iteration_sample_size

    if args.iteration_sampling_method:
        iteration_sampling_method = args.iteration_sampling_method

    if args.iteration_sampling_method_neg_variant:
        iteration_sampling_method_neg_variant = args.iteration_sampling_method_neg_variant

    if args.reduction_approach:
        reduction_approach = args.reduction_approach

    if args.experiment_identifier:
        experiment_identifier = args.experiment_identifier

    # get variants sorted by frequency of traces
    variants = get_experiment_variants(event_log)

    execute_experiment_for_all_approaches(
        args.log.split('\\')[-1].split('.')[0],
        variants,
        output_file,
        total_iterations,
        iteration_sample_size,
        iteration_sampling_method,
        iteration_sampling_method_neg_variant,
        reduction_approach,
        experiment_identifier
    )
