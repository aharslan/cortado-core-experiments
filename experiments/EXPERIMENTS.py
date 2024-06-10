import copy

import os
import random

import pickle
from typing import List

from pm4py.algo.discovery.inductive.algorithm import apply_tree as apply_im_plain
from cortado_core.lca_approach import add_trace_to_pt_language
from pm4py.objects.log.importer.xes import importer as importer
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.statistics.traces.log import case_statistics
from pm4py.evaluation.precision.evaluator import apply as calculate_precision
from pm4py.evaluation.precision.evaluator import Variants as precision_variants
from pm4py.evaluation.replay_fitness.evaluator import apply as calculate_fitness
from pm4py.evaluation.replay_fitness.evaluator import Variants as fitness_variants

from pm4py.objects.conversion.process_tree.converter import apply as pt_to_petri_net
from pm4py.objects.conversion.process_tree.converter import (
    Variants as variant_pt_to_petri_net,
)
from multiprocessing.pool import ThreadPool
import csv
from multiprocessing import Process
from cortado_core.process_tree_utils.miscellaneous import (
    get_number_leaves,
    get_height,
    get_number_silent_leaves,
    get_number_nodes,
)
import pm4py.visualization.process_tree.visualizer as tree_vis


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def experiment(
    log_path: str, log_file_name: str, sort: bool, try_pulling_lca_down: bool
):
    res: List = []
    process_tree_incremental: ProcessTree = None

    log = importer.apply(log_path + log_file_name)

    variants = variants_filter.get_variants(log)
    variants_count = case_statistics.get_variant_statistics(log)
    if sort:
        variants_count = sorted(variants_count, key=lambda x: x["count"], reverse=True)
    else:
        random.shuffle(variants_count)

    # create log that contains for each variant one trace
    variant_event_log = EventLog()
    number_traces_per_variant = []
    for v in variants_count:
        # take first trace of specified variant
        trace = variants[v["variant"]][0]
        variant_event_log.append(trace)
        number_traces_per_variant.append(v["count"])

    assert len(variant_event_log) == len(variants_count)

    log_so_far = EventLog()
    variants_processed_so_far = 0
    traces_processed_so_far = 0

    previous_im_tree: ProcessTree = None
    previous_incremental_tree: ProcessTree = None
    previous_res_im = None
    previous_res_incr = None

    for t in variant_event_log:
        variant_res = {}
        print(
            "    >",
            log_path,
            log_file_name,
            "sort",
            sort,
            "start calculating incremental process tree",
        )
        if not process_tree_incremental:
            e = EventLog()
            e.append(t)
            process_tree_incremental: ProcessTree = apply_im_plain(e, None)
        else:
            process_tree_incremental: ProcessTree = add_trace_to_pt_language(
                process_tree_incremental, log_so_far, t, try_pulling_lca_down
            )
        log_so_far.append(t)
        print("    >", log_path, log_file_name, "sort", sort, "start applying IM", 4)
        process_tree_im: ProcessTree = apply_im_plain(log_so_far, None)

        print(
            "    >",
            log_path,
            log_file_name,
            "sort",
            sort,
            "start calculating f-measure",
            4,
        )

        # calculate f-measure
        pool = ThreadPool(processes=2)
        f_measure_im_thread = None
        f_measure_incr_thread = None

        if not previous_im_tree == process_tree_im:
            print("      calculate f-measure for IM")
            f_measure_im_thread = pool.apply_async(
                calculate_f_measure, (process_tree_im, log)
            )

        if not previous_incremental_tree == process_tree_incremental:
            print("      calculate f-measure for INCR")
            f_measure_incr_thread = pool.apply_async(
                calculate_f_measure, (process_tree_incremental, log)
            )
        else:
            print("NO CALCULATION!!!")
            print(previous_incremental_tree)
            print(process_tree_incremental)

        pool.close()
        pool.join()

        previous_im_tree = copy.deepcopy(process_tree_im)
        previous_incremental_tree = copy.deepcopy(process_tree_incremental)

        if f_measure_im_thread:
            res_im = f_measure_im_thread.get()
            previous_res_im = res_im
        else:
            res_im = previous_res_im
        variant_res["im_fitness"] = res_im["fitness"]["averageFitness"]
        variant_res["im_f_measure"] = res_im["f_measure"]
        variant_res["im_precision"] = res_im["precision"]
        variant_res["im_tree_height"] = get_height(process_tree_im)
        variant_res["im_tree_silent_leaves"] = get_number_silent_leaves(process_tree_im)
        variant_res["im_tree_leaves"] = get_number_leaves(process_tree_im)
        variant_res["im_tree_nodes"] = get_number_nodes(process_tree_im)

        if f_measure_incr_thread:
            res_incr = f_measure_incr_thread.get()
            previous_res_incr = res_incr
        else:
            res_incr = previous_res_incr
        variant_res["incremental_fitness"] = res_incr["fitness"]["averageFitness"]
        variant_res["incremental_f_measure"] = res_incr["f_measure"]
        variant_res["incremental_precision"] = res_incr["precision"]
        variant_res["incremental_tree_height"] = get_height(process_tree_incremental)
        variant_res["incremental_tree_silent_leaves"] = get_number_silent_leaves(
            process_tree_incremental
        )
        variant_res["incremental_tree_leaves"] = get_number_leaves(
            process_tree_incremental
        )
        variant_res["incremental_tree_nodes"] = get_number_nodes(
            process_tree_incremental
        )

        variants_processed_so_far += 1
        traces_processed_so_far += number_traces_per_variant.pop(0)
        variant_res["variants_total"] = len(variants_count)
        variant_res["variants_processed_so_far"] = variants_processed_so_far
        variant_res["traces_processed_so_far"] = traces_processed_so_far
        res.append(variant_res)

        dir_tree_vis = "tree_vis_unsort"
        if sort:
            dir_tree_vis = "tree_vis_sort"
        save_tree_vis(
            process_tree_incremental,
            os.path.join(log_path, dir_tree_vis),
            "tree_variants_processed_"
            + str(variants_processed_so_far)
            + "_INCREMENTAL.svg",
        )
        save_tree_vis(
            process_tree_im,
            os.path.join(log_path, dir_tree_vis),
            "tree_variants_processed_" + str(variants_processed_so_far) + "_IM.svg",
        )

        print(
            log_path,
            log_file_name,
            "sort",
            sort,
            ": variants processed ",
            variants_processed_so_far,
            "/",
            len(variants_count),
        )
    try:
        if sort:
            save_obj(res, log_path + "results_sorted_most_frequent_var")
        else:
            save_obj(res, log_path + "results_unsorted")
    except:
        pass
    # save results to csv file
    keys = res[0].keys()
    if sort:
        filename = "results_sorted_most_frequent_var.csv"
    else:
        filename = "results_unsorted.csv"
    with open(log_path + filename, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(res)


def save_tree_vis(tree, path, filename):
    if not os.path.isdir(path):
        os.makedirs(path)
    tree_vis.save(
        tree_vis.apply(tree, parameters={"format": "svg"}), os.path.join(path, filename)
    )


def calculate_f_measure(pt: ProcessTree, event_log: EventLog):
    net, im, fm = pt_to_petri_net(
        pt, variant=variant_pt_to_petri_net.TO_PETRI_NET_TRANSITION_BORDERED
    )
    pool = ThreadPool(processes=2)
    fitness_thread = pool.apply_async(
        calculate_fitness,
        (event_log, net, im, fm),
        {"variant": fitness_variants.ALIGNMENT_BASED},
    )
    precision_thread = pool.apply_async(
        calculate_precision,
        (event_log, net, im, fm),
        {"variant": precision_variants.ALIGN_ETCONFORMANCE},
    )
    pool.close()
    pool.join()
    fitness = fitness_thread.get()
    precision = precision_thread.get()
    f_measure = (
        2
        * fitness["averageFitness"]
        * precision
        / (fitness["averageFitness"] + precision)
    )
    assert 0 <= f_measure <= 1
    return {"precision": precision, "fitness": fitness, "f_measure": f_measure}


def start_experiments_for_multiple_logs():
    logs = [
        {"log_path": "./logs/sepsis/", "log_file_name": "log.xes"},
        {"log_path": "./logs/hospital_billing/", "log_file_name": "log.xes"},
        {
            "log_path": "./logs/road_traffic_fine_management/",
            "log_file_name": "log.xes",
        },
    ]
    processes: List[Process] = []
    for l in logs:
        processes.append(
            Process(
                target=experiment,
                args=(
                    l["log_path"],
                    l["log_file_name"],
                    True,
                ),
            )
        )
        processes.append(
            Process(
                target=experiment,
                args=(
                    l["log_path"],
                    l["log_file_name"],
                    False,
                ),
            )
        )
    # start processes
    for p in processes:
        p.start()
    # wait for termination of processes
    for p in processes:
        p.join()


if __name__ == "__main__":
    # start_experiments_for_multiple_logs()
    # experiment("./logs/road_traffic_fine_management/", "log.xes", False)
    experiment(
        "logs/road_traffic_fine_management/", "log.xes", True, try_pulling_lca_down=True
    )
