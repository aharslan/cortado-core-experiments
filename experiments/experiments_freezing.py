import pm4py

import copy
import os
import random
import pickle
from typing import List

from pm4py.algo.discovery.inductive.algorithm import apply_tree as apply_im_plain

from cortado_core.lca_approach import add_trace_to_pt_language
from cortado_core.freezing.apply import add_trace_to_pt_language_with_freezing
from cortado_core.freezing.baseline_approach import (
    add_trace_to_pt_language_with_freezing_baseline_approach,
)

from pm4py.objects.log.importer.xes import importer as importer
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.statistics.traces.log import case_statistics
from pm4py.evaluation.precision.evaluator import apply as calculate_precision
from pm4py.evaluation.precision.evaluator import Variants as precision_variants
from pm4py.evaluation.replay_fitness.evaluator import apply as calculate_fitness
from pm4py.evaluation.replay_fitness.evaluator import Variants as fitness_variants
from pm4py.objects.process_tree.importer.importer import apply as import_pt
from pm4py.objects.conversion.process_tree.converter import apply as pt_to_petri_net
from pm4py.objects.conversion.process_tree.converter import (
    Variants as variant_pt_to_petri_net,
)
from pm4py.objects.process_tree.exporter.exporter import apply as export_pt
from pm4py.objects.petri_net.importer.importer import apply as petri_import
from pm4py.objects.petri_net.exporter.exporter import apply as export_pn
from multiprocessing.pool import Pool, ThreadPool
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
    log_path: str,
    log_file_name: str,
    sort: bool,
    try_pulling_lca_down: bool,
    initial_tree: ProcessTree,
    frozen_trees: List[ProcessTree],
    analyze_trees=False,
):
    print(
        "pm4py.util.lp.solver.DEFAULT_LP_SOLVER_VARIANT",
        pm4py.util.lp.solver.DEFAULT_LP_SOLVER_VARIANT,
    )
    res: List = []
    # load/sort log
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

    # INITIALIZATION
    e = EventLog()
    e.append(variant_event_log[0])
    log_so_far.append(variant_event_log[0])
    pt_im_first_variant: ProcessTree = apply_im_plain(e, None)

    # initial_tree = ProcessTree(operator=Operator.PARALLEL, children=[frozen_trees[0], pt_im_first_variant])
    # pt_im_first_variant.parent = initial_tree
    # frozen_trees[0].parent = initial_tree
    # tree_vis.view(tree_vis.apply(initial_tree, parameters={"format": "svg"}))
    # tree_vis.view(tree_vis.apply(frozen_trees[0], parameters={"format": "svg"}))

    pt_incr: ProcessTree = copy.deepcopy(initial_tree)
    pt_incr_freezing_advanced: ProcessTree = copy.deepcopy(initial_tree)
    frozen_subtrees_incr_freezing_advanced = [pt_incr_freezing_advanced.children[3]]
    print(
        "pt_incr_freezing_advanced:",
        pt_incr_freezing_advanced,
        "frozen_subtrees_incr_freezing_advanced:",
        frozen_subtrees_incr_freezing_advanced,
    )

    pt_incr_freezing_baseline: ProcessTree = copy.deepcopy(initial_tree)
    frozen_subtrees_incr_freezing_baseline = [pt_incr_freezing_baseline.children[3]]
    print(
        "pt_incr_freezing_baseline:",
        pt_incr_freezing_baseline,
        "frozen_subtrees_incr_freezing_baseline:",
        frozen_subtrees_incr_freezing_baseline,
    )

    previous_im_tree: ProcessTree = None
    previous_res_im = None

    previous_pt_incr: ProcessTree = None
    previous_res_incr = None

    previous_pt_incr_freezing_advanced: ProcessTree = None
    previous_res_incr_freezing_advanced = None

    previous_pt_incr_freezing_baseline: ProcessTree = None
    previous_res_incr_freezing_baseline = None

    for i, t in enumerate(variant_event_log):
        print(">", str(i), "/", str(len(variant_event_log)), "<")
        variant_res = {}

        print(">>", log_path, log_file_name, "calculating incremental process tree")
        pt_incr: ProcessTree = add_trace_to_pt_language(
            pt_incr, log_so_far, t, try_pulling_lca_down
        )

        print(
            ">>",
            log_path,
            log_file_name,
            "calculating incremental process tree freezing (baseline)",
        )
        (
            pt_incr_freezing_baseline,
            frozen_subtrees_incr_freezing_baseline,
        ) = add_trace_to_pt_language_with_freezing_baseline_approach(
            pt_incr_freezing_baseline,
            frozen_subtrees_incr_freezing_baseline,
            log_so_far,
            t,
        )

        print(
            ">>",
            log_path,
            log_file_name,
            "calculating incremental process tree freezing (advanced)",
        )
        p_t = []
        for a in t:
            p_t.append(a["concept:name"])
        print("next trace", p_t)
        (
            pt_incr_freezing_advanced,
            frozen_subtrees_incr_freezing_advanced,
        ) = add_trace_to_pt_language_with_freezing(
            pt_incr_freezing_advanced,
            frozen_subtrees_incr_freezing_advanced,
            log_so_far,
            t,
        )

        log_so_far.append(t)
        print(">>", log_path, log_file_name, "calculating process tree IM")
        pt_im: ProcessTree = apply_im_plain(log_so_far)

        # save trees as pictures and ptml files
        dir_tree_vis = "tree_visualizations"
        dir_tree_ptml = "tree_ptml_pnml/"

        save_tree_vis(
            pt_incr,
            os.path.join(log_path, dir_tree_vis),
            "tree_variants_processed_" + str(variants_processed_so_far) + "_INCR.svg",
        )
        save_tree_as_petri_net(
            pt_incr,
            os.path.join(log_path, dir_tree_ptml),
            "tree_variants_processed_" + str(variants_processed_so_far) + "_INCR",
        )

        save_tree_vis(
            pt_im,
            os.path.join(log_path, dir_tree_vis),
            "tree_variants_processed_" + str(variants_processed_so_far) + "_IM.svg",
        )
        save_tree_as_petri_net(
            pt_im,
            os.path.join(log_path, dir_tree_ptml),
            "tree_variants_processed_" + str(variants_processed_so_far) + "_IM",
        )

        save_tree_vis(
            pt_incr_freezing_advanced,
            os.path.join(log_path, dir_tree_vis),
            "tree_variants_processed_"
            + str(variants_processed_so_far)
            + "_INC_FREEZING_ADVANCED.svg",
        )
        save_tree_as_petri_net(
            pt_incr_freezing_advanced,
            os.path.join(log_path, dir_tree_ptml),
            "tree_variants_processed_"
            + str(variants_processed_so_far)
            + "_INC_FREEZING_ADVANCED",
        )

        save_tree_vis(
            pt_incr_freezing_baseline,
            os.path.join(log_path, dir_tree_vis),
            "tree_variants_processed_"
            + str(variants_processed_so_far)
            + "_INC_FREEZING_BASELINE.svg",
        )
        save_tree_as_petri_net(
            pt_incr_freezing_baseline,
            os.path.join(log_path, dir_tree_ptml),
            "tree_variants_processed_"
            + str(variants_processed_so_far)
            + "_INC_FREEZING_BASELINE",
        )

        # calculate f-measure
        print(">>", log_path, log_file_name, "calculating f-measure")
        pool = Pool()
        f_measure_im_thread = None
        f_measure_incr_thread = None
        f_measure_incr_freezing_baseline_thread = None
        f_measure_incr_freezing_advanced_thread = None

        if analyze_trees:
            if previous_im_tree is None or previous_im_tree != pt_im:
                print(">>", "calculate f-measure for IM")
                f_measure_im_thread = pool.apply_async(
                    calculate_f_measure, (pt_im, log)
                )

            if previous_pt_incr is None or previous_pt_incr != pt_incr:
                print(">>", "calculate f-measure for INCR")
                f_measure_incr_thread = pool.apply_async(
                    calculate_f_measure, (pt_incr, log)
                )

            if (
                previous_pt_incr_freezing_baseline is None
                or previous_pt_incr_freezing_baseline != pt_incr_freezing_baseline
            ):
                print(">>", "calculate f-measure for INCR Frozen Baseline")
                f_measure_incr_freezing_baseline_thread = pool.apply_async(
                    calculate_f_measure, (pt_incr_freezing_baseline, log)
                )

            if (
                previous_pt_incr_freezing_advanced is None
                or previous_pt_incr_freezing_advanced != pt_incr_freezing_advanced
            ):
                print(">>", "calculate f-measure for INCR Frozen Advanced")
                f_measure_incr_freezing_advanced_thread = pool.apply_async(
                    calculate_f_measure, (pt_incr_freezing_advanced, log)
                )
        pool.close()
        pool.join()

        previous_im_tree = copy.deepcopy(pt_im)
        previous_pt_incr = copy.deepcopy(pt_incr)
        previous_pt_incr_freezing_baseline = copy.deepcopy(pt_incr_freezing_baseline)
        previous_pt_incr_freezing_advanced = copy.deepcopy(pt_incr_freezing_advanced)

        if f_measure_im_thread:
            res_im = f_measure_im_thread.get()
            previous_res_im = res_im
        else:
            res_im = previous_res_im

        if analyze_trees:
            variant_res["im_fitness"] = res_im["fitness"]["averageFitness"]
            variant_res["im_f_measure"] = res_im["f_measure"]
            variant_res["im_precision"] = res_im["precision"]
            variant_res["im_tree_height"] = get_height(pt_im)
            variant_res["im_tree_silent_leaves"] = get_number_silent_leaves(pt_im)
            variant_res["im_tree_leaves"] = get_number_leaves(pt_im)
            variant_res["im_tree_nodes"] = get_number_nodes(pt_im)

        if f_measure_incr_thread:
            res_incr = f_measure_incr_thread.get()
            previous_res_incr = res_incr
        else:
            res_incr = previous_res_incr

        if analyze_trees:
            variant_res["incremental_fitness"] = res_incr["fitness"]["averageFitness"]
            variant_res["incremental_f_measure"] = res_incr["f_measure"]
            variant_res["incremental_precision"] = res_incr["precision"]
            variant_res["incremental_tree_height"] = get_height(pt_incr)
            variant_res["incremental_tree_silent_leaves"] = get_number_silent_leaves(
                pt_incr
            )
            variant_res["incremental_tree_leaves"] = get_number_leaves(pt_incr)
            variant_res["incremental_tree_nodes"] = get_number_nodes(pt_incr)

        if f_measure_incr_freezing_baseline_thread:
            res_incr_freezing_baseline = f_measure_incr_freezing_baseline_thread.get()
            previous_res_incr_freezing_baseline = res_incr_freezing_baseline
        else:
            res_incr_freezing_baseline = previous_res_incr_freezing_baseline

        if analyze_trees:
            variant_res[
                "incremental_freezing_baseline_fitness"
            ] = res_incr_freezing_baseline["fitness"]["averageFitness"]
            variant_res[
                "incremental_freezing_baseline_f_measure"
            ] = res_incr_freezing_baseline["f_measure"]
            variant_res[
                "incremental_freezing_baseline_precision"
            ] = res_incr_freezing_baseline["precision"]
            variant_res["incremental_freezing_baseline_tree_height"] = get_height(
                pt_incr_freezing_baseline
            )
            variant_res[
                "incremental_freezing_baseline_tree_silent_leaves"
            ] = get_number_silent_leaves(pt_incr_freezing_baseline)
            variant_res[
                "incremental_freezing_baseline_tree_leaves"
            ] = get_number_leaves(pt_incr_freezing_baseline)
            variant_res["incremental_freezing_baseline_tree_nodes"] = get_number_nodes(
                pt_incr_freezing_baseline
            )

        if f_measure_incr_freezing_advanced_thread:
            res_incr_freezing_advanced = f_measure_incr_freezing_advanced_thread.get()
            previous_res_incr_freezing_advanced = res_incr_freezing_advanced
        else:
            res_incr_freezing_advanced = previous_res_incr_freezing_advanced

        if analyze_trees:
            variant_res[
                "incremental_freezing_advanced_fitness"
            ] = res_incr_freezing_advanced["fitness"]["averageFitness"]
            variant_res[
                "incremental_freezing_advanced_f_measure"
            ] = res_incr_freezing_advanced["f_measure"]
            variant_res[
                "incremental_freezing_advanced_precision"
            ] = res_incr_freezing_advanced["precision"]
            variant_res["incremental_freezing_advanced_tree_height"] = get_height(
                pt_incr_freezing_advanced
            )
            variant_res[
                "incremental_freezing_advanced_tree_silent_leaves"
            ] = get_number_silent_leaves(pt_incr_freezing_advanced)
            variant_res[
                "incremental_freezing_advanced_tree_leaves"
            ] = get_number_leaves(pt_incr_freezing_advanced)
            variant_res["incremental_freezing_advanced_tree_nodes"] = get_number_nodes(
                pt_incr_freezing_advanced
            )

        variants_processed_so_far += 1
        traces_processed_so_far += number_traces_per_variant.pop(0)

        if analyze_trees:
            variant_res["variants_total"] = len(variants_count)
            variant_res["variants_processed_so_far"] = variants_processed_so_far
            variant_res["traces_processed_so_far"] = traces_processed_so_far
            res.append(variant_res)

            try:
                if sort:
                    save_obj(res, log_path + "results_sorted_most_frequent_var")
                else:
                    save_obj(res, log_path + "results_unsorted")
            except:
                pass
            # save results to csv file
            # keys = res[0].keys()
            if sort:
                filename = "results_sorted_most_frequent_var.csv"
            else:
                filename = "results_unsorted.csv"
            # with open(log_path + filename, 'w') as output_file:
            #     dict_writer = csv.DictWriter(output_file, keys)
            #     dict_writer.writeheader()
            #     dict_writer.writerows(res)
            save_list_of_dicts_as_csv(res, filename, log_path)


def save_list_of_dicts_as_csv(l: List, filename: str, log_path: str):
    # save results to csv file
    keys = l[0].keys()
    with open(log_path + filename, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(l)


def analyze_trees_from_file(log, dir_path, indices):
    res = []
    pool = Pool(processes=16)
    approaches = ["IM", "INC_FREEZING_ADVANCED", "INC_FREEZING_BASELINE", "INCR"]
    processes = {}
    for i in indices:
        for a in approaches:
            net, im, fm = petri_import(
                os.path.join(
                    dir_path, "tree_variants_processed_" + str(i) + "_" + a + ".ptml"
                )
            )
            process = pool.apply_async(calculate_f_measure_from_net, (net, im, fm, log))
            processes[(i, a)] = process
    pool.join()
    pool.close()

    for i in indices:
        res.append({})

    for p in processes:
        f_measure_res = processes[p].get()
        res[p[0]][p[1] + "_fitness"] = f_measure_res["fitness"]["averageFitness"]
        res[p[0]][p[1] + "_precision"] = f_measure_res["precision"]
        res[p[0]][p[1] + "_f_measure"] = f_measure_res["f_measure"]
    save_list_of_dicts_as_csv(res, "f_measure_results.csv", dir_path)


def save_tree_vis(tree, path, filename):
    if not os.path.isdir(path):
        os.makedirs(path)
    tree_vis.save(
        tree_vis.apply(tree, parameters={"format": "svg"}), os.path.join(path, filename)
    )


def save_tree_as_petri_net(tree, path, filename):
    if not os.path.isdir(path):
        os.makedirs(path)
    # export_pt(tree, path + filename)
    net, im, fm = pt_to_petri_net(tree)
    export_pn(net, im, path + filename + ".pnml", fm)
    tree_copy = copy.deepcopy(tree)
    export_pt(tree_copy, path + filename + ".ptml")


def calculate_f_measure(pt: ProcessTree, event_log: EventLog):
    print(pt)
    net, im, fm = pt_to_petri_net(pt, variant=variant_pt_to_petri_net.TO_PETRI_NET)
    return calculate_f_measure_from_net(net, im, fm, event_log)


def calculate_f_measure_from_net(net, im, fm, event_log: EventLog):
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
    loaded_tree: ProcessTree = import_pt(
        "logs/road_traffic_fine_management/initial_tree_rtfm_freeze_children_index_3.ptml"
    )
    # loaded_subtree = loaded_tree.children[1]
    # tau = ProcessTree()
    # choice_root = ProcessTree(operator=Operator.XOR, children=[tau, loaded_subtree])
    # loaded_subtree.parent = choice_root
    # tau.parent = choice_root
    # tree_vis.view(tree_vis.apply(choice_root, parameters={"format": "svg"}))

    experiment(
        "logs/road_traffic_fine_management/",
        "log.xes",
        True,
        try_pulling_lca_down=True,
        initial_tree=loaded_tree,
        frozen_trees=[loaded_tree.children[3]],
        analyze_trees=False,
    )
