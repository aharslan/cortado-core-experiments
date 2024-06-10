import pickle
import timeit

import numpy as np
from cortado_core.utils.cvariants import get_concurrency_variants, get_detailed_variants
from cortado_core.utils.split_graph import LeafGroup, ParallelGroup, SequenceGroup
from pm4py.objects.log.importer.xes.importer import apply as import_xes
from pm4py.objects.log.obj import EventLog
from pm4py import get_variants

# from pm4py.utils import sample_cases
from . import config
from .config import (
    AVAILABLE_EVENT_LOGS,
    EVENT_LOG_FOLDER,
    MEASURE_EXECUTION_TIME_ITERATIONS,
    PICKLE_FILES_DIR,
    RESULT_NUM_CASES_KEY,
    RESULT_NUM_EVENTS_KEY,
    RESULT_NUM_HIGH_LEVEL_VARIANTS_KEY,
    RESULT_NUM_LOW_LEVEL_VARIANTS_KEY,
    RESULT_SUB_VARIANT_COMPUTATION_TIME_KEY,
    RESULT_SUB_VARIANTS_KEY,
    RESULT_VARIANT_COMPUTATION_TIME_KEY,
    RESULT_VARIANTS_KEY,
)

if not config.PERSIST_RESULTS:
    result = {}


def all_zero(timestamps, time_unit_key):
    return all(getattr(timestamp, time_unit_key) == 0 for timestamp in timestamps)


# def get_time_granularity(event_log: EventLog, sample_size):
#     sample = sample_cases(event_log, sample_size)
#     timestamps = [event[DEFAULT_TIMESTAMP_KEY]
#                   for trace in sample for event in trace]

#     if not all_zero(timestamps, 'microsecond'):
#         return TimeUnit.MS
#     elif not all_zero(timestamps, 'second'):
#         return TimeUnit.SEC
#     elif not all_zero(timestamps, 'minute'):
#         return TimeUnit.MIN
#     elif not all_zero(timestamps, 'hour'):
#         return TimeUnit.HOUR
#     elif not all_zero(timestamps, 'day'):
#         return TimeUnit.DAY


def get_sub_variants(variants, time_granularity):
    res_variants = []
    for v in variants:
        variant = {"variant": v.serialize(), "sub_variants": []}
        sub_variants = get_detailed_variants(variants[v], time_granularity)

        for sub_v in sub_variants:
            variant["sub_variants"].append(
                {
                    "variant": sub_v,
                }
            )

        # If the variant is only a single activity leaf, wrap it up as a sequence
        if (
            "leaf" in variant["variant"].keys()
            or "parallel" in variant["variant"].keys()
        ):
            variant["variant"] = {"follows": [variant["variant"]]}

        res_variants.append(variant)

    return res_variants


def get_height_statistics(variants):
    heights = [get_variant_height(variant) for variant in variants]
    return np.mean(heights), max(heights), np.median(heights)


def get_width_statistics(variants):
    variant_widths = [len(variant) for variant in variants]
    return np.mean(variant_widths), max(variant_widths), np.median(variant_widths)


def get_average_variant_height_for_granularities(variants_list):
    avg_heights = []
    for variants in variants_list:
        avg_heights.append(get_height_statistics(variants)[0])
    return avg_heights


def get_average_variant_width_for_granularities(variants_list):
    avg_widths = []
    for variants in variants_list:
        avg_widths.append(get_width_statistics(variants)[0])
    return avg_widths


def get_variant_height(variant):
    if isinstance(variant, SequenceGroup):
        return max([get_variant_height(v) for v in variant])
    elif isinstance(variant, ParallelGroup):
        return sum([get_variant_height(v) for v in variant])

    return len(list(variant))


def get_average_subvariant_height_for_granularities(variants_list):
    avg_heights = []

    for variants in variants_list:
        avg_heights.append(get_subvariant_height_statistic(variants)[0])

    return avg_heights


def get_subvariant_height_statistic(variants):
    subvariant_heights = []

    for variant in variants:
        for subvariant in variant["sub_variants"]:
            subvariant_heights.append(get_subvariant_height(subvariant["variant"]))

    return (
        np.mean(subvariant_heights),
        max(subvariant_heights),
        np.median(subvariant_heights),
    )


def get_subvariant_height(subvariant):
    max_height = 0
    n_active = 0

    for subvariant_parallel_events in subvariant:
        n_started = 0
        n_completed = 0
        for subvariant_node in subvariant_parallel_events:
            if subvariant_node.lifecycle == "start":
                n_started += 1
            elif subvariant_node.lifecycle == "complete":
                n_completed += 1

        n_active += n_started

        if n_active > max_height:
            max_height = n_active

        n_active -= n_completed

    return max_height


def get_average_subvariant_width_for_granularities(variants_list):
    avg_widths = []

    for variants in variants_list:
        avg_widths.append(get_subvariant_width_statistic(variants)[0])

    return avg_widths


def get_subvariant_width_statistic(variants):
    subvariant_widths = []

    for variant in variants:
        for subvariant in variant["sub_variants"]:
            subvariant_widths.append(get_subvariant_width(subvariant["variant"]))

    return (
        np.mean(subvariant_widths),
        max(subvariant_widths),
        np.median(subvariant_widths),
    )


def get_subvariant_width(subvariant):
    return len(subvariant)


def get_num_low_level_variants(variants):
    return np.sum([len(variant["sub_variants"]) for variant in variants])


def load_result():
    if config.PERSIST_RESULTS:
        print("Loading result for " + config.LOG_NAME)
        return load_result_by_log_name(config.LOG_NAME)
    else:
        return result


def load_result_by_log_name(log_name):
    return pickle.load(open(PICKLE_FILES_DIR + log_name + ".p", "rb"))


def save_result(result):
    if config.PERSIST_RESULTS:
        print("Saving result for log " + config.LOG_NAME)
        pickle.dump(result, open(PICKLE_FILES_DIR + config.LOG_NAME + ".p", "wb"))


def measure_execution_time_sub_variant_calculation(variants, time_granularity):
    t = timeit.Timer(
        lambda: get_sub_variants(variants, time_granularity=time_granularity)
    )
    res = t.timeit(MEASURE_EXECUTION_TIME_ITERATIONS)
    return res[0] / MEASURE_EXECUTION_TIME_ITERATIONS


def measure_execution_time_sub_variant_calculation_for_each_granularity():
    times = []
    variants_list = load_result()[RESULT_VARIANTS_KEY]
    for i in range(len(variants_list)):
        times.append(
            measure_execution_time_sub_variant_calculation(
                variants_list[i], list(config.GRANULARITIES)[i]
            )
        )
    return times


def measure_execution_time_variant_calculation(log, use_mp=True):
    times = []
    variants = []
    for unit in config.GRANULARITIES:
        print("Measuring for ")
        print(unit)
        t = timeit.Timer(
            lambda: get_concurrency_variants(log, use_mp=use_mp, time_granularity=unit)
        )
        res = t.timeit(MEASURE_EXECUTION_TIME_ITERATIONS)
        times.append(res[0] / MEASURE_EXECUTION_TIME_ITERATIONS)
        variants.append(res[1])
    return times, variants


def measure_execution_time_plain_variant_calculation(log):
    print(f"Measuring plain variant calculation")
    t = timeit.Timer(lambda: get_variants(log))
    res = t.timeit(MEASURE_EXECUTION_TIME_ITERATIONS)
    return res[0] / MEASURE_EXECUTION_TIME_ITERATIONS


def get_variants_for_each_granularity(use_mp=True):
    print("Get variants for each granularity for log " + config.LOG_NAME)
    log = import_xes(EVENT_LOG_FOLDER + "/" + config.LOG_NAME)
    result = load_result()
    variants_list = []
    for unit in config.GRANULARITIES:
        print("Computing variants for granularity " + unit.value)
        variants = get_concurrency_variants(log, use_mp=use_mp, time_granularity=unit)
        variants_list.append(variants)
    result[RESULT_VARIANTS_KEY] = variants_list

    save_result(result)


def get_sub_variants_for_each_granularity():
    result = load_result()
    variants_with_sub_variants_list = []

    for i in range(len(result[RESULT_VARIANTS_KEY])):
        print(
            "Computing sub-variants for granularity "
            + list(config.GRANULARITIES)[i].value
        )
        res_variants = get_sub_variants(
            result[RESULT_VARIANTS_KEY][i], list(config.GRANULARITIES)[i]
        )
        variants_with_sub_variants_list.append(res_variants)

    result[RESULT_SUB_VARIANTS_KEY] = variants_with_sub_variants_list
    save_result(result)


def most_common_element_frequency(lst):
    return max([lst.count(elem) for elem in lst])


def get_result(key):
    print("Get result for event log" + config.LOG_NAME)
    return load_result(config.LOG_NAME)[key]


def aggregate_results():
    aggregated = {}

    for log_name in AVAILABLE_EVENT_LOGS:
        print(log_name)
        result = load_result_by_log_name(log_name)
        aggregated[log_name] = {
            RESULT_NUM_CASES_KEY: result[RESULT_NUM_CASES_KEY],
            RESULT_NUM_HIGH_LEVEL_VARIANTS_KEY: result[
                RESULT_NUM_HIGH_LEVEL_VARIANTS_KEY
            ],
            RESULT_NUM_EVENTS_KEY: result[RESULT_NUM_EVENTS_KEY],
            RESULT_NUM_LOW_LEVEL_VARIANTS_KEY: result[
                RESULT_NUM_LOW_LEVEL_VARIANTS_KEY
            ],
            RESULT_VARIANT_COMPUTATION_TIME_KEY: result[
                RESULT_VARIANT_COMPUTATION_TIME_KEY
            ],
            RESULT_SUB_VARIANT_COMPUTATION_TIME_KEY: result[
                RESULT_SUB_VARIANT_COMPUTATION_TIME_KEY
            ],
        }

    pickle.dump(aggregated, open(PICKLE_FILES_DIR + "aggregated.p", "wb"))


def load_aggregated_results(log_name):
    return pickle.load(open(PICKLE_FILES_DIR + "aggregated.p", "rb"))[log_name]


def num_times_cuts_could_not_be_applied(
    variant, counter=0, sizes=[], depth=0, limitation_depths=[]
):
    # if there is a Leaf node which contains more than one event, cutting was not possible
    if isinstance(variant, LeafGroup) and len(variant.serialize()["leaf"]) > 1:
        counter += 1
        sizes.append(len(variant.serialize()["leaf"]))
        limitation_depths.append(depth)
    # if we have a parallel or sequence group apply this function recursively to subgroups
    if not isinstance(variant, LeafGroup):
        return (
            sum(
                [
                    num_times_cuts_could_not_be_applied(
                        group, counter, sizes, depth + 1, limitation_depths
                    )[0]
                    for group in variant
                ]
            ),
            sizes,
            limitation_depths,
        )
    # if we have a leaf group return the counter, sizes and depths
    else:
        return counter, sizes, limitation_depths


def cuts_not_possible_statistics(variants):
    counter = 0
    group_sizes = []
    depths = []
    num_variants = len(variants)

    for variant in variants:
        num, sizes, limitation_depths = num_times_cuts_could_not_be_applied(
            variant, sizes=[], limitation_depths=[]
        )
        if num > 0:
            counter += 1
            group_sizes.append(np.mean(sizes))
            depths.append(np.mean(limitation_depths))

    return counter / num_variants, np.mean(group_sizes), np.mean(depths)
