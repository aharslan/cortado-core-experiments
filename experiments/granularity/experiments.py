import sys
import timeit

import numpy as np
from matplotlib import pyplot as plt
from pm4py.objects.log.importer.xes.importer import apply as import_xes
from pm4py.objects.log.obj import EventLog
from pm4py import get_variants

from . import config
from .config import (
    AVAILABLE_EVENT_LOGS,
    EVENT_LOG_FOLDER,
    FIGURES_DIR,
    RESULT_AVG_HEIGHTS_KEY,
    RESULT_AVG_LENGTH_PLAIN_VARIANTS_KEY,
    RESULT_AVG_WIDTH_KEY,
    RESULT_NUM_CASES_KEY,
    RESULT_NUM_EVENTS_KEY,
    RESULT_NUM_HIGH_LEVEL_VARIANTS_KEY,
    RESULT_NUM_LOW_LEVEL_VARIANTS_KEY,
    RESULT_NUM_PLAIN_VARIANTS_KEY,
    RESULT_SUB_VARIANT_COMPUTATION_TIME_KEY,
    RESULT_SUB_VARIANTS_KEY,
    RESULT_VARIANT_COMPUTATION_TIME_KEY,
    RESULT_VARIANTS_KEY,
    RESULT_PLAIN_VARIANT_COMPUTATION_TIME_KEY,
)
from .plot import (
    bar_chart,
    plot_avg_heights,
    plot_avg_widths,
    plot_execution_time,
    plot_num_high_level_conc_variants,
    plot_num_high_level_variants,
    scatter_plot,
    scatter_plot_multiple,
    plot_avg_sub_variant_heights,
    plot_avg_sub_variant_widths,
)
from .util import (
    cuts_not_possible_statistics,
    get_average_variant_height_for_granularities,
    get_average_variant_width_for_granularities,
    get_num_low_level_variants,
    get_result,
    get_sub_variants_for_each_granularity,
    get_variant_height,
    get_subvariant_height,
    get_variants_for_each_granularity,
    load_aggregated_results,
    load_result,
    load_result_by_log_name,
    measure_execution_time_plain_variant_calculation,
    measure_execution_time_sub_variant_calculation_for_each_granularity,
    measure_execution_time_variant_calculation,
    save_result,
    get_average_subvariant_height_for_granularities,
    get_average_subvariant_width_for_granularities,
)


def get_event_log_properties():
    global result
    print(f"Calculating event log properties for {config.LOG_NAME}")
    log = import_xes(EVENT_LOG_FOLDER + "/" + config.LOG_NAME)

    variants = get_variants(log)

    result = {
        RESULT_NUM_CASES_KEY: len(log),
        RESULT_NUM_EVENTS_KEY: sum([len(trace) for trace in log]),
        RESULT_NUM_PLAIN_VARIANTS_KEY: len(variants),
        RESULT_AVG_LENGTH_PLAIN_VARIANTS_KEY: sum(
            [len(variant) for variant in variants]
        )
        / len(variants),
    }

    with open(f"{FIGURES_DIR}{config.LOG_NAME}_properties.txt", "w") as file:
        file.writelines(
            [
                f"Properties of event log {config.LOG_NAME}\n\n",
                f"Number of cases: {result[RESULT_NUM_CASES_KEY]}\n",
                f"Number of events: {result[RESULT_NUM_EVENTS_KEY]}\n",
                f"Number of variants: {result[RESULT_NUM_PLAIN_VARIANTS_KEY]}\n",
                f"Average variant length: {result[RESULT_AVG_LENGTH_PLAIN_VARIANTS_KEY]}\n",
            ]
        )

    save_result(result)


def execution_time_plain_variants_experiment():
    result = load_result()

    log = import_xes(EVENT_LOG_FOLDER + "/" + config.LOG_NAME)
    execution_time = measure_execution_time_plain_variant_calculation(log)

    with open(
        f"{FIGURES_DIR}{config.LOG_NAME}_plain_variant_calc_time.txt", "w"
    ) as file:
        file.write(
            f"Time to get plain variants for event-log {config.LOG_NAME}: {execution_time} seconds"
        )

    result[RESULT_PLAIN_VARIANT_COMPUTATION_TIME_KEY] = execution_time
    save_result(result)


def execution_time_variants_experiment(use_mp=True):
    result = load_result()
    # import log
    log: EventLog = import_xes(EVENT_LOG_FOLDER + "/" + config.LOG_NAME)
    # measure execution time and get variants
    times, variants = measure_execution_time_variant_calculation(log, use_mp=use_mp)
    # plot and save execution time figure
    plot_execution_time(times, config.LOG_NAME)
    result[RESULT_VARIANT_COMPUTATION_TIME_KEY] = times
    save_result(result)


def execution_time_sub_variants_experiment():
    result = load_result()
    # measure execution time of subvariant calc
    times_subvariant_calc = (
        measure_execution_time_sub_variant_calculation_for_each_granularity()
    )
    result[RESULT_SUB_VARIANT_COMPUTATION_TIME_KEY] = times_subvariant_calc
    bar_chart(
        times_subvariant_calc,
        config.LOG_NAME,
        file_name="execution_time_sub_variants",
        x_label="Time unit",
        y_label="Time (s)",
    )
    save_result(result)


def num_high_level_variants_experiment():
    result = load_result()
    num_high_level_variants = [
        len(variants) for variants in result[RESULT_VARIANTS_KEY]
    ]
    result[RESULT_NUM_HIGH_LEVEL_VARIANTS_KEY] = num_high_level_variants
    plot_num_high_level_variants(num_high_level_variants, config.LOG_NAME)
    save_result(result)


def num_low_level_variants_experiment():
    result = load_result()
    num_low_level_variants_for_each_granularity = []
    variants_with_subvariants_list = result[RESULT_SUB_VARIANTS_KEY]

    for variants in variants_with_subvariants_list:
        num_low_level_variants_for_each_granularity.append(
            get_num_low_level_variants(variants)
        )

    bar_chart(
        num_low_level_variants_for_each_granularity,
        config.LOG_NAME,
        file_name="num_low_level_variants",
        x_label="Time unit",
        y_label="num. sub-variants",
    )

    result[
        RESULT_NUM_LOW_LEVEL_VARIANTS_KEY
    ] = num_low_level_variants_for_each_granularity
    save_result(result)


def num_high_level_concurrent_variants_experiment():
    result = load_result()

    num_high_level_conc_variants = []

    for variants in result[RESULT_VARIANTS_KEY]:
        num_parallel_variants = 0
        for variant in variants.keys():
            num_parallel_variants += int(get_variant_height(variant) > 1)
        num_high_level_conc_variants.append(num_parallel_variants)

    result[
        config.RESULT_NUM_HIGH_LEVEL_CONC_VARIANTS_KEY
    ] = num_high_level_conc_variants
    plot_num_high_level_conc_variants(num_high_level_conc_variants, config.LOG_NAME)
    save_result(result)


def num_low_level_concurrent_variants_experiment():
    result = load_result()
    variants_with_subvariants_list = result[RESULT_SUB_VARIANTS_KEY]

    num_low_level_conc_variants_for_each_granularity = []

    for variants in variants_with_subvariants_list:
        num_conc_subvariants = 0
        for variant in variants:
            for subvariant in variant["sub_variants"]:
                num_conc_subvariants += int(
                    get_subvariant_height(subvariant["variant"]) > 1
                )
        num_low_level_conc_variants_for_each_granularity.append(num_conc_subvariants)

    bar_chart(
        num_low_level_conc_variants_for_each_granularity,
        config.LOG_NAME,
        file_name="num_low_level_conc_variants",
        x_label="Time unit",
        y_label="num. sub-variants with concurrency",
    )

    result[
        config.RESULT_NUM_LOW_LEVEL_CONC_VARIANTS_KEY
    ] = num_low_level_conc_variants_for_each_granularity
    save_result(result)


def frequency_distribution_experiment():
    variants_with_subvariants_list = load_result()[RESULT_SUB_VARIANTS_KEY]

    data = [
        [len(variant["sub_variants"]) for variant in subvariants]
        for subvariants in variants_with_subvariants_list
    ]
    flattened_data = [item for sublist in data for item in sublist]

    plt.xticks(np.arange(min(flattened_data), max(flattened_data) + 2))
    plt.hist(
        data,
        alpha=0.7,
        bins=np.arange(min(flattened_data), max(flattened_data) + 2),
        label=[unit.value for unit in list(config.GRANULARITIES)],
        align="left",
    )
    plt.legend(ncol=2)
    plt.savefig(
        FIGURES_DIR
        + config.LOG_NAME
        + "_"
        + "num_sub_variant_per_high_level_variant.pdf"
    )

    # plt.show()


def height_width_of_variants_experiment():
    result = load_result()
    variants_list = result[RESULT_VARIANTS_KEY]
    avg_heights = get_average_variant_height_for_granularities(variants_list)
    avg_widths = get_average_variant_width_for_granularities(variants_list)
    result[RESULT_AVG_HEIGHTS_KEY] = avg_heights
    result[RESULT_AVG_WIDTH_KEY] = avg_widths
    plot_avg_heights(avg_heights, config.LOG_NAME)
    plot_avg_widths(avg_widths, config.LOG_NAME)


def height_width_of_subvariants_experiment():
    result = load_result()

    variants_with_subvariants_list = result[RESULT_SUB_VARIANTS_KEY]

    avg_heights = get_average_subvariant_height_for_granularities(
        variants_with_subvariants_list
    )
    avg_widths = get_average_subvariant_width_for_granularities(
        variants_with_subvariants_list
    )
    result[config.RESULT_SUB_VARIANTS_AVG_HEIGHTS_KEY] = avg_heights
    result[config.RESULT_SUB_VARIANTS_AVG_WIDTHS_KEY] = avg_widths
    plot_avg_sub_variant_heights(avg_heights, config.LOG_NAME)
    plot_avg_sub_variant_widths(avg_widths, config.LOG_NAME)
    save_result(result)


def correlation_cases_execution_time_experiment():
    num_cases_of_logs = [
        load_result_by_log_name(log_name)[RESULT_NUM_CASES_KEY]
        for log_name in AVAILABLE_EVENT_LOGS
    ]

    # calculation_times = [load_result(log_name)[RESULT_VARIANT_COMPUTATION_TIME_KEY]
    #                     for log_name in AVAILABLE_EVENT_LOGS]

    calculation_times = [
        np.random.randint(5, 100, 5) for log_name in AVAILABLE_EVENT_LOGS
    ]

    scatter_plot_multiple(num_cases_of_logs, calculation_times)


def correlation_experiment():
    result = [load_aggregated_results(log_name) for log_name in AVAILABLE_EVENT_LOGS]

    # correlation between the #cases and #low_level_variants
    correlation_between(
        RESULT_NUM_CASES_KEY,
        RESULT_NUM_LOW_LEVEL_VARIANTS_KEY,
        result,
        "correlation_cases_num_sub_variants",
    )
    # correlation between #events and #low_level_variants
    correlation_between(
        RESULT_NUM_EVENTS_KEY,
        RESULT_NUM_LOW_LEVEL_VARIANTS_KEY,
        result,
        "correlation_events_num_subvariants",
    )
    # correlation between #cases and computation time of variants
    correlation_between(
        RESULT_NUM_CASES_KEY,
        RESULT_VARIANT_COMPUTATION_TIME_KEY,
        result,
        "correlation_cases_computation_time",
    )
    # correlation between #events and #high_level_variants
    correlation_between(
        RESULT_NUM_EVENTS_KEY,
        RESULT_NUM_HIGH_LEVEL_VARIANTS_KEY,
        result,
        "correlation_events_num_variants",
    )
    # correlation between #cases and #high_level_variants
    correlation_between(
        RESULT_NUM_CASES_KEY,
        RESULT_NUM_HIGH_LEVEL_VARIANTS_KEY,
        result,
        "correlation_cases_num_variants",
    )
    # correlation between #events and computation time of variants
    correlation_between(
        RESULT_NUM_EVENTS_KEY,
        RESULT_VARIANT_COMPUTATION_TIME_KEY,
        result,
        "correlation_events_computation_time",
    )


def correlation_between(
    x_variable_key, y_variable_key, x_label, y_label, results, file_name
):
    x_data = [[result[x_variable_key]] * 5 for result in results]

    y_data = [result[y_variable_key] for result in results]

    y_max = np.max(y_data) * 1.1

    for unit in list(config.GRANULARITIES):
        scatter_plot(
            x_data,
            y_data,
            y_max,
            x_label,
            y_label,
            file_name=file_name + unit.value,
            granularity=unit,
        )


def cuts_not_possible_experiment():
    variants_for_each_granularity = load_result()[RESULT_VARIANTS_KEY]

    cuts_not_possible = []
    average_sizes = []
    average_depths = []

    for variants in variants_for_each_granularity:
        num_in_percent, average_size, average_depth = cuts_not_possible_statistics(
            variants
        )

        cuts_not_possible.append(num_in_percent)
        average_sizes.append(average_size)
        average_depths.append(average_depth)

    bar_chart(
        cuts_not_possible,
        config.LOG_NAME,
        file_name="cuts_not_possible",
        x_label="Time unit",
        y_label="Variants with limitation (%)",
    )
    bar_chart(
        average_sizes,
        config.LOG_NAME,
        file_name="cuts_not_possible_avg_sizes",
        x_label="Time unit",
        y_label="Avg. sizes",
    )
    bar_chart(
        average_depths,
        config.LOG_NAME,
        file_name="cuts_not_possible_avg_depth",
        x_label="Time unit",
        y_label="Avg. depth",
    )


if __name__ == "__main__":
    timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

config.LOG_NAME = sys.argv[1]

for arg in sys.argv[2:]:
    operation = int(arg)
    print(f"Executing Operation {operation}")

    if operation == 1:
        get_event_log_properties()
    elif operation == 2:
        get_variants_for_each_granularity(use_mp=False)
    elif operation == 3:
        get_sub_variants_for_each_granularity()
    elif operation == 4:
        execution_time_variants_experiment(use_mp=False)
    elif operation == 5:
        execution_time_sub_variants_experiment()
    elif operation == 6:
        num_high_level_variants_experiment()
    elif operation == 7:
        num_low_level_variants_experiment()
    elif operation == 8:
        frequency_distribution_experiment()
    elif operation == 9:
        height_width_of_variants_experiment()
    elif operation == 10:
        correlation_experiment()
    elif operation == 11:
        cuts_not_possible_experiment()
    elif operation == 12:
        height_width_of_subvariants_experiment()
    elif operation == 13:
        num_high_level_concurrent_variants_experiment()
    elif operation == 14:
        num_low_level_concurrent_variants_experiment()
    elif operation == 15:
        execution_time_plain_variants_experiment()
