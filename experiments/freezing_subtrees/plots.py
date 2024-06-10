import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Arial"

import numpy as np
import pandas as pd
import glob
import os

from experiments.freezing_subtrees.check_output_log import (
    get_experiment_output_log,
)


def get_percentage_variants(length, total_variants):
    cumulative_percentages = np.zeros([length])
    i = 0
    for x in np.nditer(cumulative_percentages, op_flags=["readwrite"]):
        i += 1
        x[...] = (i / total_variants) * 100

    return cumulative_percentages


def get_percentage_traces(variant_occurrences, total_traces):
    cumulative_percentages = np.zeros([len(variant_occurrences)])
    total_traces_processed = sum(variant_occurrences)
    y = 0
    i = 0

    for x in np.nditer(variant_occurrences, op_flags=["readwrite"]):
        percentage = (x / total_traces) * 100
        cumulative_percentages[i] = y + percentage
        y += percentage
        i += 1

    return cumulative_percentages


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def get_dataframe_from_log_data(data):
    return pd.DataFrame(
        data,
        columns=[
            "input_proc_tree",
            "frozen_subtrees_in_input_pt",
            "added_variant",
            "variant_occurrences",
            "output_proc_tree",
            "f_measure",
            "fitness",
            "precision",
        ],
    )


def plot_f_measure(filename, baseline_filename, title, total_traces, total_variants):
    data1 = get_experiment_output_log(filename)
    data2 = get_experiment_output_log(baseline_filename)

    df1 = get_dataframe_from_log_data(data1)
    df2 = get_dataframe_from_log_data(data2)

    indices_of_frozen = find_indices(
        data1, lambda e: len(e["frozen_subtrees_in_input_pt"][0]) > 0
    )

    f_measures_1 = df1["f_measure"]
    percentage_of_variants_1 = get_percentage_variants(len(data1), total_variants)
    percentage_of_traces_1 = get_percentage_traces(
        np.array(df1["variant_occurrences"]), total_traces
    )

    f_measures_2 = df2["f_measure"]
    percentage_of_variants_2 = get_percentage_variants(len(data2), total_variants)
    percentage_of_traces_2 = get_percentage_traces(
        np.array(df2["variant_occurrences"]), total_traces
    )

    (line_freezing,) = plt.plot(
        percentage_of_variants_1,
        f_measures_1,
        "--bo",
        alpha=0.6,
    )
    line_freezing.set_label("freezing-enabled LCA-IPDA")

    (line_baseline,) = plt.plot(
        percentage_of_variants_2,
        f_measures_2,
        "--go",
        alpha=0.6,
    )
    line_baseline.set_label("LCA-IPDA")

    line_tree_frozen = None
    for index in indices_of_frozen:
        (line_tree_frozen,) = plt.plot(
            percentage_of_variants_1[index], f_measures_1[index], "ro", alpha=0.6
        )
    if line_tree_frozen:
        line_tree_frozen.set_label("Subtree was frozen")

    plt.xlabel("% of processed variants")
    plt.ylabel("F-measure")

    # plt.title(title)

    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig("./plots_output/" + title + "_variants" + ".pdf", bbox_inches="tight")
    plt.clf()

    (line_freezing,) = plt.plot(percentage_of_traces_1, f_measures_1, "--bo", alpha=0.6)
    line_freezing.set_label("With freezing")

    (line_baseline,) = plt.plot(percentage_of_traces_2, f_measures_2, "--go", alpha=0.6)
    line_baseline.set_label("Without freezing")

    line_tree_frozen = None
    for index in indices_of_frozen:
        (line_tree_frozen,) = plt.plot(
            percentage_of_traces_1[index], f_measures_1[index], "ro", alpha=0.6
        )
    if line_tree_frozen:
        line_tree_frozen.set_label("Subtree was frozen")

    plt.xlabel("% of processed traces")
    plt.ylabel("F-measure")

    # plt.title(title)

    plt.grid()
    plt.legend()
    cm = 1 / 2.54
    plt.subplots(figsize=(15 * cm, 5 * cm))
    # plt.show()
    plt.savefig("./plots_output/" + title + "_traces" + ".pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    for filename in glob.iglob("./plots_output/**/*.pdf", recursive=True):
        os.remove(filename)

    plot_f_measure(
        r"./experiments/rtfm/pickle.dat",
        r"./experiments/rtfm/baseline.dat",
        "rtfm",
        150370,
        231,
    )

    plot_f_measure(
        r"./experiments/sepsis/pickle.dat",
        r"./experiments/sepsis/baseline.dat",
        "sepsis",
        1050,
        846,
    )

    plot_f_measure(
        r"./experiments/ccc_20/CoSeLoG_WABO_2.dat",
        r"./experiments/ccc_20/CoSeLoG_WABO_2_Baseline.dat",
        "ccc_20_CoSeLoG_WABO_2",
        1087,
        1032,
    )

    plot_f_measure(
        r"./experiments/ccc_20/CoSeLoG_WABO_3.dat",
        r"./experiments/ccc_20/CoSeLoG_WABO_3_Baseline.dat",
        "ccc_20_CoSeLoG_WABO_3",
        1087,
        1032,
    )

    plot_f_measure(
        r"./experiments/receipt/pickle.dat",
        r"./experiments/receipt/baseline.dat",
        "receipt",
        1434,
        116,
    )

    plot_f_measure(
        r"./experiments/bpi_ch_12/pickle.dat",
        r"./experiments/bpi_ch_12/baseline.dat",
        "bpi_ch_12",
        13087,
        4366,
    )

    plot_f_measure(
        r"./experiments/bpi_ch_20/DomesticDeclarations.dat",
        r"./experiments/bpi_ch_20/domesticDeclarationsBaseline.dat",
        "bpi_ch_20_domestic_declarations",
        10500,
        99,
    )

    plot_f_measure(
        r"./experiments/bpi_ch_20/RequestForPayment.dat",
        r"./experiments/bpi_ch_20/RequestForPaymentBaseline.dat",
        "bpi_ch_20_request_for_payment",
        6886,
        89,
    )

    plot_f_measure(
        r"./experiments/bpi_ch_20/PermitLog.dat",
        r"./experiments/bpi_ch_20/PermitLogBaseline.dat",
        "bpi_ch_20_permit_log",
        7065,
        1478,
    )

    plot_f_measure(
        r"./experiments/bpi_ch_20/PrepaidTravelCost.dat",
        r"./experiments/bpi_ch_20/PrepaidTravelCostBaseline.dat",
        "bpi_ch_20_prepaid_travel_cost",
        2099,
        202,
    )

    plot_f_measure(
        r"./experiments/hospital_billing/pickle.dat",
        r"./experiments/hospital_billing/baseline.dat",
        "hospital_billing",
        100000,
        1020,
    )
