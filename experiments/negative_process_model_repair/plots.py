import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Arial"

import numpy as np
import pandas as pd
import glob
import os

import pickle


def get_experiment_output_log(filename: str):
    data2 = []
    infile = open(filename, "rb")
    while 1:
        try:
            data2.append(pickle.load(infile))
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()

    return data2


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
            "proc_tree_input",
            "proc_tree_post_repair",
            "sampled_variants_positive",
            "sampled_variant_negative",
            "approach_used",
            "applied_rules_if_rule_based_approach_used",
            "positive_variants_conformance_post_repair",
            "resulting_tree_edit_distance"
        ],
    )


def plot_f_measure(filename):
    data1 = get_experiment_output_log(filename)

    df1 = get_dataframe_from_log_data(data1)
    df1['index'] = df1.index

    #f_measures_1 = df1["f_measure"]
    # percentage_of_variants_1 = get_percentage_variants(len(data1), total_variants)
    # percentage_of_traces_1 = get_percentage_traces(
    #     np.array(df1["variant_occurrences"]), total_traces
    # )


    (positive_variants_fitness,) = plt.plot(
        df1['index'],
        df1['positive_variants_conformance_post_repair'],
        "--bo",
        alpha=0.6,
    )
    positive_variants_fitness.set_label("random iterations")

    plt.xlabel("iteration")
    plt.ylabel("% positive variants fitness")

    plt.grid()
    plt.legend()
    plt.show()

    (edit_distance,) = plt.plot(
        df1['index'],
        df1['resulting_tree_edit_distance'],
        "--bo",
        alpha=0.6,
    )
    edit_distance.set_label("random iterations")

    plt.xlabel("iteration")
    plt.ylabel("resulting tree edit distance")


    plt.grid()
    plt.legend()
    plt.show()
    # plt.savefig("./plots_output/" + title + "_variants" + ".pdf", bbox_inches="tight")
    plt.clf()




if __name__ == "__main__":
    for filename in glob.iglob("./plots_output/**/*.pdf", recursive=True):
        os.remove(filename)

    plot_f_measure(
        r"C:\Users\ahmad\OneDrive\Desktop\Sem6\Thesis\experiments\pickle_cbf_mf_sw_10_20_RequestForPayment_85_5.dat"
    )

