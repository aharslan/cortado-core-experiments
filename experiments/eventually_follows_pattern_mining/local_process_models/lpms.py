import os
import pickle
import random
from multiprocessing import Pool
from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
from cortado_core.eventually_follows_pattern_mining.local_process_models.clustering.edit_distance_agglomerative_clusterer import (
    EditDistanceAgglomerativeClusterer,
)
from matplotlib import pyplot as plt
from pm4py.objects.log.importer.xes.importer import apply as xes_import
from pm4py.objects.log.util.interval_lifecycle import to_interval
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.process_tree.obj import ProcessTree

from cortado_core.eventually_follows_pattern_mining.algorithm import (
    generate_eventually_follows_patterns_from_groups,
    Algorithm,
)
from cortado_core.eventually_follows_pattern_mining.local_process_models.clustering.distance_matrix import (
    calculate_distance_matrix,
)
from cortado_core.eventually_follows_pattern_mining.local_process_models.clustering.edit_dist_aggl_with_preclustering import (
    EditDistanceAgglomerativeClustererWithPreclustering,
)
from cortado_core.eventually_follows_pattern_mining.local_process_models.clustering.label_vector_clusterer import (
    LabelVectorClusterer,
)
from cortado_core.eventually_follows_pattern_mining.local_process_models.discovery.inductive_miner import (
    InductiveMiner,
)
from cortado_core.eventually_follows_pattern_mining.local_process_models.lpm_discoverer import (
    LpmDiscoverer,
)
from cortado_core.eventually_follows_pattern_mining.local_process_models.lpm_discoverer_load_external_models import (
    LpmDiscovererLoadExternalModels,
)
from cortado_core.eventually_follows_pattern_mining.local_process_models.metrics import (
    calculate_metrics,
)
from cortado_core.eventually_follows_pattern_mining.local_process_models.similarity.causal_footprint_similarity import (
    similarity_score_model_lists,
    calculate_footprints_for_models,
    combine_footprints,
)
from cortado_core.eventually_follows_pattern_mining.util.pattern import flatten_patterns
from experiments.eventually_follows_pattern_mining.util import (
    get_support_count,
)
from cortado_core.utils.cvariants import get_concurrency_variants

from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)

REL_SUPPORT = float(os.getenv("MIN_REL_SUP", "0.1"))
LOG_FILE = os.getenv("LOG_FILE", "BPI_Ch_2020_PrepaidTravelCost.xes")
EVENT_LOG_DIRECTORY = os.getenv(
    "EVENT_LOG_DIRECTORY", "C:\\sources\\arbeit\\cortado\\event-logs"
)
RESULT_DIRECTORY = os.getenv(
    "RESULT_DIRECTORY",
    "C:\\sources\\arbeit\\cortado\\master_thesis\\Master_Thesis_Results_new\\lpm_results",
)
FREQ_COUNT_STRAT = FrequencyCountingStrategy(
    int(os.getenv("FREQ_COUNT_STRAT", FrequencyCountingStrategy.TraceTransaction.value))
)
COMPARE_APPROACHES = ["Context-Rich LPMS", "Tax", "Place Nets"]
MATRIX_THRESHOLD = 200
RANDOM_SAMPLE_SIZE = 500
# When renaming alg names -> also rename checks in code below
ALGORITHMS = {
    "Context-Rich LPMS": lambda _, a, rel_support: LpmDiscovererLoadExternalModels(
        os.path.join(RESULT_DIRECTORY, "external_lpms", "context_rich", LOG_FILE)
    ),
    "Tax": lambda _, a, rel_support: LpmDiscovererLoadExternalModels(
        os.path.join(RESULT_DIRECTORY, "external_lpms", "tax", LOG_FILE)
    ),
    "Place Nets": lambda _, a, rel_support: LpmDiscovererLoadExternalModels(
        os.path.join(RESULT_DIRECTORY, "external_lpms", "viki", LOG_FILE)
    ),
    "Edit Distance 2": lambda _, d, rel: LpmDiscoverer(
        EditDistanceAgglomerativeClusterer(max_distance=2, distance_matrix=d),
        InductiveMiner(),
    ),
    "Edit Distance 3": lambda _, d, rel: LpmDiscoverer(
        EditDistanceAgglomerativeClusterer(max_distance=3, distance_matrix=d),
        InductiveMiner(),
    ),
    "Label Vector 1_3": lambda n_patterns, d, rel: LpmDiscoverer(
        LabelVectorClusterer(n_clusters=max(1, n_patterns // 3)), InductiveMiner()
    ),
    "Label Vector 1_2": lambda n_patterns, d, rel: LpmDiscoverer(
        LabelVectorClusterer(n_clusters=max(1, n_patterns // 2)), InductiveMiner()
    ),
}
sns.set(style="ticks", rc={"figure.figsize": (5, 3.5)})

COLUMNS = [
    "Algorithm",
    "Patterns_name",
    "Relative Support",
    "Support",
    "Support_trans",
    "Support_occ",
    "Confidence",
    "Precision",
    "Coverage",
    "Simplicity",
    "Transitions",
    "Skip Precision",
    "Mean_range",
    "Min_range",
    "Max_range",
    "Model",
    "Cluster",
]

processed_chunks = 0
number_chunks = 0


def build_local_process_models():
    if FREQ_COUNT_STRAT != FrequencyCountingStrategy.TraceTransaction:
        return
    log_dir = EVENT_LOG_DIRECTORY
    log_filename = LOG_FILE
    frequency_strategy = FREQ_COUNT_STRAT
    log = xes_import(os.path.join(log_dir, log_filename))
    log = to_interval(log)
    n_traces = len(log)
    variants = get_concurrency_variants(log)
    n_variants = len(variants)
    save_dir = os.path.join(
        RESULT_DIRECTORY,
        os.path.splitext(log_filename)[0],
        "local_process_models",
        "data",
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = []
    support_count = get_support_count(
        REL_SUPPORT, frequency_strategy, n_traces, n_variants
    )
    all_patterns = generate_eventually_follows_patterns_from_groups(
        variants,
        support_count,
        frequency_strategy,
        Algorithm.InfixPatternCombinationEnumerationGraph,
    )
    all_patterns = flatten_patterns(all_patterns)

    if len(all_patterns) > RANDOM_SAMPLE_SIZE:
        all_patterns = random.sample(all_patterns, RANDOM_SAMPLE_SIZE)

    dist_all = calculate_distance_matrix(all_patterns)

    for alg_name, algorithm in ALGORITHMS.items():
        pipeline = algorithm(len(all_patterns), dist_all, REL_SUPPORT)
        models = pipeline.discover_lpms(all_patterns)

        is_place_net_algorithm = alg_name == "Place Nets"

        if alg_name == "Context-Rich LPMS":
            models = __remove_plus_postfix_from_models(models)

        with open(
            os.path.join(save_dir, f"models_{alg_name}_{REL_SUPPORT}.pkl"), "wb"
        ) as file:
            pickle.dump(models, file)

        pool = Pool()
        processes = []

        global processed_chunks, number_chunks
        processed_chunks = 0
        number_chunks = len(models) // 20

        for models_chunk in split_list_in_chunks(models, 20):
            p = pool.apply_async(
                calculate_metrics_for_models,
                args=(
                    models_chunk,
                    log,
                    alg_name,
                    "all",
                    REL_SUPPORT,
                    is_place_net_algorithm,
                ),
                callback=print_progress_on_console,
            )
            processes.append(p)

        pool.close()
        pool.join()
        for p in processes:
            results = results + p.get()

    df = pd.DataFrame(results, columns=COLUMNS)
    data_filename = os.path.join(
        save_dir, f"{frequency_strategy.name}_support_{REL_SUPPORT}.csv"
    )
    df.to_csv(data_filename)
    generate_plots(log_filename, data_filename, REL_SUPPORT)

    # generate_model_similarity_plots(save_dir)


def print_progress_on_console(res):
    global processed_chunks, number_chunks
    processed_chunks += 1
    print(processed_chunks, "/", number_chunks)


def calculate_metrics_for_models(
    models, log, alg_name, patterns_name, rel_support, is_place_net_algorithm
):
    results = []
    for model in models:
        (
            support_tax,
            support_trans,
            support_occ,
            confidence,
            precision,
            coverage,
            simplicity,
            n_transitions,
            skip_precision,
            mean_range,
            min_range,
            max_range,
        ) = calculate_metrics(
            model[0], log, is_place_net_algorithm=is_place_net_algorithm
        )
        results.append(
            [
                alg_name,
                patterns_name,
                rel_support,
                support_tax,
                support_trans,
                support_occ,
                confidence,
                precision,
                coverage,
                simplicity,
                n_transitions,
                skip_precision,
                mean_range,
                min_range,
                max_range,
                str(model[0]) if isinstance(model[0], ProcessTree) else "",
                repr(model[1]),
            ]
        )

    return results


def split_list_in_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        new_chunk = lst[i : i + chunk_size]
        if len(new_chunk) > 0:
            yield new_chunk


def generate_plots(log_filename, data_filename, rel_support):
    df = pd.read_csv(data_filename)
    plot_dir = os.path.join(
        RESULT_DIRECTORY,
        os.path.splitext(log_filename)[0],
        "local_process_models",
        "plots",
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    generate_plot(
        plot_dir,
        df,
        f"lpm_metrics_{FREQ_COUNT_STRAT.name}_support_{rel_support}.pdf",
        f"LPM metrics: {log_filename}, support = {rel_support}",
    )

    generate_plot_for_metric(
        "Mean_range",
        df,
        f"lpm_metrics_{FREQ_COUNT_STRAT.name}_support_{rel_support}_range.pdf",
        "",
        "Range",
    )


def change_alg_name(row):
    if row["Algorithm"] in COMPARE_APPROACHES:
        return row
    row["Algorithm"] = row["Algorithm"] + " (" + row["Patterns_name"] + ")"
    return row


def generate_plot(plot_dir, plot_df, filename, title):
    # plot_df = plot_df.apply(change_alg_name, axis=1)
    plot_df["Algorithm"] = plot_df["Algorithm"].map(
        {
            "Place Nets": "Peeva et al.",
            "Context-Rich LPMS": "Brunings et al.",
            "Tax": "Tax et al.",
            "Label Vector 1_3": "Label Vector 1/3",
            "Label Vector 1_2": "Label Vector 1/2",
            "Edit Distance 2": "Edit Distance 2",
            "Edit Distance 3": "Edit Distance 3",
        }
    )
    plot_df = pd.melt(
        plot_df,
        id_vars=["Algorithm"],
        value_vars=["Support", "Confidence", "Precision", "Coverage", "Skip Precision"],
    )

    hue_order = [
        "Tax et al.",
        "Peeva et al.",
        "Brunings et al.",
        "Edit Distance 2",
        "Edit Distance 3",
        "Label Vector 1/2",
        "Label Vector 1/3",
    ]

    plot = sns.boxplot(
        data=plot_df, x="variable", y="value", hue="Algorithm", hue_order=hue_order
    )

    legend_handles = plot.axes.get_legend_handles_labels()
    plt.legend([], [], frameon=False)
    # plt.title(title)
    # plt.legend(bbox_to_anchor=(0, 1.42), loc=2, borderaxespad=0., ncol=1)
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches="tight")
    plt.close()

    legend_fig = plt.figure()
    legend_fig.legend(legend_handles[0], legend_handles[1], ncol=4)
    legend_fig.savefig(os.path.join(plot_dir, f"lpm_legend.pdf"), bbox_inches="tight")
    plt.close(legend_fig)


def generate_plot_for_metric(metric: str, df, filename, title, metric_name: str):
    plot_df = df[["Algorithm", metric]]
    # plot_df = plot_df.apply(change_alg_name, axis=1)
    plot_dir = os.path.join(
        RESULT_DIRECTORY, os.path.splitext(LOG_FILE)[0], "local_process_models", "plots"
    )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    hue_order = [
        "Tax et al.",
        "Peeva et al.",
        "Brunings et al.",
        "Edit Distance 2",
        "Edit Distance 3",
        "Label Vector 1/2",
        "Label Vector 1/3",
    ]
    sns.boxplot(data=plot_df, x="Algorithm", y=metric, order=hue_order)
    # plt.title(title)
    plt.xlabel("Algorithm")
    plt.ylabel(metric_name)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches="tight")
    plt.close()


# def generate_model_similarity_plots(save_dir):
#     plot_dir = os.path.join(RESULT_DIRECTORY, os.path.splitext(LOG_FILE)[0], 'local_process_models', 'plots')
#
#     for rel_support in REL_SUPPORTS:
#         models = dict()
#         footprints = dict()
#         for alg_name in ALGORITHMS:
#             for patterns_name in PATTERNS_NAMES:
#                 if alg_name in COMPARE_APPROACHES and patterns_name != PATTERNS_NAMES_DICT['all']:
#                     continue
#                 with open(os.path.join(save_dir, f'models_{alg_name}_{rel_support}_{patterns_name}.pkl'),
#                           'rb') as model_file:
#                     models[f'{alg_name} ({patterns_name})'] = process_models(pickle.load(model_file))
#                     print(f'Calculating footprints for alg {alg_name}, {patterns_name} and support {rel_support}')
#                     footprints[f'{alg_name} ({patterns_name})'] = calculate_footprints_for_models(models[alg_name])
#                     print(
#                         f'Finished calculating footprints for alg {alg_name}, {patterns_name} and support {rel_support}')
#
#         results = []
#         alg_names = []
#         for alg in list(ALGORITHMS.keys()):
#             for patterns_name in PATTERNS_NAMES:
#                 alg_names.append(f'{alg} ({patterns_name})')
#         for i in range(len(alg_names)):
#             results.append([alg_names[i], alg_names[i], 1])
#             models_1 = models[alg_names[i]]
#             for j in range(len(alg_names)):
#                 if i == j:
#                     continue
#                 models_2 = models[alg_names[j]]
#
#                 similarity = similarity_score_model_lists(models_1, models_2,
#                                                           precalculated_footprints=combine_footprints(
#                                                               [footprints[alg_names[i]], footprints[alg_names[j]]]))
#                 results.append([alg_names[i], alg_names[j], similarity])
#
#         df = pd.DataFrame(results, columns=['Algorithm1', 'Algorithm2', 'Similarity'])
#         df = df.pivot('Algorithm1', 'Algorithm2', 'Similarity')
#         df.index = pd.CategoricalIndex(df.index, categories=alg_names)
#         df.sort_index(level=0, inplace=True)
#         df = df[alg_names]
#         sns.heatmap(df, annot=True)
#         plt.draw()
#         plt.savefig(os.path.join(plot_dir, f'similarity_{rel_support}_{FREQ_COUNT_STRAT.name}.pdf'),
#                     bbox_inches='tight')
#         plt.close()


def process_models(models):
    if isinstance(models[0], tuple):
        return models

    res = []
    for tree in models:
        res.append(pt_converter.apply(tree))

    return res


def __remove_plus_postfix_from_models(models):
    for (net, _, _), _ in models:
        for transition in net.transitions:
            if transition.label is None:
                continue

            if transition.label.endswith("+"):
                transition.label = transition.label[:-1]

    return models


def calculate_distance_matrices(all_patterns, maximal, closed, infix):
    dist_all = None
    dist_max = None
    dist_closed = None
    dist_infix = None

    if len(all_patterns) < MATRIX_THRESHOLD:
        dist_all = calculate_distance_matrix(all_patterns)

    if len(maximal) < MATRIX_THRESHOLD:
        dist_max = calculate_distance_matrix(maximal)

    if len(closed) < MATRIX_THRESHOLD:
        dist_closed = calculate_distance_matrix(closed)

    if len(infix) < MATRIX_THRESHOLD:
        dist_infix = calculate_distance_matrix(infix)

    return dist_all, dist_closed, dist_max, dist_infix


if __name__ == "__main__":
    # generate_plots('BPI_CH_2020_PrepaidTravelCost.xes',
    #                'C:\\sources\\arbeit\\cortado\\master_thesis\\BPI_Ch_2020_PrepaidTravelCost\\local_process_models\\data\\TraceTransaction.csv')

    build_local_process_models()
