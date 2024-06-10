from statistics import mean, median
import pandas as pd

from cortado_core.subprocess_discovery.concurrency_trees.cTrees import cTreeOperator
from cortado_core.subprocess_discovery.subtree_mining.treebank import (
    create_treebank_from_cv_variants,
)
from cortado_core.utils.cvariants import get_concurrency_variants
from pm4py.objects.log.importer.xes.importer import apply as xes_import
from cortado_core.utils.timestamp_utils import TimeUnit


def __compute_size_of_tree(tree):
    size = 1

    if tree.op:
        for child in tree.children:
            size += __compute_size_of_tree(child)

    return size


def count_concurrent_sequential(tree):
    concurrent = 0
    sequential = 0
    fallthrough = 0

    for child in tree.children:
        if child.op:
            nC, nS, nF = count_concurrent_sequential(child)

            concurrent += nC
            sequential += nS
            fallthrough += nF

        else:
            if tree.op == cTreeOperator.Sequential:
                sequential += 1
            elif tree.op == cTreeOperator.Concurrent:
                concurrent += 1
            elif tree.op == cTreeOperator.Fallthrough:
                fallthrough += 1

    return concurrent, sequential, fallthrough


if __name__ == "__main__":
    from pm4py.util.xes_constants import DEFAULT_TRACEID_KEY

    files = [
        "Sepsis Cases - Event Log",
        "BPI_CH_2020_PrepaidTravelCost",
        "BPI Challenge 2017",
        "BPI_Challenge_2012",
    ]
    # files = ['reduced1000_BPI_2017']

    for file in files:
        df_dicts = []
        log_path = ""
        log = log_path + file + ".xes"
        log_name = file

        l = xes_import(log)

        nEvents = sum([len(trace) for trace in l])

        nUnique = len(
            set.union(
                *[set([event[DEFAULT_TRACEID_KEY] for event in trace]) for trace in l]
            )
        )

        for time_gran in [
            TimeUnit.MS,
            TimeUnit.SEC,
            TimeUnit.MIN,
            TimeUnit.HOUR,
            TimeUnit.DAY,
        ]:
            # l = create_example_log_2()
            variants = get_concurrency_variants(l, False, time_gran)
            treebank = create_treebank_from_cv_variants(variants, artifical_start=False)

            nVariants = len(treebank)
            nTraces = sum([entry.nTraces for entry in treebank.values()])
            maxNumberOfTraces = max([entry.nTraces for entry in treebank.values()])
            minNumberOfTraces = min([entry.nTraces for entry in treebank.values()])

            maxTreeSize = max(
                [__compute_size_of_tree(tree.tree) for tid, tree in treebank.items()]
            )
            minTreeSize = min(
                [__compute_size_of_tree(tree.tree) for tid, tree in treebank.items()]
            )
            medianTreeSize = median(
                [__compute_size_of_tree(tree.tree) for tid, tree in treebank.items()]
            )

            minActivites = min(
                [
                    sum(count_concurrent_sequential(tree.tree))
                    for tid, tree in treebank.items()
                ]
            )
            maxActivites = max(
                [
                    sum(count_concurrent_sequential(tree.tree))
                    for tid, tree in treebank.items()
                ]
            )
            medianActivities = median(
                [
                    sum(count_concurrent_sequential(tree.tree))
                    for tid, tree in treebank.items()
                ]
            )

            const_mult = lambda x, y: tuple([l * y for l in x])

            nCTraces, nSTraces, nFTraces = zip(
                *[
                    const_mult(count_concurrent_sequential(tree.tree), tree.nTraces)
                    for tid, tree in treebank.items()
                ]
            )
            nC, nS, nF = zip(
                *[
                    count_concurrent_sequential(tree.tree)
                    for tid, tree in treebank.items()
                ]
            )

            print("Activities:", nUnique, minActivites, maxActivites, medianActivities)
            print("Trees:", nVariants, minTreeSize, maxTreeSize, medianTreeSize)
            print("Traces", nEvents, nTraces)
            print("Distribution", sum(nC), sum(nS), sum(nF))
            print("Distribution Traces", sum(nCTraces), sum(nSTraces), sum(nFTraces))

            dict = {
                "nUnique": nUnique,
                "granularity": time_gran,
                "minActivites": minActivites,
                "maxActivites": maxActivites,
                "medianActivities": medianActivities,
                "nVariants": nVariants,
                "minTreeSize": minTreeSize,
                "maxTreeSize": maxTreeSize,
                "medianTreeSize": medianTreeSize,
                "nEvents": nEvents,
                "nTraces": nTraces,
                "Tot_Concurrent": sum(nC),
                "Tot_Sequential": sum(nS),
                "Tot_Fallthrough": sum(nF),
                "Median_Concurrent": median(nC),
                "Median_Sequential": median(nS),
                "Median_Fallthrough": median(nF),
                "Tot_Concurrent Traces": sum(nCTraces),
                "Tot_Sequential Traces": sum(nSTraces),
                "Tot_Fallthrough Traces": sum(nFTraces),
                "Median_Concurrent Traces": median(nCTraces),
                "Median_Sequential Traces": median(nSTraces),
                "Median_Fallthrough Traces": median(nFTraces),
            }

            df_dicts.append(dict)

        print("Writing Results to File ... ")
        pd.DataFrame.from_dict(df_dicts).to_csv(
            ".\\cortado_core\\experiments\\subpattern_eval\\Eval_Res\\"
            + log_name
            + "_Log_Stats.csv"
        )
