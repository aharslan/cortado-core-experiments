import random
import statistics
import timeit
from unittest.mock import NonCallableMagicMock

import pandas as pd
from bootstrap_query_trees import (
    create_expression_leaf_nodes,
    join_leafs,
    join_trees_into_queries,
    join_unary_leaf_operators_into_groups,
    left_join_binary_leaf_operators_into_groups,
    right_join_binary_leaf_operators_into_groups,
)
from check_query_tree_evaluation import check_query_tree_with_counts
from compute_frequent_actvities import compute_frequent_activity_sets
from experiments.vql_evaluation.check_query_tree_no_early_stopping_count_nodes import (
    check_query_tree_no_early_stopping_count_nodes,
)
from experiments.vql_evaluation.check_query_tree_no_stopping import (
    check_query_tree_no_early_stopping,
)
from cortado_core.variant_query_language.query_tree import BinaryOperator, UnaryOperator
from cortado_core.utils.cvariants import get_concurrency_variants
from pm4py.objects.log.importer.xes.importer import apply as xes_import


from cortado_core.variant_query_language.check_query_tree_against_graph import (
    check_query_tree,
)


def run_query(qt, variants, aA):
    for i, variant in enumerate(variants):
        res = check_query_tree(qt, variant, aA, root=True)


def run_query_no_early_stopping(qt, variants, aA):
    for i, variant in enumerate(variants):
        res = check_query_tree_no_early_stopping(qt, variant, aA, root=True)


def bootstrap_queries(variants, seed=366591, nSamples=1500, multiProcessing=False):
    # Set the Seed for Replicatability
    print("SEED: ", seed)
    random.seed(seed)

    print("Number of Samples: ", nSamples)

    print("Computing Frequent Activities ...")
    aA, sA, eA, dfR, dfS, fR, fS, cR, cA = compute_frequent_activity_sets(variants)

    aA = list(aA)
    sA = list(sA)
    eA = list(eA)
    dfS = list(dfS)
    fS = list(fS)
    cA = list(cA)

    # Create the base set of
    print("Creating Leaf Candidates...")
    candidates = create_expression_leaf_nodes(
        variants, aA, sA, eA, dfR, dfS, fR, fS, cR, cA
    )

    # Create new Candidates by joining Unary and Binary Leafs to create groups
    print("Joining Leaf Candidates...")
    newCandidates = join_unary_leaf_operators_into_groups(
        candidates["end"], UnaryOperator.isEnd, len(variants), 2
    )
    newCandidates += join_unary_leaf_operators_into_groups(
        candidates["start"], UnaryOperator.isStart, len(variants), 2
    )
    newCandidates += join_unary_leaf_operators_into_groups(
        candidates["has"], UnaryOperator.contains, len(variants), 3
    )
    newCandidates += left_join_binary_leaf_operators_into_groups(
        candidates["directlyfollows"], BinaryOperator.DirectlyFollows, len(variants), 2
    )
    newCandidates += right_join_binary_leaf_operators_into_groups(
        candidates["directlyfollows"], BinaryOperator.DirectlyFollows, 2, variants, aA
    )
    newCandidates += left_join_binary_leaf_operators_into_groups(
        candidates["follows"], BinaryOperator.EventualyFollows, len(variants), 2
    )
    newCandidates += right_join_binary_leaf_operators_into_groups(
        candidates["follows"], BinaryOperator.EventualyFollows, 2, variants, aA
    )
    newCandidates += left_join_binary_leaf_operators_into_groups(
        candidates["concurrent"],
        BinaryOperator.Concurrent,
        len(variants),
        2,
    )
    newCandidates += right_join_binary_leaf_operators_into_groups(
        candidates["concurrent"], BinaryOperator.Concurrent, 2, variants, aA
    )

    # Add the single activitiy elements too
    for key in candidates:
        newCandidates += candidates[key]

    # Join Expressions via And and Or Log Operators
    print("Joining Leaf Candidates to Initial Trees...")
    andTrees, orTrees = join_leafs(newCandidates, len(variants), 3, 150, 200)

    # Joining trees into trees with larger height
    print("Growing Trees...")
    joined_And_Trees, joined_Or_Trees = join_trees_into_queries(
        andTrees, orTrees, newCandidates, len(variants), 3, 150, 200
    )

    # Complete set as Union of all previous sets
    print("Finished Query Set")
    complete_query_set = (
        joined_And_Trees + joined_Or_Trees + newCandidates + andTrees + orTrees
    )

    del newCandidates
    del joined_And_Trees
    del joined_Or_Trees
    del andTrees
    del orTrees

    return complete_query_set, aA


def run_general_performance_tests(
    complete_query_set, nSamples, reps, aA, log_name, seed=366591
):
    # Set the Seed for Replicatability
    print("SEED: ", seed)
    random.seed(seed)

    print("Performing General Performance")

    sample = random.sample(complete_query_set, nSamples)
    # del complete_query_set
    df_dicts = []

    for k, c in enumerate(sample):
        df_dict = {}
        rTimes = []

        c[0].set_height()
        c[0].sort()

        for rep in range(reps):
            start_time = timeit.default_timer()
            run_query(c[0], variants, aA)
            rTime = timeit.default_timer() - start_time
            rTimes.append(rTime)

        ids = []
        nEvaluatedNodes = []
        nEvaluatedBinaryLeafs = []
        nEvaluatedUnaryLeafs = []
        nEvaluatedLeafs = []
        nEvaluatedExpressions = []
        nEvaluatedBinaryExpressions = []
        nEvaluatedUnaryExpressions = []

        for i, variant in enumerate(variants):
            (
                res,
                nNodes,
                nExpressions,
                nBinaryExpr,
                nUnaryExpr,
                nBinaryLeafs,
                nUnaryLeafs,
            ) = check_query_tree_with_counts(c[0], variant, aA, root=True)

            nEvaluatedNodes.append(nNodes)
            nEvaluatedExpressions.append(nExpressions)
            nEvaluatedBinaryExpressions.append(nBinaryExpr)
            nEvaluatedUnaryExpressions.append(nUnaryExpr)
            nEvaluatedBinaryLeafs.append(nBinaryLeafs)
            nEvaluatedUnaryLeafs.append(nUnaryLeafs)
            nEvaluatedLeafs.append(nBinaryLeafs + nUnaryLeafs)

            if res:
                ids.append(i)

        df_dict["k"] = k
        df_dict["time"] = rTimes

        df_dict["nRes"] = len(ids)
        df_dict["nNodes"] = nEvaluatedNodes
        df_dict["nExpr"] = nEvaluatedExpressions

        df_dict["nLeafs"] = nEvaluatedLeafs
        df_dict["nBinaryLeafs"] = nEvaluatedBinaryLeafs
        df_dict["nUnaryLeafs"] = nEvaluatedUnaryLeafs

        df_dict["nBinaryExpr"] = nEvaluatedBinaryExpressions
        df_dict["nUnaryExpr"] = nEvaluatedUnaryExpressions
        df_dict["nUnaryExpr"] = nEvaluatedUnaryExpressions

        df_dict["nQTLeafs"] = c[0].get_nLeafs()
        df_dict["nQTExpr"] = c[0].get_nExpressions()
        df_dict["nQTNodes"] = c[0].get_nNodes()

        df_dict["height"] = c[0].height

        df_dicts.append(df_dict)

        if k % 100 == 0:
            print("K", k)

    print("Adding Extra Information to Results ...")
    df = pd.DataFrame.from_dict(df_dicts)

    df["TotalNodesEvaluated"] = df.nNodes.apply(lambda x: sum(x))
    df["TotalLeafsEvaluated"] = df.nLeafs.apply(lambda x: sum(x))
    df["TotalExprEvaluated"] = df.nExpr.apply(lambda x: sum(x))

    df["MedianNodesEvaluated"] = df.nNodes.apply(lambda x: statistics.median(x))
    df["MedianLeafsEvaluated"] = df.nLeafs.apply(lambda x: statistics.median(x))
    df["MedianExprEvaluated"] = df.nExpr.apply(lambda x: statistics.median(x))

    df["MaxNodesEvaluated"] = df.nNodes.apply(lambda x: max(x))
    df["MaxLeafsEvaluated"] = df.nLeafs.apply(lambda x: max(x))
    df["MaxExprEvaluated"] = df.nExpr.apply(lambda x: max(x))

    # Take the best Time
    df["runtime"] = df.time.apply(lambda x: min(x))

    df["TotalUnaryLeafsEvaluated"] = df.nUnaryLeafs.apply(lambda x: sum(x))
    df["TotalBinaryLeafsEvaluated"] = df.nBinaryLeafs.apply(lambda x: sum(x))

    df["MedianUnaryLeafsEvaluated"] = df.nUnaryLeafs.apply(
        lambda x: statistics.median(x)
    )
    df["MedianBinaryLeafsEvaluated"] = df.nBinaryLeafs.apply(
        lambda x: statistics.median(x)
    )

    df["MaxUnaryLeafsEvaluated"] = df.nUnaryLeafs.apply(lambda x: max(x))
    df["MaxBinaryLeafsEvaluated"] = df.nBinaryLeafs.apply(lambda x: max(x))

    df["TotalUnaryExprEvaluated"] = df.nUnaryExpr.apply(lambda x: sum(x))
    df["TotalBinaryExprEvaluated"] = df.nBinaryExpr.apply(lambda x: sum(x))

    df["MedianUnaryExprEvaluated"] = df.nUnaryExpr.apply(lambda x: statistics.median(x))
    df["MedianBinaryExprEvaluated"] = df.nBinaryExpr.apply(
        lambda x: statistics.median(x)
    )

    df["MaxUnaryExprEvaluated"] = df.nUnaryExpr.apply(lambda x: max(x))
    df["MaxBinaryExprEvaluated"] = df.nBinaryExpr.apply(lambda x: max(x))

    print("Writing CSV ...")
    df.to_csv(log_name + "general_tests_eval.csv")

    print("FINISHED!")


def run_no_early_stopping_performance_tests(
    complete_query_set, nSamples, reps, aA, log_name, seed=366591
):
    # Set the Seed for Replicatability
    print("SEED: ", seed)
    random.seed(seed)

    print("Performing No Early Stopping Test")

    sample = random.sample(complete_query_set, nSamples)
    # del complete_query_set
    df_dicts = []

    for k, c in enumerate(sample):
        df_dict = {}
        earlyStoppingTimes = []
        noEarlyStoppingTimes = []

        c[0].set_height()
        c[0].sort()

        for rep in range(reps):
            start_time = timeit.default_timer()
            run_query(c[0], variants, aA)
            earlyStoppingTime = timeit.default_timer() - start_time
            earlyStoppingTimes.append(earlyStoppingTime)

            start_time = timeit.default_timer()
            run_query_no_early_stopping(c[0], variants, aA)
            noEarlyStoppingTime = timeit.default_timer() - start_time
            noEarlyStoppingTimes.append(noEarlyStoppingTime)

        ids = []
        early_stopping_nEvaluatedNodes = []
        early_stopping_nEvaluatedBinaryLeafs = []
        early_stopping_nEvaluatedUnaryLeafs = []
        early_stopping_nEvaluatedLeafs = []
        early_stopping_nEvaluatedExpressions = []
        early_stopping_nEvaluatedBinaryExpressions = []
        early_stopping_nEvaluatedUnaryExpressions = []

        no_early_stopping_nEvaluatedNodes = []
        no_early_stopping_nEvaluatedBinaryLeafs = []
        no_early_stopping_nEvaluatedUnaryLeafs = []
        no_early_stopping_nEvaluatedLeafs = []
        no_early_stopping_nEvaluatedExpressions = []
        no_early_stopping_nEvaluatedBinaryExpressions = []
        no_early_stopping_nEvaluatedUnaryExpressions = []

        for i, variant in enumerate(variants):
            (
                res,
                nNodes,
                nExpressions,
                nBinaryExpr,
                nUnaryExpr,
                nBinaryLeafs,
                nUnaryLeafs,
            ) = check_query_tree_with_counts(c[0], variant, aA, root=True)

            early_stopping_nEvaluatedNodes.append(nNodes)
            early_stopping_nEvaluatedExpressions.append(nExpressions)
            early_stopping_nEvaluatedBinaryExpressions.append(nBinaryExpr)
            early_stopping_nEvaluatedUnaryExpressions.append(nUnaryExpr)
            early_stopping_nEvaluatedBinaryLeafs.append(nBinaryLeafs)
            early_stopping_nEvaluatedUnaryLeafs.append(nUnaryLeafs)
            early_stopping_nEvaluatedLeafs.append(nBinaryLeafs + nUnaryLeafs)

            (
                res,
                nNodes,
                nExpressions,
                nBinaryExpr,
                nUnaryExpr,
                nBinaryLeafs,
                nUnaryLeafs,
            ) = check_query_tree_no_early_stopping_count_nodes(
                c[0], variant, aA, root=True
            )

            no_early_stopping_nEvaluatedNodes.append(nNodes)
            no_early_stopping_nEvaluatedExpressions.append(nExpressions)
            no_early_stopping_nEvaluatedBinaryExpressions.append(nBinaryExpr)
            no_early_stopping_nEvaluatedUnaryExpressions.append(nUnaryExpr)
            no_early_stopping_nEvaluatedBinaryLeafs.append(nBinaryLeafs)
            no_early_stopping_nEvaluatedUnaryLeafs.append(nUnaryLeafs)
            no_early_stopping_nEvaluatedLeafs.append(nBinaryLeafs + nUnaryLeafs)

            if res:
                ids.append(i)

        df_dict["k"] = k
        df_dict["no_early_stoppping_time"] = noEarlyStoppingTimes
        df_dict["early_stoppping_time"] = earlyStoppingTimes

        df_dict["nRes"] = len(ids)
        df_dict["early_stopping_nNodes"] = early_stopping_nEvaluatedNodes
        df_dict["early_stopping_nExpr"] = early_stopping_nEvaluatedExpressions

        df_dict["early_stopping_nLeafs"] = early_stopping_nEvaluatedLeafs
        df_dict["early_stopping_nBinaryLeafs"] = early_stopping_nEvaluatedBinaryLeafs
        df_dict["early_stopping_nUnaryLeafs"] = early_stopping_nEvaluatedUnaryLeafs

        df_dict[
            "early_stopping_nBinaryExpr"
        ] = early_stopping_nEvaluatedBinaryExpressions
        df_dict["early_stopping_nUnaryExpr"] = early_stopping_nEvaluatedUnaryExpressions
        df_dict["early_stopping_nUnaryExpr"] = early_stopping_nEvaluatedUnaryExpressions

        df_dict["nQTLeafs"] = c[0].get_nLeafs()
        df_dict["nQTExpr"] = c[0].get_nExpressions()
        df_dict["nQTNodes"] = c[0].get_nNodes()

        df_dict["no_early_stopping_nNodes"] = no_early_stopping_nEvaluatedNodes
        df_dict["no_early_stopping_nExpr"] = no_early_stopping_nEvaluatedExpressions

        df_dict["no_early_stopping_nLeafs"] = no_early_stopping_nEvaluatedLeafs
        df_dict[
            "no_early_stopping_nBinaryLeafs"
        ] = no_early_stopping_nEvaluatedBinaryLeafs
        df_dict[
            "no_early_stopping_nUnaryLeafs"
        ] = no_early_stopping_nEvaluatedUnaryLeafs

        df_dict[
            "no_early_stopping_nBinaryExpr"
        ] = no_early_stopping_nEvaluatedBinaryExpressions
        df_dict[
            "no_early_stopping_nUnaryExpr"
        ] = no_early_stopping_nEvaluatedUnaryExpressions
        df_dict[
            "no_early_stopping_nUnaryExpr"
        ] = no_early_stopping_nEvaluatedUnaryExpressions

        df_dict["height"] = c[0].height

        df_dicts.append(df_dict)

        if k % 100 == 0:
            print("K", k)

    print("Adding Extra Information to Results ...")
    df = pd.DataFrame.from_dict(df_dicts)

    df["early_stopping_TotalNodesEvaluated"] = df.early_stopping_nNodes.apply(
        lambda x: sum(x)
    )
    df["early_stopping_TotalLeafsEvaluated"] = df.early_stopping_nLeafs.apply(
        lambda x: sum(x)
    )
    df["early_stopping_TotalExprEvaluated"] = df.early_stopping_nExpr.apply(
        lambda x: sum(x)
    )

    df["no_early_stopping_TotalNodesEvaluated"] = df.no_early_stopping_nNodes.apply(
        lambda x: sum(x)
    )
    df["no_early_stopping_TotalLeafsEvaluated"] = df.no_early_stopping_nLeafs.apply(
        lambda x: sum(x)
    )
    df["no_early_stopping_TotalExprEvaluated"] = df.no_early_stopping_nExpr.apply(
        lambda x: sum(x)
    )

    df["early_stopping_MedianNodesEvaluated"] = df.early_stopping_nNodes.apply(
        lambda x: statistics.median(x)
    )
    df["early_stopping_MedianLeafsEvaluated"] = df.early_stopping_nLeafs.apply(
        lambda x: statistics.median(x)
    )
    df["early_stopping_MedianExprEvaluated"] = df.early_stopping_nExpr.apply(
        lambda x: statistics.median(x)
    )

    df["no_early_stopping_MedianNodesEvaluated"] = df.no_early_stopping_nNodes.apply(
        lambda x: statistics.median(x)
    )
    df["no_early_stopping_MedianLeafsEvaluated"] = df.no_early_stopping_nLeafs.apply(
        lambda x: statistics.median(x)
    )
    df["no_early_stopping_MedianExprEvaluated"] = df.no_early_stopping_nExpr.apply(
        lambda x: statistics.median(x)
    )

    df["early_stopping_MaxNodesEvaluated"] = df.early_stopping_nNodes.apply(
        lambda x: max(x)
    )
    df["early_stopping_MaxLeafsEvaluated"] = df.early_stopping_nLeafs.apply(
        lambda x: max(x)
    )
    df["early_stopping_MaxExprEvaluated"] = df.early_stopping_nExpr.apply(
        lambda x: max(x)
    )

    df["no_early_stopping_MaxNodesEvaluated"] = df.no_early_stopping_nNodes.apply(
        lambda x: max(x)
    )
    df["no_early_stopping_MaxLeafsEvaluated"] = df.no_early_stopping_nLeafs.apply(
        lambda x: max(x)
    )
    df["no_early_stopping_MaxExprEvaluated"] = df.no_early_stopping_nExpr.apply(
        lambda x: max(x)
    )

    # Take the best Time
    df["stopping_runtime"] = df.early_stoppping_time.apply(lambda x: min(x))
    df["no_stopping_runtime"] = df.no_early_stoppping_time.apply(lambda x: min(x))

    df["early_stopping_TotalUnaryLeafsEvaluated"] = df.early_stopping_nUnaryLeafs.apply(
        lambda x: sum(x)
    )
    df[
        "early_stopping_TotalBinaryLeafsEvaluated"
    ] = df.early_stopping_nBinaryLeafs.apply(lambda x: sum(x))

    df[
        "no_early_stopping_TotalUnaryLeafsEvaluated"
    ] = df.no_early_stopping_nUnaryLeafs.apply(lambda x: sum(x))
    df[
        "no_early_stopping_TotalBinaryLeafsEvaluated"
    ] = df.no_early_stopping_nBinaryLeafs.apply(lambda x: sum(x))

    df[
        "early_stopping_MedianUnaryLeafsEvaluated"
    ] = df.early_stopping_nUnaryLeafs.apply(lambda x: statistics.median(x))
    df[
        "early_stopping_MedianBinaryLeafsEvaluated"
    ] = df.early_stopping_nBinaryLeafs.apply(lambda x: statistics.median(x))

    df[
        "no_early_stopping_MedianUnaryLeafsEvaluated"
    ] = df.no_early_stopping_nUnaryLeafs.apply(lambda x: statistics.median(x))
    df[
        "no_early_stopping_MedianBinaryLeafsEvaluated"
    ] = df.no_early_stopping_nBinaryLeafs.apply(lambda x: statistics.median(x))

    df["early_stopping_MaxUnaryLeafsEvaluated"] = df.early_stopping_nUnaryLeafs.apply(
        lambda x: max(x)
    )
    df["early_stopping_MaxBinaryLeafsEvaluated"] = df.early_stopping_nBinaryLeafs.apply(
        lambda x: max(x)
    )

    df[
        "no_early_stopping_MaxUnaryLeafsEvaluated"
    ] = df.no_early_stopping_nUnaryLeafs.apply(lambda x: max(x))
    df[
        "no_early_stopping_MaxBinaryLeafsEvaluated"
    ] = df.no_early_stopping_nBinaryLeafs.apply(lambda x: max(x))

    df["early_stopping_TotalUnaryExprEvaluated"] = df.early_stopping_nUnaryExpr.apply(
        lambda x: sum(x)
    )
    df["early_stopping_TotalBinaryExprEvaluated"] = df.early_stopping_nBinaryExpr.apply(
        lambda x: sum(x)
    )

    df[
        "no_early_stopping_TotalUnaryExprEvaluated"
    ] = df.no_early_stopping_nUnaryExpr.apply(lambda x: sum(x))
    df[
        "no_early_stopping_TotalBinaryExprEvaluated"
    ] = df.no_early_stopping_nBinaryExpr.apply(lambda x: sum(x))

    df["early_stopping_MedianUnaryExprEvaluated"] = df.early_stopping_nUnaryExpr.apply(
        lambda x: statistics.median(x)
    )
    df[
        "early_stopping_MedianBinaryExprEvaluated"
    ] = df.early_stopping_nBinaryExpr.apply(lambda x: statistics.median(x))

    df[
        "no_early_stopping_MedianUnaryExprEvaluated"
    ] = df.no_early_stopping_nUnaryExpr.apply(lambda x: statistics.median(x))
    df[
        "no_early_stopping_MedianBinaryExprEvaluated"
    ] = df.no_early_stopping_nBinaryExpr.apply(lambda x: statistics.median(x))

    df["early_stopping_MaxUnaryExprEvaluated"] = df.early_stopping_nUnaryExpr.apply(
        lambda x: max(x)
    )
    df["early_stopping_MaxBinaryExprEvaluated"] = df.early_stopping_nBinaryExpr.apply(
        lambda x: max(x)
    )

    df[
        "no_early_stopping_MaxUnaryExprEvaluated"
    ] = df.no_early_stopping_nUnaryExpr.apply(lambda x: max(x))
    df[
        "no_early_stopping_MaxBinaryExprEvaluated"
    ] = df.no_early_stopping_nBinaryExpr.apply(lambda x: max(x))

    print("Writing CSV ...")
    df.to_csv(log_name + "_no_early_stopping.csv")

    print("FINISHED!")


def run_sort_no_sort_performance_tests(
    complete_query_set, nSamples, reps, aA, log_name, seed=366591
):
    # Set the Seed for Replicatability
    print("SEED: ", seed)
    random.seed(seed)

    print("Performing Test Sort vs. No Sort")
    sample = random.sample(complete_query_set, nSamples)
    # del complete_query_set

    df_dicts = []

    for k, c in enumerate(sample):
        df_dict = {}
        noSortTimes = []
        sortTimes = []

        for rep in range(reps):
            start_time = timeit.default_timer()
            run_query(c[0], variants, aA)
            noSortTime = timeit.default_timer() - start_time
            noSortTimes.append(noSortTime)

            c[0].set_height()
            c[0].sort()

            start_time = timeit.default_timer()
            run_query(c[0], variants, aA)
            sortTime = timeit.default_timer() - start_time
            sortTimes.append(sortTime)

        ids = []
        nEvaluatedNodes = []
        nEvaluatedBinaryLeafs = []
        nEvaluatedUnaryLeafs = []
        nEvaluatedLeafs = []
        nEvaluatedExpressions = []
        nEvaluatedBinaryExpressions = []
        nEvaluatedUnaryExpressions = []

        for i, variant in enumerate(variants):
            (
                res,
                nNodes,
                nExpressions,
                nBinaryExpr,
                nUnaryExpr,
                nBinaryLeafs,
                nUnaryLeafs,
            ) = check_query_tree_with_counts(c[0], variant, aA, root=True)

            nEvaluatedNodes.append(nNodes)
            nEvaluatedExpressions.append(nExpressions)
            nEvaluatedBinaryExpressions.append(nBinaryExpr)
            nEvaluatedUnaryExpressions.append(nUnaryExpr)
            nEvaluatedBinaryLeafs.append(nBinaryLeafs)
            nEvaluatedUnaryLeafs.append(nUnaryLeafs)
            nEvaluatedLeafs.append(nBinaryLeafs + nUnaryLeafs)

            if res:
                ids.append(i)

        df_dict["k"] = k
        df_dict["noSortTime"] = noSortTimes
        df_dict["sortTime"] = sortTimes

        df_dict["nRes"] = len(ids)
        df_dict["nNodes"] = nEvaluatedNodes
        df_dict["nExpr"] = nEvaluatedExpressions

        df_dict["nLeafs"] = nEvaluatedLeafs
        df_dict["nBinaryLeafs"] = nEvaluatedBinaryLeafs
        df_dict["nUnaryLeafs"] = nEvaluatedUnaryLeafs

        df_dict["nBinaryExpr"] = nEvaluatedBinaryExpressions
        df_dict["nUnaryExpr"] = nEvaluatedUnaryExpressions
        df_dict["nUnaryExpr"] = nEvaluatedUnaryExpressions

        df_dict["nQTLeafs"] = c[0].get_nLeafs()
        df_dict["nQTExpr"] = c[0].get_nExpressions()
        df_dict["nQTNodes"] = c[0].get_nNodes()

        df_dict["height"] = c[0].height

        df_dicts.append(df_dict)

        if k % 100 == 0:
            print("K", k)

    print("Adding Extra Information to Results ...")
    df = pd.DataFrame.from_dict(df_dicts)

    df["TotalNodesEvaluated"] = df.nNodes.apply(lambda x: sum(x))
    df["TotalLeafsEvaluated"] = df.nLeafs.apply(lambda x: sum(x))
    df["TotalExprEvaluated"] = df.nExpr.apply(lambda x: sum(x))

    df["MedianNodesEvaluated"] = df.nNodes.apply(lambda x: statistics.median(x))
    df["MedianLeafsEvaluated"] = df.nLeafs.apply(lambda x: statistics.median(x))
    df["MedianExprEvaluated"] = df.nExpr.apply(lambda x: statistics.median(x))

    df["MaxNodesEvaluated"] = df.nNodes.apply(lambda x: max(x))
    df["MaxLeafsEvaluated"] = df.nLeafs.apply(lambda x: max(x))
    df["MaxExprEvaluated"] = df.nExpr.apply(lambda x: max(x))

    # Take the best Time
    df["no_sort_runtime"] = df.noSortTime.apply(lambda x: min(x))
    df["sort_runtime"] = df.sortTime.apply(lambda x: min(x))

    df["TotalUnaryLeafsEvaluated"] = df.nUnaryLeafs.apply(lambda x: sum(x))
    df["TotalBinaryLeafsEvaluated"] = df.nBinaryLeafs.apply(lambda x: sum(x))

    df["MedianUnaryLeafsEvaluated"] = df.nUnaryLeafs.apply(
        lambda x: statistics.median(x)
    )
    df["MedianBinaryLeafsEvaluated"] = df.nBinaryLeafs.apply(
        lambda x: statistics.median(x)
    )

    df["MaxUnaryLeafsEvaluated"] = df.nUnaryLeafs.apply(lambda x: max(x))
    df["MaxBinaryLeafsEvaluated"] = df.nBinaryLeafs.apply(lambda x: max(x))

    df["TotalUnaryExprEvaluated"] = df.nUnaryExpr.apply(lambda x: sum(x))
    df["TotalBinaryExprEvaluated"] = df.nBinaryExpr.apply(lambda x: sum(x))

    df["MedianUnaryExprEvaluated"] = df.nUnaryExpr.apply(lambda x: statistics.median(x))
    df["MedianBinaryExprEvaluated"] = df.nBinaryExpr.apply(
        lambda x: statistics.median(x)
    )

    df["MaxUnaryExprEvaluated"] = df.nUnaryExpr.apply(lambda x: max(x))
    df["MaxBinaryExprEvaluated"] = df.nBinaryExpr.apply(lambda x: max(x))

    print("Writing CSV ...")
    df.to_csv(log_name + "_sort_no_sort_eval.csv")

    print("FINISHED!")

    return NonCallableMagicMock


if __name__ == "__main__":
    for log_name in [
        "BPI Challenge 2017",
        "BPI_CH_2020_PrepaidTravelCost",
        "BPI_Challenge_2012",
        "RoadTrafficFineManagement",
        "Sepsis Cases - Event Log",
    ]:
        print()
        print("Log Name:", log_name)

        log_path = ""
        log = xes_import(log_path + log_name + ".xes")
        variants = get_concurrency_variants(log, False)

        print("Bootstrapping Queries... ")
        query_set, aA = bootstrap_queries(variants)
        # run_general_performance_tests(query_set, 1500, 5, aA, log_name = 'reducedRoadTraffic1000')
        # run_sort_no_sort_performance_tests(query_set, 1500, 5, aA, log_name="Sepsis", seed=366591)
        run_no_early_stopping_performance_tests(
            query_set, 1500, 5, aA, log_name=log_name, seed=366591
        )
