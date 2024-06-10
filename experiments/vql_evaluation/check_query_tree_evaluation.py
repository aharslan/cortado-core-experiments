from functools import partial
from typing import Set
from cortado_core.utils.cgroups_graph import ConcurrencyGroup
from cortado_core.variant_query_language.query_tree import (
    AllGroup,
    BinaryExpressionLeaf,
    BinaryOperator,
    ExpressionLeaf,
    LOperator,
    OperatorNode,
    QuantifierOperator,
    QueryTree,
    UnaryExpressionLeaf,
    UnaryOperator,
)


def check_query_tree_with_counts(
    qT: QueryTree, graph: ConcurrencyGroup, activities: Set[str], root=False
) -> bool:
    nEvaluatedNodes: int = 1
    nEvaluatedLeafs: int = 0
    nEvaluatedBinaryExpressions = 0
    nEvaluatedUnaryExpressions = 0
    nEvaluatedBinaryLeafs = 0
    nEvaluatedUnaryLeafs = 0

    def evaluate_expression_leaf_with_counts(
        qT: ExpressionLeaf, graph: ConcurrencyGroup, activities: Set[str]
    ) -> bool:
        eVal = None
        evaluted_expressions = 0
        nEvaluatedBinaryExpressions = 0
        nEvaluatedUnaryExpressions = 0
        nEvaluatedBinaryLeafs = 0
        nEvaluatedUnaryLeafs = 0

        if isinstance(qT, UnaryExpressionLeaf):
            eVal, evaluted_expressions = evaluate_unary_leaf_with_counts(
                qT, graph, activities
            )
            nEvaluatedUnaryExpressions = evaluted_expressions
            nEvaluatedUnaryLeafs = 1

        elif isinstance(qT, BinaryExpressionLeaf):
            eVal, evaluted_expressions = evaluate_binary_leaf_with_counts(
                qT, graph, activities
            )
            nEvaluatedBinaryExpressions = evaluted_expressions
            nEvaluatedBinaryLeafs = 1

        return (
            eVal,
            evaluted_expressions,
            nEvaluatedBinaryExpressions,
            nEvaluatedUnaryExpressions,
            nEvaluatedBinaryLeafs,
            nEvaluatedUnaryLeafs,
        )

    cVal: bool

    # Evaluate Inner Nodes
    if isinstance(qT, OperatorNode):
        cVal = False

        # AND Node, if any one is false, return False
        if qT.lOp == LOperator.AND:
            cVal = True

            for child in qT.children:
                (
                    ev,
                    nChildrenEvaluatedNodes,
                    nChildrenEvaluatedLeafs,
                    nBinaryExpressions,
                    nUnaryExpressions,
                    nBinaryLeafs,
                    nUnaryLeafs,
                ) = check_query_tree_with_counts(child, graph, activities)

                nEvaluatedNodes += nChildrenEvaluatedNodes
                nEvaluatedLeafs += nChildrenEvaluatedLeafs
                nEvaluatedBinaryExpressions += nBinaryExpressions
                nEvaluatedUnaryExpressions += nUnaryExpressions
                nEvaluatedBinaryLeafs += nBinaryLeafs
                nEvaluatedUnaryLeafs += nUnaryLeafs

                if not ev:
                    cVal = False
                    break

        # OR Node, if any one is true, return True
        elif qT.lOp == LOperator.OR:
            cVal = False

            for child in qT.children:
                (
                    ev,
                    nChildrenEvaluatedNodes,
                    nChildrenEvaluatedLeafs,
                    nBinaryExpressions,
                    nUnaryExpressions,
                    nBinaryLeafs,
                    nUnaryLeafs,
                ) = check_query_tree_with_counts(child, graph, activities)

                nEvaluatedNodes += nChildrenEvaluatedNodes
                nEvaluatedLeafs += nChildrenEvaluatedLeafs
                nEvaluatedBinaryExpressions += nBinaryExpressions
                nEvaluatedUnaryExpressions += nUnaryExpressions
                nEvaluatedBinaryLeafs += nBinaryLeafs
                nEvaluatedUnaryLeafs += nUnaryLeafs

                if ev:
                    cVal = True
                    break

    # Evaluate Expression Leaf
    elif isinstance(qT, ExpressionLeaf):
        (
            cVal,
            nEvaluatedLeafs,
            nBinaryExpressions,
            nUnaryExpressions,
            nBinaryLeafs,
            nUnaryLeafs,
        ) = evaluate_expression_leaf_with_counts(qT, graph, activities)

        nEvaluatedBinaryExpressions = nBinaryExpressions
        nEvaluatedUnaryExpressions = nUnaryExpressions
        nEvaluatedBinaryLeafs = nBinaryLeafs
        nEvaluatedUnaryLeafs = nUnaryLeafs

    # Invert the result
    if qT.neg:
        cVal = not cVal

    return (
        cVal,
        nEvaluatedNodes,
        nEvaluatedLeafs,
        nEvaluatedBinaryExpressions,
        nEvaluatedUnaryExpressions,
        nEvaluatedBinaryLeafs,
        nEvaluatedUnaryLeafs,
    )


def evaluate_unary_leaf_with_counts(
    qT: UnaryExpressionLeaf,
    graph: ConcurrencyGroup,
    activities: Set[str],
) -> bool:
    cActivities = qT.activities.getMembers(activities)

    any_activity = not isinstance(qT.activities, AllGroup)
    eVal: bool
    evaluted_expressions = 0

    if qT.operator == UnaryOperator.contains:
        eVal, evaluted_expressions = check_unary_groups_with_counts(
            graph.events,
            cActivities,
            any_activity,
            quantifier=qT.qOp,
            occ=qT.number,
        )

    elif qT.operator == UnaryOperator.isStart:
        eVal, evaluted_expressions = check_unary_groups_with_counts(
            graph.start_activities,
            cActivities,
            any_activity,
            quantifier=qT.qOp,
            occ=qT.number,
        )

    elif qT.operator == UnaryOperator.isEnd:
        eVal, evaluted_expressions = check_unary_groups_with_counts(
            graph.end_activities,
            cActivities,
            any_activity,
            quantifier=qT.qOp,
            occ=qT.number,
        )

    return eVal, evaluted_expressions


def evaluate_binary_leaf_with_counts(
    qT: BinaryExpressionLeaf, graph: ConcurrencyGroup, activities: Set[str]
) -> bool:
    lactivities = qT.lactivities.getMembers(activities)
    ractivities = qT.ractivities.getMembers(activities)
    evaluted_expressions = 0

    any = not (
        isinstance(qT.lactivities, AllGroup) or isinstance(qT.ractivities, AllGroup)
    )

    eVal: bool

    if qT.operator == BinaryOperator.DirectlyFollows:
        eVal, evaluted_expressions = check_binary_expressions_with_counts(
            graph.directly_follows,
            graph.events,
            lactivities,
            ractivities,
            any_activity=any,
            quantifier=qT.qOp,
            occ=qT.number,
        )

    elif qT.operator == BinaryOperator.EventualyFollows:
        eVal, evaluted_expressions = check_binary_expressions_with_counts(
            graph.follows,
            graph.events,
            lactivities,
            ractivities,
            any_activity=any,
            quantifier=qT.qOp,
            occ=qT.number,
        )

    elif qT.operator == BinaryOperator.Concurrent:
        eVal, evaluted_expressions = check_concurrent_with_counts(
            graph.concurrency_pairs,
            graph.events,
            lactivities,
            ractivities,
            any_activity=any,
            quantifier=qT.qOp,
            occ=qT.number,
        )

    return eVal, evaluted_expressions


def get_accessor(lactivity, ractivity):
    """
    Returns an accessor to the lActivity for the Concurrency Elements
    """

    if lactivity <= ractivity:
        return lambda x: x[0]

    else:
        return lambda x: x[1]


def get_lIDs(graph_elements, lAct, rAct, key=lambda x: x[0]):
    return set(map(key, graph_elements.get((lAct, rAct), [])))


def lex_order(x, y):
    if x > y:
        return (y, x)

    else:
        return (x, y)


def check_quantified_operator(obs_occ, quantifier, occ):
    if quantifier == QuantifierOperator.Equals:
        return obs_occ == occ

    elif quantifier == QuantifierOperator.Less:
        return obs_occ < occ

    elif quantifier == QuantifierOperator.Greater:
        return obs_occ > occ


def check_quantified_binary_expressions_with_counts(
    graph_elements,
    lActivities: Set[str],
    rActivities: Set[str],
    any_activity: bool,
    quantifier: QuantifierOperator,
    occ,
):
    evaluted_expressions = 0

    # We have a group on the right side
    if len(rActivities) > 1:
        evaluted_expressions = len(rActivities)

        lactivity = lActivities.copy().pop()

        lIds = [
            get_lIDs(graph_elements, lactivity, ractivity) for ractivity in rActivities
        ]

        if any_activity:
            # Compute the length of the set of lActivites followed ANY of the rActivities
            return (
                check_quantified_operator(quantifier, len(set.union(*lIds)), occ),
                evaluted_expressions,
            )

        else:
            # Compute the length of the set of lActivites followed by All of the rActivities
            return (
                check_quantified_operator(
                    quantifier, len(set.intersection(*lIds)), occ
                ),
                evaluted_expressions,
            )

    else:
        check_qant = partial(check_quantified_operator, quantifier=quantifier, occ=occ)
        ractivity = rActivities.pop()

        for lactivity in lActivities:
            evaluted_expressions += 1
            # Compute the number of unique lActivity that are followed by rActivity and check it against the quantifier
            if check_qant(len(get_lIDs(graph_elements, lactivity, ractivity))):
                if any_activity:
                    return True, evaluted_expressions

            elif not any_activity:
                return False, evaluted_expressions

        return not any_activity, evaluted_expressions


def check_non_quantified_binary_expressions_with_counts(
    graph_elements,
    events,
    lActivities: Set[str],
    rActivities: Set[str],
    any_activity: bool,
):
    evaluted_expressions = 0

    if len(rActivities) > 1 and any_activity:
        lactivity = lActivities.copy().pop()

        if not lactivity in events:
            return False, len(rActivities)

        lIds = [
            get_lIDs(graph_elements, lactivity, ractivity) for ractivity in rActivities
        ]

        return set.union(*lIds) == events[lactivity], len(rActivities)

    else:
        if any_activity:
            for lactivity in lActivities:
                if lactivity not in events:
                    evaluted_expressions += 1
                    continue

                for ractivity in rActivities:
                    evaluted_expressions += 1

                    if (
                        lactivity,
                        ractivity,
                    ) not in graph_elements or ractivity not in events:
                        continue

                    if (
                        set(map(lambda x: x[0], graph_elements[(lactivity, ractivity)]))
                        == events[lactivity]
                    ):
                        return True, evaluted_expressions

        else:
            for lactivity in lActivities:
                if lactivity not in events:
                    evaluted_expressions += 1
                    return False, evaluted_expressions

                for ractivity in rActivities:
                    evaluted_expressions += 1

                    if ractivity not in events:
                        return False, evaluted_expressions

                    if (lactivity, ractivity) not in graph_elements or not set(
                        map(lambda x: x[0], graph_elements[(lactivity, ractivity)])
                    ) == events[lactivity]:
                        return False, evaluted_expressions

        return not any_activity, evaluted_expressions


def check_binary_expressions_with_counts(
    graph_elements,
    events,
    lActivities: Set[str],
    rActivities: Set[str],
    any_activity: bool,
    quantifier: QuantifierOperator = None,
    occ=None,
):
    if occ:
        res, nEval = check_quantified_binary_expressions_with_counts(
            graph_elements, lActivities, rActivities, any_activity, quantifier, occ
        )

        return res, nEval

    else:
        res, nEval = check_non_quantified_binary_expressions_with_counts(
            graph_elements, events, lActivities, rActivities, any_activity
        )

        return res, nEval


def check_quantified_concurrent_with_counts(
    graph_elements,
    lActivities: Set[str],
    rActivities: Set[str],
    any_activity: bool,
    quantifier: QuantifierOperator,
    occ,
):
    # We have a group on the right side

    evaluted_expressions = 0

    if len(rActivities) > 1:
        lactivity = lActivities.copy().pop()

        lIds = [
            get_lIDs(
                graph_elements,
                *lex_order(lactivity, ractivity),
                get_accessor(lactivity, ractivity),
            )
            for ractivity in rActivities
        ]

        if any_activity:
            # Compute the length of the set of lActivites followed ANY of the rActivities
            return (
                check_quantified_operator(quantifier, len(set.union(*lIds)), occ),
                len(rActivities),
                len(rActivities),
            )

        else:
            # Compute the length of the set of lActivites followed by All of the rActivities
            return check_quantified_operator(
                quantifier, len(set.intersection(*lIds)), occ
            ), len(rActivities)

    else:
        check_qant = partial(check_quantified_operator, quantifier=quantifier, occ=occ)
        ractivity = rActivities.copy().pop()

        for lactivity in lActivities:
            evaluted_expressions += 1

            lIds = get_lIDs(
                graph_elements,
                *lex_order(lactivity, ractivity),
                get_accessor(lactivity, ractivity),
            )
            # Compute the number of unique lActivity that are followed by rActivity and check it against the quantifier

            if check_qant(len(lIds)):
                if any_activity:
                    return True, evaluted_expressions

            elif not any_activity:
                return False, evaluted_expressions

        return not any_activity, evaluted_expressions


def check_non_quantified_concurrent_with_counts(
    graph_elements,
    events,
    lActivities: Set[str],
    rActivities: Set[str],
    any_activity: bool,
):
    evaluted_expressions = 0

    if any_activity and len(rActivities) > 1:
        lactivity = lActivities.copy().pop()

        if not lactivity in events:
            return False, len(rActivities)

        lIds = [
            get_lIDs(
                graph_elements,
                lex_order(lactivity, rActivity),
                get_accessor(lactivity, rActivity),
            )
            for rActivity in rActivities
        ]

        return set.union(*lIds) == events[lactivity], len(rActivities)

    elif any_activity:
        for lactivity in lActivities:
            if lactivity not in events:
                evaluted_expressions += 1
                continue

            for ractivity in rActivities:
                l, r = lex_order(lactivity, ractivity)
                evaluted_expressions += 1

                if ractivity not in events or (l, r) not in graph_elements:
                    continue

                getLID = get_accessor(lactivity, ractivity)

                if set(map(getLID, graph_elements[(l, r)])) == events[lactivity]:
                    return True, evaluted_expressions

    else:
        for lactivity in lActivities:
            if lactivity not in events:
                evaluted_expressions += 1
                return False, evaluted_expressions

            for ractivity in rActivities:
                l, r = lex_order(lactivity, ractivity)
                evaluted_expressions += 1

                if ractivity not in events or (l, r) not in graph_elements:
                    return False, evaluted_expressions

                getLID = get_accessor(lactivity, ractivity)

                if not set(map(getLID, graph_elements[(l, r)])) == events[lactivity]:
                    return False, evaluted_expressions

    return not any_activity, evaluted_expressions


def check_concurrent_with_counts(
    graph_elements,
    events,
    lActivities,
    rActivities,
    any_activity: bool,
    quantifier: QuantifierOperator = None,
    occ=None,
):
    if occ:
        res, nEval = check_quantified_concurrent_with_counts(
            graph_elements, lActivities, rActivities, any_activity, quantifier, occ
        )
        return res, nEval

    else:
        res, nEval = check_non_quantified_concurrent_with_counts(
            graph_elements, events, lActivities, rActivities, any_activity
        )

        return res, nEval


def check_unary_groups_with_counts(
    graph_elements,
    activties: Set[str],
    any_activity: bool,
    quantifier: QuantifierOperator = None,
    occ=None,
):
    if occ:
        check_qant = partial(check_quantified_operator, quantifier=quantifier, occ=occ)
        res = map(check_qant, [len(graph_elements.get(act, [])) for act in activties])

        if any_activity:
            return any(res), len(activties)

        else:
            return all(res), len(activties)

    else:
        if any_activity:
            return not set(graph_elements).isdisjoint(activties), len(activties)

        else:
            return activties.issubset(set(graph_elements)), len(activties)
