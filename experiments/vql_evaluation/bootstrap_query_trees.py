from itertools import product
from cortado_core.variant_query_language.check_query_tree_against_graph import (
    check_query_tree,
)
from cortado_core.variant_query_language.query_tree import (
    AllGroup,
    AnyGroup,
    BinaryExpressionLeaf,
    BinaryOperator,
    LOperator,
    OperatorNode,
    UnaryExpressionLeaf,
    UnaryOperator,
)


def copy_group(group):
    if isinstance(group, AnyGroup):
        return AnyGroup(group, group.inv)

    elif isinstance(group, AllGroup):
        return AllGroup(group, group.inv)


def copy_leaf(leaf):
    if isinstance(leaf, UnaryExpressionLeaf):
        return UnaryExpressionLeaf(leaf.activities, leaf.operator, leaf.neg)

    else:
        return BinaryExpressionLeaf(
            leaf.lactivities, leaf.ractivities, leaf.operator, leaf.neg
        )


def copy_operator_node(node):
    cchildren = []

    for child in node.children:
        if isinstance(child, OperatorNode):
            cchildren.append(copy_operator_node(child))

        else:
            cchildren.append(child)

    0
    return OperatorNode(node.lOp, cchildren, neg=node.neg)


def get_query_variant_indexes(qt, variants, aA):
    ids = []

    for i, variant in enumerate(variants):
        res = check_query_tree(qt, variant, aA, root=True)

        if res:
            ids.append(i)

    return set(ids)


def create_expression_leaf_nodes(variants, aA, sA, eA, dfR, dfS, fR, fS, cR, cA):
    results = {}

    candidates = []

    for activity in sA:
        leaf = UnaryExpressionLeaf(AnyGroup([activity]), UnaryOperator.isStart)
        ids = get_query_variant_indexes(leaf, variants, aA)

        if len(ids) > 0 and len(ids) < len(variants):
            candidates.append((leaf, ids))

    results["start"] = candidates
    candidates = []

    for activity in eA:
        leaf = UnaryExpressionLeaf(AnyGroup([activity]), UnaryOperator.isEnd)
        ids = get_query_variant_indexes(leaf, variants, aA)

        if len(ids) > 0 and len(ids) < len(variants):
            candidates.append((leaf, ids))

    results["end"] = candidates
    candidates = []

    for activtiy in aA:
        leaf = UnaryExpressionLeaf(AnyGroup([activtiy]), UnaryOperator.contains)
        ids = get_query_variant_indexes(leaf, variants, aA)

        if len(ids) > 0 and len(ids) < len(variants):
            candidates.append((leaf, ids))

    results["has"] = candidates
    candidates = []

    for lactivtiy in dfS:
        for ractivity in dfR[lactivtiy]:
            leaf = BinaryExpressionLeaf(
                AnyGroup([lactivtiy]),
                AnyGroup([ractivity]),
                BinaryOperator.DirectlyFollows,
            )
            ids = get_query_variant_indexes(leaf, variants, aA)

            if len(ids) > 0 and len(ids) < len(variants):
                candidates.append((leaf, ids))

    results["directlyfollows"] = candidates
    candidates = []

    for lactivtiy in fS:
        for ractivity in fR[lactivtiy]:
            leaf = BinaryExpressionLeaf(
                AnyGroup([lactivtiy]),
                AnyGroup([ractivity]),
                BinaryOperator.EventualyFollows,
            )
            ids = get_query_variant_indexes(leaf, variants, aA)

            if len(ids) > 0 and len(ids) < len(variants):
                candidates.append((leaf, ids))

    results["follows"] = candidates
    candidates = []

    for lactivtiy in cA:
        if lactivtiy in cR:
            for ractivity in cR[lactivtiy]:
                leaf = BinaryExpressionLeaf(
                    AnyGroup([lactivtiy]),
                    AnyGroup([ractivity]),
                    BinaryOperator.Concurrent,
                )
                ids = get_query_variant_indexes(leaf, variants, aA)

                if len(ids) > 0 and len(ids) < len(variants):
                    candidates.append((leaf, ids))

    results["concurrent"] = candidates

    return results


def join_unary_leaf_operators_into_groups(unary_leafs, op, nVariants, max_width):
    allCandidates = []
    extended_trees = unary_leafs

    for k in range(max_width - 1):
        newTrees = []

        for c1, c2 in product(unary_leafs, extended_trees):
            a1 = c1[0].activities[0]
            a2 = c2[0].activities

            c1_ids = c1[1]
            c2_ids = c2[1]

            if isinstance(a2, AnyGroup):
                if a1 > a2[-1]:
                    union = c1_ids.union(c2_ids)
                    activites = a2 + [a1]

                    if len(union) != nVariants:
                        # Create a new AnyGroup
                        leaf = UnaryExpressionLeaf(AnyGroup(activites), op)
                        newTrees.append((leaf, union))

            if isinstance(a2, AllGroup) or (isinstance(a2, AnyGroup) and len(a2) == 1):
                if a1 > a2[-1]:
                    intersection = c1_ids.intersection(c2_ids)
                    activites = a2 + [a1]

                    if len(intersection) != 0:
                        # Create a new AllGroup
                        leaf = UnaryExpressionLeaf(AllGroup(activites), op)
                        newTrees.append((leaf, intersection))

        allCandidates.extend(newTrees)
        extended_trees = newTrees

    return allCandidates


def left_join_binary_leaf_operators_into_groups(binary_leafs, op, nVariants, max_width):
    allCandidates = []
    extended_trees = binary_leafs

    for k in range(max_width - 1):
        newTrees = []

        for c1, c2 in product(binary_leafs, extended_trees):
            l1 = c1[0].lactivities[0]
            l2 = c2[0].lactivities

            r1 = c1[0].ractivities[0]
            r2 = c2[0].ractivities[0]

            if r1 == r2:
                c1_ids = c1[1]
                c2_ids = c2[1]

                if isinstance(l2, AnyGroup):
                    if l1 > l2[-1]:
                        union = c1_ids.union(c2_ids)
                        activites = l2 + [l1]

                        if len(union) != nVariants:
                            # Create a new AnyGroup
                            leaf = BinaryExpressionLeaf(
                                lActivities=AnyGroup(activites),
                                rActivities=AnyGroup([r1]),
                                operator=op,
                            )
                            newTrees.append((leaf, union))

                if isinstance(l2, AllGroup) or (
                    isinstance(l2, AnyGroup) and len(l2) == 1
                ):
                    if l1 > l2[-1]:
                        intersection = c1_ids.intersection(c2_ids)
                        activites = l2 + [l1]

                        if len(intersection) != 0:
                            # Create a new AllGroup
                            leaf = BinaryExpressionLeaf(
                                lActivities=AllGroup(activites),
                                rActivities=AnyGroup([r1]),
                                operator=op,
                            )
                            newTrees.append((leaf, intersection))

        allCandidates.extend(newTrees)
        extended_trees = newTrees

    return allCandidates


def right_join_binary_leaf_operators_into_groups(
    binary_leafs, op, max_width, variants, aA
):
    allCandidates = []
    extended_trees = binary_leafs

    for k in range(max_width - 1):
        newTrees = []

        for c1, c2 in product(binary_leafs, extended_trees):
            l1 = c1[0].lactivities[0]
            l2 = c2[0].lactivities[0]

            r1 = c1[0].ractivities[0]
            r2 = c2[0].ractivities

            if l1 == l2:
                c1_ids = c1[1]
                c2_ids = c2[1]

                if isinstance(r2, AnyGroup):
                    if r1 > r2[-1]:
                        activites = r2 + [r1]

                        # Create a new AnyGroup
                        leaf = BinaryExpressionLeaf(
                            lActivities=AnyGroup([l1]),
                            rActivities=AnyGroup(activites),
                            operator=op,
                        )

                        ids = set(get_query_variant_indexes(leaf, variants, aA))
                        newTrees.append((leaf, ids))

                if isinstance(r2, AllGroup) or (
                    isinstance(r2, AnyGroup) and len(r2) == 1
                ):
                    if r1 > r2[-1]:
                        intersection = c1_ids.intersection(c2_ids)
                        activites = r2 + [r1]

                        if len(intersection) != 0:
                            # Create a new AllGroup
                            leaf = BinaryExpressionLeaf(
                                lActivities=AnyGroup([l1]),
                                rActivities=AllGroup(activites),
                                operator=op,
                            )
                            newTrees.append((leaf, intersection))

        allCandidates.extend(newTrees)
        extended_trees = newTrees

    return allCandidates


import random


def join_leafs(leafs, nVariants, max_width, nTreeSamples, nLeafSamples):
    andTrees = []
    orTrees = []

    extended_trees = random.sample(leafs, nTreeSamples)

    for k in range(max_width - 1):
        newAndTrees = []
        newOrTrees = []
        print(k)

        for c2 in extended_trees:
            sample_leafs = random.sample(leafs, nLeafSamples)

            for c1 in sample_leafs:
                c1_ids = c1[1]
                c2_ids = c2[1]

                if c1_ids != c2_ids:
                    if not isinstance(c2[0], OperatorNode):
                        union = c1_ids.union(c2_ids)

                        if len(union) < 0.95 * nVariants:
                            tree = OperatorNode(
                                LOperator.OR, children=[c1[0], c2[0]], neg=False
                            )
                            newOrTrees.append((tree, union))

                        intersection = c1_ids.intersection(c2_ids)

                        if len(intersection) > 0.05 * nVariants:
                            tree = OperatorNode(
                                LOperator.AND, children=[c1[0], c2[0]], neg=False
                            )
                            newAndTrees.append((tree, intersection))

                    else:
                        if c2[0].lOp == LOperator.AND:
                            intersection = c1_ids.intersection(c2_ids)

                            if len(intersection) > 0.05 * nVariants:
                                cTree = copy_operator_node(c2[0])
                                cTree.children += [c1[0]]

                                newAndTrees.append((cTree, intersection))

                        else:
                            union = c1_ids.union(c2_ids)

                            if len(union) < 0.95 * nVariants:
                                cTree = copy_operator_node(c2[0])
                                cTree.children += [c1[0]]

                                newOrTrees.append((cTree, union))

        andTrees.extend(newAndTrees)
        orTrees.extend(newOrTrees)

        extended_trees = random.sample(newOrTrees + newAndTrees, nTreeSamples)

    return andTrees, orTrees


def join_trees_into_queries(
    andTrees, orTrees, leafs, nVariants, max_height, nTreeSamples, nJointTreeSamples
):
    extended_AndTrees = random.sample(andTrees, nTreeSamples)
    extended_OrTrees = random.sample(orTrees, nTreeSamples)

    res_AndTrees = []
    res_OrTrees = []

    for k in range(max_height - 1):
        print(k)

        for c2 in extended_AndTrees:
            newOrTrees = []

            samples = random.sample(andTrees + leafs, nJointTreeSamples)

            for c1 in samples:
                c1_ids = c1[1]
                c2_ids = c2[1]

                if c1_ids != c2_ids:
                    union = c1_ids.union(c2_ids)

                    if len(union) < 0.95 * nVariants:
                        tree = OperatorNode(
                            LOperator.OR, children=[c1[0], c2[0]], neg=False
                        )
                        newOrTrees.append((tree, union))

        for c2 in extended_OrTrees:
            newAndTrees = []
            samples = random.sample(orTrees + leafs, nJointTreeSamples)

            for c1 in samples:
                c1_ids = c1[1]
                c2_ids = c2[1]

                if c1_ids != c2_ids:
                    intersection = c1_ids.intersection(c2_ids)

                    if len(intersection) > 0.05 * nVariants:
                        tree = OperatorNode(
                            LOperator.AND, children=[c1[0], c2[0]], neg=False
                        )
                        newAndTrees.append((tree, intersection))

            if len(newOrTrees) > nTreeSamples:
                extended_OrTrees = random.sample(newOrTrees, nTreeSamples)
            else:
                extended_OrTrees = newOrTrees

            if len(newAndTrees) > nTreeSamples:
                extended_AndTrees = random.sample(newAndTrees, nTreeSamples)
            else:
                extended_AndTrees = newAndTrees

            res_AndTrees.extend(newAndTrees)
            res_OrTrees.extend(newOrTrees)

    return res_AndTrees, res_OrTrees
