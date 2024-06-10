from cortado_core.subprocess_discovery.subtree_mining.maximal_connected_components.maximal_connected_check import (
    check_if_valid_tree,
)


def print_mining_results(patterns_1, patterns_2):
    print()
    print(
        "Closed RMO",
        sum(
            [
                len(
                    [
                        pattern
                        for pattern in patterns
                        if pattern.closed and check_if_valid_tree(pattern.tree)
                    ]
                )
                for patterns in patterns_1.values()
            ]
        ),
    )
    print(
        "Maximal RMO",
        sum(
            [
                len(
                    [
                        pattern
                        for pattern in patterns
                        if pattern.maximal and check_if_valid_tree(pattern.tree)
                    ]
                )
                for patterns in patterns_1.values()
            ]
        ),
    )
    print()
    print(
        "Closed CM",
        sum(
            [
                len(
                    [
                        pattern
                        for pattern in patterns
                        if pattern.closed and check_if_valid_tree(pattern.tree)
                    ]
                )
                for patterns in patterns_2.values()
            ]
        ),
    )
    print(
        "Maximal CM",
        sum(
            [
                len(
                    [
                        pattern
                        for pattern in patterns
                        if pattern.maximal and check_if_valid_tree(pattern.tree)
                    ]
                )
                for patterns in patterns_2.values()
            ]
        ),
    )

    print()

    rm_k_patterns_nested = {
        k: {str(pattern): pattern for pattern in patterns}
        for k, patterns in patterns_1.items()
    }
    patterns_2_nested = {
        k: {str(pattern): pattern for pattern in patterns}
        for k, patterns in patterns_2.items()
    }

    for k in patterns_2_nested.keys():
        print()
        print("K:", k)
        print("Total:", "RMO:", len(patterns_1[k]), "CM:", len(patterns_2[k]))
        print(
            "Valid:",
            "RMO:",
            len(
                [
                    pattern
                    for pattern in patterns_1[k]
                    if check_if_valid_tree(pattern.tree)
                ]
            ),
            "CM:",
            len(
                [
                    pattern
                    for pattern in patterns_2[k]
                    if check_if_valid_tree(pattern.tree)
                ]
            ),
        )
        print(
            "Closed:",
            "RMO:",
            len(
                [
                    pattern
                    for pattern in patterns_1[k]
                    if pattern.closed and check_if_valid_tree(pattern.tree)
                ]
            ),
            "CM:",
            len(
                [
                    pattern
                    for pattern in patterns_2[k]
                    if pattern.closed and check_if_valid_tree(pattern.tree)
                ]
            ),
        )
        print(
            "Maxmial:",
            "RMO:",
            len(
                [
                    pattern
                    for pattern in patterns_1[k]
                    if pattern.maximal and check_if_valid_tree(pattern.tree)
                ]
            ),
            "CM:",
            len(
                [
                    pattern
                    for pattern in patterns_2[k]
                    if pattern.maximal and check_if_valid_tree(pattern.tree)
                ]
            ),
        )

        patterns_rmo = set(rm_k_patterns_nested[k].keys())
        patterns_cm = set(patterns_2_nested[k].keys())

        print()
        print("Intersection")
        for pattern in patterns_cm.intersection(patterns_rmo):
            if check_if_valid_tree(rm_k_patterns_nested[k][pattern].tree):
                if (
                    rm_k_patterns_nested[k][pattern].closed
                    != patterns_2_nested[k][pattern].closed
                ):
                    print(
                        pattern,
                        "Closed RMO",
                        rm_k_patterns_nested[k][pattern].closed,
                        "Closed CM",
                        patterns_2_nested[k][pattern].closed,
                    )

                    print(pattern)
                    print(repr(rm_k_patterns_nested[k][pattern].tree))

                if (
                    rm_k_patterns_nested[k][pattern].maximal
                    != patterns_2_nested[k][pattern].maximal
                ):
                    print(
                        pattern,
                        "Maximal RMO",
                        rm_k_patterns_nested[k][pattern].maximal,
                        "Maximal CM",
                        patterns_2_nested[k][pattern].maximal,
                    )

        print()
        print("Only in RMO")
        for pattern in patterns_rmo.difference(patterns_cm):
            if check_if_valid_tree(rm_k_patterns_nested[k][pattern].tree):
                if (
                    rm_k_patterns_nested[k][pattern].closed
                    or rm_k_patterns_nested[k][pattern].maximal
                ):
                    print(
                        pattern,
                        "Closed RMO",
                        rm_k_patterns_nested[k][pattern].closed,
                        "Maximal RMO",
                        rm_k_patterns_nested[k][pattern].maximal,
                    )

    if patterns_cm.difference(patterns_rmo):
        print("CM has pattern not in RMO")
