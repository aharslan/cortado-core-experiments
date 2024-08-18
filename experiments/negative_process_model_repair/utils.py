import dataclasses
from typing import Tuple, Callable

from pm4py.objects.log.obj import EventLog, Trace
from pm4py.util.variants_util import variant_to_trace
from tqdm import tqdm
from cortado_core.process_tree_utils.miscellaneous import (
    subtree_is_part_of_tree_based_on_obj_id,
)
from cortado_core.utils.sequentializations import generate_sequentializations
from pm4py.objects.process_tree.obj import Operator, ProcessTree
from cortado_core.utils.cvariants import get_concurrency_variants, get_detailed_variants
from cortado_core.utils.split_graph import Group
from cortado_core.utils.timestamp_utils import TimeUnit
from cortado_core.freezing.apply import add_trace_to_pt_language_with_freezing
from cortado_core.lca_approach import add_trace_to_pt_language
from cortado_core.models.infix_type import InfixType
from cortado_core.utils.trace import TypedTrace
from pm4py.objects.log.obj import EventLog

import multiprocessing
from typing import List

from experiments.negative_process_model_repair.PoolFactory import PoolFactory


@dataclasses.dataclass
class VariantInformation:
    infix_type: InfixType
    is_user_defined: bool


SEQUENCE_CHAR = "\u2794"
CHOICE_CHAR = "\u2715"
LOOP_CHAR = "\u21BA"
PARALLELISM_CHAR = "\u2227"
TAU_CHAR = "\u03C4"


def __convert_operator_string_from_frontend_for_pm4py_core(operator: str) -> str:
    if operator == CHOICE_CHAR:
        return Operator.XOR
    if operator == SEQUENCE_CHAR:
        return Operator.SEQUENCE
    if operator == LOOP_CHAR:
        return Operator.LOOP
    if operator == PARALLELISM_CHAR:
        return Operator.PARALLEL
    return None


def __convert_label_string_from_frontend_for_pm4py_core(label: str) -> str:
    if label == TAU_CHAR:
        return None
    else:
        return label


def dict_to_process_tree(
    pt: dict, res=None, frozen_subtrees=None
) -> Tuple[ProcessTree, List[ProcessTree]]:
    if frozen_subtrees is None:
        frozen_subtrees = []
    if not res:
        res = ProcessTree(
            operator=__convert_operator_string_from_frontend_for_pm4py_core(
                pt["operator"]
            ),
            label=__convert_label_string_from_frontend_for_pm4py_core(pt["label"]),
        )
    else:
        subtree = ProcessTree(
            operator=__convert_operator_string_from_frontend_for_pm4py_core(
                pt["operator"]
            ),
            label=__convert_label_string_from_frontend_for_pm4py_core(pt["label"]),
            parent=res,
        )
        res.children.append(subtree)
        res = subtree
        if pt["frozen"]:
            current_node_already_considered = False
            for frozen_tree in frozen_subtrees:
                if subtree_is_part_of_tree_based_on_obj_id(subtree, frozen_tree):
                    current_node_already_considered = True
                    break
            if not current_node_already_considered:
                frozen_subtrees.append(subtree)
    if pt["children"]:
        for c in pt["children"]:
            dict_to_process_tree(c, res, frozen_subtrees)

    return res, frozen_subtrees


def create_subvariants(ts: list[Trace], time_granularity: TimeUnit):
    sub_vars = get_detailed_variants(ts, time_granularity=time_granularity)

    return sub_vars


def create_variant_object(
    time_granularity: TimeUnit,
    total_traces: int,
    bid: int,
    v: Group,
    ts: list[Trace],
    info: VariantInformation,
):
    sub_variants = create_subvariants(ts, time_granularity)

    # Default value of clusterId in a variant = -1
    variant = {
        "count": len(ts),
        "variant": v.serialize(),
        "bid": bid,
        "length": len(v),
        "number_of_activities": v.number_of_activities(),
        "percentage": round(len(ts) / total_traces * 100, 2),
        "nSubVariants": len(sub_variants.keys()),
        "userDefined": info.is_user_defined,
        "infixType": info.infix_type.value,
        "clusterId": -1,
    }

    # If the variant is only a single activity leaf, wrap it up as a sequence
    if "leaf" in variant["variant"].keys() or "parallel" in variant["variant"].keys():
        variant["variant"] = {"follows": [variant["variant"]]}

    return variant, sub_variants


def variants_to_variant_objects(
    variants: dict[Group, list[Trace]],
    time_granularity: TimeUnit,
    total_traces: int,
    info_generator: Callable[[list[Trace]], VariantInformation],
):
    res_variants = []

    for bid, (v, ts) in enumerate(
        sorted(list(variants.items()), key=lambda e: len(e[1]), reverse=True)
    ):
        info: VariantInformation = info_generator(ts)
        v.infix_type = info.infix_type
        variant, sub_vars = create_variant_object(
            time_granularity, total_traces, bid, v, ts, info
        )

        res_variants.append(variant)

    return sorted(res_variants, key=lambda variant: variant["count"], reverse=True)


def get_c_variants(
    event_log: EventLog,
    use_mp: bool = False,
    time_granularity: TimeUnit = min(TimeUnit),
):
    variants: dict[Group, list[Trace]] = get_concurrency_variants(
        event_log, use_mp, time_granularity, PoolFactory.instance().get_pool()
    )

    total_traces: int = len(event_log)
    info_generator: Callable[
        [list[Trace]], VariantInformation
    ] = lambda _: VariantInformation(
        infix_type=InfixType.NOT_AN_INFIX, is_user_defined=False
    )

    return variants_to_variant_objects(
        variants, time_granularity, total_traces, info_generator
    )


def get_traces_from_variant(variant):
    is_n_sequentialization_reduction_enabled = True
    number_of_sequentializations_per_variant = 10

    n_sequentializations = (
        -1
        if not is_n_sequentialization_reduction_enabled
        else number_of_sequentializations_per_variant
    )

    traces = []
    cvariant = {"follows": variant["follows"]}

    sequentializations = generate_sequentializations(
        Group.deserialize(cvariant), n_sequentializations=n_sequentializations
    )
    traces += [
        TypedTrace(variant_to_trace(seq), InfixType.NOT_AN_INFIX)
        for seq in sequentializations
    ]

    return traces


def convert_sequence_variants_to_cvariants(variants):
    result_c_variants = []

    for variant in variants:
        follows = []
        for activity_name in variant:
            follows.append({"leaf": [activity_name]})

        result_c_variants.append(
            {
                "follows": follows,
                "count": len(variants[variant]),
                "variant": variant,
                "traces": variants[variant],
            }
        )

    result_c_variants.sort(reverse=True, key=lambda x: x["count"])

    return result_c_variants


def add_variants_to_process_model(
    pt: ProcessTree,
    frozen_subtrees: List[ProcessTree],
    fitting_traces: List[TypedTrace],
    traces_to_be_added: List[TypedTrace],
    pool: multiprocessing.pool.Pool,
):
    frozen_subtrees_are_present = len(frozen_subtrees) > 0

    description = "adding variants to process tree without frozen subtrees"
    if frozen_subtrees_are_present:
        description = "adding variants to process tree including frozen subtrees"

    for t in tqdm(traces_to_be_added, desc=description):
        if not frozen_subtrees_are_present:
            pt = add_trace_to_pt_language(
                pt, fitting_traces, t, try_pulling_lca_down=True, pool=pool
            )
        else:
            # TODO fix format and check how to adapt for infixes
            pt, frozen_subtrees = add_trace_to_pt_language_with_freezing(
                pt,
                frozen_subtrees,
                EventLog(
                    [
                        t.trace
                        for t in fitting_traces
                        if t.infix_type == InfixType.NOT_AN_INFIX
                    ]
                ),
                t.trace,
                try_pulling_lca_down=True,
                pool=pool,
            )
        fitting_traces.append(t)
    # res = process_tree_to_dict(pt, frozen_subtrees)
    return pt
