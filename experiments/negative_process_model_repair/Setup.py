import pm4py.objects.process_tree.importer.importer as ptml_importer
import pm4py.objects.log.importer.xes.importer as xes_importer
from cortado_core.utils.timestamp_utils import TimeUnit
from pm4py.algo.filtering.log.variants import variants_filter

from experiments.negative_process_model_repair.utils import get_c_variants, convert_sequence_variants_to_cvariants


def import_initial_model(file_path):
    pt = ptml_importer.apply(file_path)
    return pt


def import_event_log(file_path):
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    complete_log = xes_importer.apply(file_path, variant=variant, parameters=parameters)

    return complete_log


def get_c_variants_from_log(event_log):
    use_mp = len(event_log) > 1000
    time_granularity = min(TimeUnit)

    # variants = get_variants(event_log)
    variants = get_c_variants(event_log, use_mp, time_granularity)

    return variants


def get_variants_from_log(event_log):
    variants = variants_filter.get_variants(event_log)
    return variants


def get_experiment_variants(event_log):
    variant_information = variants_filter.get_variants(event_log)

    traces_in_variants = []
    c_variants_1 = convert_sequence_variants_to_cvariants(variant_information)

    for v in c_variants_1:
        traces_in_variants.append(
            [
                {"follows": v["follows"]},
                4,
                v["count"],
                v
            ]
        )

    return traces_in_variants
