import random

from pm4py.objects.log.importer.xes import importer as importer
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.log import case_statistics
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def create_variant_log(log_path, log_file_name, sort=True):
    log = importer.apply(log_path + log_file_name)
    variants = variants_filter.get_variants(log)
    variants_count = case_statistics.get_variant_statistics(log)
    if sort:
        variants_count = sorted(variants_count, key=lambda x: x["count"], reverse=True)
    #    else:
    #        random.shuffle(variants_count)
    print(len(variants), " variants found")

    first_variant = True
    variant_event_log = EventLog()
    for v in variants_count:
        # take first trace of specified variant
        trace = variants[v["variant"]][0]
        variant_event_log.append(trace)
        # store only first variance in an event log for usage in PROM
        if first_variant:
            if sort:
                variant_log_file = log_path + log_file_name.replace(
                    ".xes", "_first_variant_only_once_sorted.xes"
                )
            else:
                variant_log_file = log_path + log_file_name.replace(
                    ".xes", "_first_variant_only_once_unsorted.xes"
                )
            xes_exporter.apply(variant_event_log, variant_log_file)
        first_variant = False

    if sort:
        variant_log_file = log_path + log_file_name.replace(
            ".xes", "_each_variant_only_once_sorted.xes"
        )
    else:
        variant_log_file = log_path + log_file_name.replace(
            ".xes", "_each_variant_only_once_unsorted.xes"
        )
    xes_exporter.apply(variant_event_log, variant_log_file)


if __name__ == "__main__":
    create_variant_log("road_traffic_fine_management/", "log.xes", True)
    create_variant_log("road_traffic_fine_management/", "log.xes", False)
