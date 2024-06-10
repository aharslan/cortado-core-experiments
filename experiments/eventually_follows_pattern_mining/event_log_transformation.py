from copy import copy

import pm4py
from pm4py.objects.log.importer.xes.importer import apply as xes_import
from pm4py.objects.log.obj import Trace, EventLog
from pm4py.objects.log.util.interval_lifecycle import to_interval
from pm4py.util.xes_constants import DEFAULT_TRANSITION_KEY

LOG_PATH = "C:\\sources\\arbeit\\cortado\\event-logs\\sepsis_cases.xes"
SAVE_PATH = "C:\\sources\\arbeit\\cortado\\event-logs\\sepsis_cases_interval_format.xes"


def transform_event_log():
    log = xes_import(LOG_PATH)

    if log.attributes.get("PM4PY_TYPE", "") != "interval":
        if DEFAULT_TRANSITION_KEY in log[0][0]:
            traces = [
                Trace(
                    [
                        e
                        for e in trace
                        if e[DEFAULT_TRANSITION_KEY].lower() == "start"
                        or e[DEFAULT_TRANSITION_KEY].lower() == "complete"
                    ],
                    attributes=trace.attributes,
                    properties=trace.properties,
                )
                for trace in log
            ]
            log = EventLog(
                traces,
                attributes=copy(log.attributes),
                extensions=log.extensions,
                classifiers=log.classifiers,
                omni_present=log.omni_present,
                properties=log.properties,
            )

    interval_log = to_interval(log)
    pm4py.write_xes(interval_log, SAVE_PATH)


if __name__ == "__main__":
    transform_event_log()
