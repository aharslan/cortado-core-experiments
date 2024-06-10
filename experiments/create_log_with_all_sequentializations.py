from collections import defaultdict

import pm4py
from pm4py.util.xes_constants import DEFAULT_START_TIMESTAMP_KEY, DEFAULT_TIMESTAMP_KEY

from cortado_core.utils.sequentializations import generate_variants
from cortado_core.utils.split_graph import Group
from pm4py.objects.log.obj import Trace, Event
from cortado_core.utils.cvariants import get_concurrency_variants
from pm4py.objects.log.exporter.xes.variants import line_by_line

from typing import Tuple, List, Dict
import os
import argparse
from tqdm import tqdm
import threading
import ctypes


def create_log_with_al_sequentializations(
    log_filename: str, use_multiprocessing: bool, store_event_attributes: bool
):
    print(use_multiprocessing)
    event_log = pm4py.read_xes(log_filename)
    variants: Dict[Group, List[Trace]] = get_concurrency_variants(
        event_log, use_multiprocessing
    )

    log_exporter_sequentializations = LogExporter(
        f"{os.path.splitext(log_filename)[0]}_CORTADO_SEQUENTIALIZATIONS.xes"
    )
    log_exporter_traces_no_timeout = LogExporter(
        f"{os.path.splitext(log_filename)[0]}_CORTADO_WITHOUT_TIMEOUT_VARIANTS.xes"
    )

    with log_exporter_sequentializations, log_exporter_traces_no_timeout:
        n_timeouted_variants = 0

        for variant, variant_traces in tqdm(variants.items()):
            try:
                get_sequentializations = lambda v: generate_variants(
                    variant.serialize(include_performance=False)
                )
                sequentializations = execute_with_timeout(
                    get_sequentializations, args=(variant,), timeout=2
                )

                for trace in variant_traces:
                    log_exporter_traces_no_timeout.export_trace(trace)
                    for sequentialization in sequentializations:
                        log_exporter_sequentializations.export_trace(
                            convert_to_pm4py_trace(
                                sequentialization, trace, store_event_attributes
                            )
                        )

            except TimeoutException:
                n_timeouted_variants += 1
                continue

    print(f"Sequentialization finished, {n_timeouted_variants} variants with timeout")


def convert_to_pm4py_trace(
    trace_as_list: List[str], original_trace: Trace, add_event_attributes: bool
):
    trace = Trace(
        attributes=original_trace.attributes, properties=original_trace.properties
    )

    if not add_event_attributes:
        for activity_name in trace_as_list:
            event = Event()
            event["concept:name"] = activity_name

            trace.append(event)

        return trace

    original_events = sorted(
        original_trace._list,
        key=lambda x: (x[DEFAULT_START_TIMESTAMP_KEY], x[DEFAULT_TIMESTAMP_KEY]),
    )
    act_events = get_activity_events_dict(original_events)
    act_counter = defaultdict(int)

    for activity_name in trace_as_list:
        trace.append(act_events[activity_name][act_counter[activity_name]])
        act_counter[activity_name] += 1

    return trace


def get_activity_events_dict(events: List[Event]) -> Dict[str, List[Event]]:
    act_events = dict()

    for event in events:
        act_name = event["concept:name"]
        act_events[act_name] = act_events.get(act_name, []) + [event]

    return act_events


class TimeoutException(BaseException):
    pass


def execute_with_timeout(func, timeout, args=()):
    result = []
    exception = []

    def func_wrapper(*args_wrapper):
        try:
            result.append(func(*args_wrapper))
        except Exception as e:
            exception.append(e)

    t = threading.Thread(target=func_wrapper, args=args)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise_in_thread(t, TimeoutException())

    if len(result) != 0:
        return result[0]

    if len(exception) != 0:
        raise exception[0]

    raise TimeoutException()


def raise_in_thread(thread, exception):
    # see https://stackoverflow.com/questions/36484151/throw-an-exception-into-another-thread
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), ctypes.py_object(exception)
    )
    if ret == 0:
        raise ValueError("Invalid thread ID")
    elif ret > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class LogExporter:
    def __init__(self, filename: str):
        self._filename = filename

    def __enter__(self):
        if os.path.exists(self._filename):
            os.remove(self._filename)

        self._file = open(self._filename, "ab")

        lines = [
            '<?xml version="1.0" encoding="utf-8" ?>\n',
            '<log xes.version="1849-2016" xes.features="nested-attributes" xmlns="http://www.xes-standard.org/">\n',
        ]
        lines = [line.encode("utf-8") for line in lines]

        self._file.writelines(lines)

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.write("</log>".encode("utf-8"))
        self._file.close()

    def export_trace(self, trace: Trace):
        line_by_line.export_trace_line_by_line(trace, self._file, "utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all cortado sequentializations for a given log"
    )
    parser.add_argument(
        "--filename",
        dest="filename",
        required=True,
        help="filename of the input .xes-file",
    )
    parser.add_argument(
        "--mp",
        dest="use_multiprocessing",
        default=False,
        action="store_true",
        help="indicates that multiprocessing should be used",
    )
    parser.add_argument(
        "--add-event-attributes",
        dest="add_event_attributes",
        default=False,
        action="store_true",
        help="indicates that event attributes should be stored",
    )

    args = parser.parse_args()

    create_log_with_al_sequentializations(
        args.filename, args.use_multiprocessing, args.add_event_attributes
    )
