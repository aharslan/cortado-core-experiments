import glob
import os
from pathlib import Path

import pandas as pd
import pm4py

from experiments.trace_fragments.experiments import (
    evaluate_models,
    COLUMNS,
    eval_function,
)
from pm4py.objects.log.importer.xes.importer import apply as xes_import


LOG_FILE = os.getenv("LOG_FILE", "filtered_log.xes")
EVENT_LOG_DIRECTORY = os.getenv(
    "EVENT_LOG_DIRECTORY",
    "C:\\sources\\arbeit\\cortado\\trace_fragments\\all_results_final\\RoadTrafficFineManagement.xes_0.2\\logs",
)
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "C:\\sources\\arbeit\\cortado\\trace_fragments\\all_results_final\\RoadTrafficFineManagement.xes_0.2\\classical\\etm",
)


def calculate_statistics_for_models():
    results = []
    log = xes_import(os.path.join(EVENT_LOG_DIRECTORY, LOG_FILE))
    for model_file in glob.glob(os.path.join(MODEL_DIR, "*.ptml")):
        percentage_added_traces = Path(model_file).stem
        model = pm4py.read_ptml(model_file)
        results.append([percentage_added_traces] + eval_function(log, model) + [None])

    results_df = pd.DataFrame(results, columns=COLUMNS)
    results_path = os.path.join(MODEL_DIR, "results.csv")
    results_df.to_csv(results_path)


if __name__ == "__main__":
    calculate_statistics_for_models()
