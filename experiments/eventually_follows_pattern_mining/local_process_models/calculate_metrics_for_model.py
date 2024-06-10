import glob
import os
from pathlib import Path

from pm4py.objects.log.importer.xes.importer import apply as xes_import
import pm4py

from cortado_core.eventually_follows_pattern_mining.local_process_models.metrics import (
    calculate_metrics,
)

MODELS_PATH = "C:\\sources\\arbeit\\cortado\\master_thesis\\external_lpms\\tax\\BPI_Challenge_2012.xes"
LOG_FILE = "C:\\sources\\arbeit\\cortado\\event-logs\\master_thesis\\BPI_Challenge_2012_interval_format.xes"


def print_metrics_for_models():
    log = xes_import(LOG_FILE)
    files = glob.glob(os.path.join(MODELS_PATH, "*.pnml"))
    for file in files:
        print("filename:", Path(file).stem)
        metrics = calculate_metrics(
            pm4py.read_pnml(file),
            log,
            include_skip_metrics=True,
            is_place_net_algorithm=False,
        )
        support_trans = metrics[0]
        print("support_trans:", support_trans)
        print("skip precision", metrics[-4])


if __name__ == "__main__":
    print_metrics_for_models()
