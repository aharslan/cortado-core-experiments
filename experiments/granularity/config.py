from cortado_core.utils.timestamp_utils import TimeUnit

EVENT_LOG_FOLDER = "/app/logs/"
LOG_NAME = "BPI_Challenge_2012.xes"
PICKLE_FILES_DIR = "/app/results/"
FIGURES_DIR = "/app/figures/"
MEASURE_EXECUTION_TIME_ITERATIONS = 10
AVAILABLE_EVENT_LOGS = [
    "2018",
    "BPIC2011.xes",
    "sepsis.xes",
    "Road_Traffic_Fine_Management_Process.xes",
    "BPI_Challenge_2012.xes",
]

AVAILABLE_EVENT_LOGS_LABELS = ["2018", "2011", "Sepsis", "RTFM", "2012"]

GRANULARITIES = [TimeUnit.MS]
PERSIST_RESULTS = False

# keys of the pickle files (for each event log)

RESULT_VARIANTS_KEY = "variants"
RESULT_SUB_VARIANTS_KEY = "variants_with_subvariants_list"
RESULT_VARIANT_COMPUTATION_TIME_KEY = "times"
RESULT_PLAIN_VARIANT_COMPUTATION_TIME_KEY = "times_plain_variants"
RESULT_SUB_VARIANT_COMPUTATION_TIME_KEY = "times_subvariant_calc"
RESULT_NUM_HIGH_LEVEL_VARIANTS_KEY = "num_high_level_variants"
RESULT_NUM_LOW_LEVEL_VARIANTS_KEY = "num_low_level_variants"
RESULT_NUM_HIGH_LEVEL_CONC_VARIANTS_KEY = "num_high_level_conc_variants"
RESULT_NUM_LOW_LEVEL_CONC_VARIANTS_KEY = "num_low_level_conc_variants"
RESULT_AVG_HEIGHTS_KEY = "avg_heights"
RESULT_AVG_WIDTH_KEY = "avg_widths"
RESULT_SUB_VARIANTS_AVG_HEIGHTS_KEY = "avg_subvariant_heights"
RESULT_SUB_VARIANTS_AVG_WIDTHS_KEY = "avg_subvariant_widths"
RESULT_NUM_CASES_KEY = "num_cases"
RESULT_NUM_EVENTS_KEY = "num_events"
RESULT_NUM_PLAIN_VARIANTS_KEY = "num_variants"
RESULT_AVG_LENGTH_PLAIN_VARIANTS_KEY = "avg_variant_length"
