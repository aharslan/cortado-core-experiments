from experiments.eventually_follows_pattern_mining.algorithms_comparison import (
    compare_algorithms,
)
from experiments.eventually_follows_pattern_mining.algorithms_memory_comparison import (
    compare_memory_of_algorithms,
)
from experiments.eventually_follows_pattern_mining.local_process_models.lpms import (
    build_local_process_models,
)
from experiments.eventually_follows_pattern_mining.overlap_support_count import (
    overlap_experiments,
)
from experiments.eventually_follows_pattern_mining.time_granularity_experiments import (
    run_time_granularity_experiments,
)
from experiments.eventually_follows_pattern_mining.time_granularity_length_distribution import (
    time_granularity_distribution_experiments,
)

if __name__ == "__main__":
    print("--- start overlap experiments ---")
    # overlap_experiments()
    print("--- start comparison experiments ---")
    # support_dict = compare_algorithms()
    print("--- start memory allocation experiments ---")
    # compare_memory_of_algorithms(support_dict)
    print("--- start lpm experiments ---")
    build_local_process_models()
    print("--- start time granularity experiments ---")
    # time_granularity_distribution_experiments()
    # run_time_granularity_experiments()
