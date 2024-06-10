from cortado_core.subprocess_discovery.subtree_mining.obj import (
    FrequencyCountingStrategy,
)


def get_support_count(
    rel_support: float,
    frequency_strategy: FrequencyCountingStrategy,
    n_traces: int,
    n_variants: int,
) -> int:
    if (
        frequency_strategy == FrequencyCountingStrategy.TraceOccurence
        or frequency_strategy == FrequencyCountingStrategy.TraceTransaction
    ):
        return round(n_traces * rel_support)

    return round(n_variants * rel_support)
