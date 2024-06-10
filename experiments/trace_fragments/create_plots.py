import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", rc={"figure.figsize": (5, 3.5)})


def create_plots(data_path: str, is_fragments_plots: bool = True):
    results_df = pd.read_csv(os.path.join(data_path, "data.csv"))
    create_f_measure_fitness_precision_plots(data_path, results_df, is_fragments_plots)


def create_f_measure_fitness_precision_plots(
    filename: str, df: pd.DataFrame, is_fragments_plots: bool
):
    Path(os.path.join(filename, "plots")).mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset="processed_variants", keep="last")
    create_measurement_plot(filename, "f_measure", df)
    create_measurement_plot(filename, "fitness", df)
    create_measurement_plot(filename, "precision", df)
    create_combined_plot(filename, df, is_fragments_plots)


def create_measurement_plot(filename: str, measure: str, df: pd.DataFrame):
    sns.lineplot(data=df, x="processed_variants", y=measure)
    plt.ylim(0, 1.05)
    plt.savefig(
        os.path.join(filename, "plots", "variants_" + measure + ".pdf"),
        bbox_inches="tight",
    )
    plt.close()


def create_combined_plot(filename: str, df: pd.DataFrame, is_fragments_plots: bool):
    df = df.rename(
        columns={
            "fitness": "Fitness",
            "precision": "Precision",
            "f_measure": "F-Measure",
        }
    )
    plot_df = df.melt(
        id_vars=["processed_variants"], value_vars=["Fitness", "F-Measure", "Precision"]
    )
    plot_df = plot_df.rename(columns={"variable": "Metric"})
    sns.lineplot(data=plot_df, x="processed_variants", y="value", hue="Metric")
    if is_fragments_plots:
        plt.xlabel("Added trace (fragment) variants")
    else:
        plt.xlabel("Added trace (fragment) variants")
    plt.ylabel("Metric")
    plt.ylim(0, 1.05)
    plt.savefig(
        os.path.join(filename, "plots", "combined_plot" + ".pdf"), bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":
    create_plots(
        data_path="C:\\sources\\arbeit\\cortado\\trace_fragments\\BPI_Challenge_2020_RequestForPayment\\inc_infixes_variant_transaction",
        is_fragments_plots=True,
    )
