import os
from pathlib import Path

import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_plots(filename: str):
    results_df = pd.read_csv(filename + "_results.csv")
    create_f_measure_fitness_precision_plots(filename, results_df)


def create_f_measure_fitness_precision_plots(filename: str, df: pd.DataFrame):
    Path(os.path.join(filename, "plots")).mkdir(parents=True, exist_ok=True)
    create_measurement_plot(filename, "f_measure", df)
    create_measurement_plot(filename, "fitness", df)
    create_measurement_plot(filename, "precision", df)


def create_measurement_plot(filename: str, measure: str, df: pandas.DataFrame):
    sns.lineplot(
        data=df, x="processed_variants", y=measure, hue="strategy", style="strategy"
    )
    plt.savefig(
        os.path.join(filename, "plots", "variants_" + measure + ".pdf"),
        bbox_inches="tight",
    )
    plt.close()

    sns.lineplot(
        data=df, x="processed_traces", y=measure, hue="strategy", style="strategy"
    )
    plt.savefig(
        os.path.join(filename, "plots", "traces_" + measure + ".pdf"),
        bbox_inches="tight",
    )
    plt.close()
