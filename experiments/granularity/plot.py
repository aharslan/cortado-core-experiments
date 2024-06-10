import itertools
from matplotlib import pyplot as plt
from .config import AVAILABLE_EVENT_LOGS_LABELS, FIGURES_DIR
from cortado_core.utils.timestamp_utils import TimeUnit
from . import config


def bar_chart(data, log_name, file_name, x_label, y_label):
    fig, ax = plt.subplots()

    fig.set_figheight(4)
    fig.set_figwidth(7)

    # Save the chart so we can loop through the bars below.
    bars = ax.bar([unit.value for unit in config.GRANULARITIES], data, width=0.5)

    # Axis formatting.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")
    ax.xaxis.grid(False)

    # Add text annotations to the top of the bars.
    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 3,
            bar.get_height(),
            "{:10.2f}".format(bar.get_height()),
            horizontalalignment="center",
            color="black",
            weight="bold",
        )

    # fig.tight_layout()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.savefig(FIGURES_DIR + log_name + "_" + file_name + ".pdf")
    # plt.show()


def scatter_plot_multiple(x_data, y_data_list, file_name):
    colors = itertools.cycle(["r", "b", "g", "k", "c"])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i, val in enumerate(y_data_list):
        ax1.scatter(
            x_data[i],
            y_data_list[i],
            s=10,
            marker="s",
            color=next(colors),
            label=AVAILABLE_EVENT_LOGS_LABELS[i],
        )
        for idx, unit in enumerate(list(config.GRANULARITIES)):
            ax1.annotate(unit.value, (x_data[i][idx], y_data_list[i][idx]), fontsize=7)

    plt.legend(loc="upper right")
    plt.savefig(FIGURES_DIR + "_" + file_name + ".pdf")
    # plt.show()


def scatter_plot(
    x_data, y_data, y_max, x_label, y_label, file_name, granularity: TimeUnit
):
    index = list(TimeUnit).index(granularity)

    x_data = [[data[index]] for data in x_data]
    y_data = [[data[index]] for data in y_data]

    colors = itertools.cycle(["r", "b", "g", "k", "c"])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, y_max])
    for i, val in enumerate(y_data):
        ax1.scatter(
            x_data[i],
            y_data[i],
            s=10,
            marker="s",
            color=next(colors),
            label=AVAILABLE_EVENT_LOGS_LABELS[i],
        )

    plt.legend(loc="upper left")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(FIGURES_DIR + "_" + file_name + ".pdf")
    # plt.show()


def plot_num_high_level_variants(num_high_level_variants, log_name):
    bar_chart(
        num_high_level_variants,
        log_name,
        "num_variants",
        "Time unit",
        "Num. high level variants",
    )


def plot_num_high_level_conc_variants(num_high_level_conc_variants, log_name):
    bar_chart(
        num_high_level_conc_variants,
        log_name,
        "num_conc_variants",
        "Time unit",
        "Num. high level variants with concurrency",
    )


def plot_execution_time(times, log_name):
    bar_chart(times, log_name, "execution_time", "Time unit", "Time (s)")


def plot_avg_heights(avg_heights, log_name):
    bar_chart(
        avg_heights, log_name, "avg_height", "Time unit", "Avg. height of variants"
    )


def plot_avg_widths(avg_widths, log_name):
    bar_chart(avg_widths, log_name, "avg_width", "Time unit", "Avg. width of variants")


def plot_avg_sub_variant_heights(avg_heights, log_name):
    bar_chart(
        avg_heights,
        log_name,
        "avg_subvariant_height",
        "Time unit",
        "Avg. height of low level variants",
    )


def plot_avg_sub_variant_widths(avg_heights, log_name):
    bar_chart(
        avg_heights,
        log_name,
        "avg_subvariant_width",
        "Time unit",
        "Avg. width of low level variants",
    )
