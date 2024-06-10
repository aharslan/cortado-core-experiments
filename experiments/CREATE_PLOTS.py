from typing import Dict, List, Tuple

import matplotlib as mpl
from matplotlib import pyplot as plt
import pickle
from cycler import cycler

mpl.rcParams["axes.prop_cycle"] = cycler(color="bgrcmyk")
mpl.rcParams["figure.figsize"] = [6.4 * 0.7, 4.8 * 0.7]
# plt.rc('text', usetex=True)
plt.rc("font", family="serif")


def create_line_plot(
    data: List[Dict],
    x_axis_name="",
    y_axis_name="",
    title="",
    path_to_store: str = None,
    svg: bool = False,
    y_scale: Tuple[float, float] = None,
):
    # fig = plt.figure(figsize=(3, 6))

    legend_names = []

    for e in data:
        plt.step(e["x"], e["y"], "-")
        if "name" in e:
            legend_names.append(e["name"])
        # plt.fill_between(x, y, step="pre", alpha=0.4)

    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    if legend_names:
        plt.legend(tuple(legend_names))
    plt.grid(True, linestyle="-")
    if y_scale:
        axes = plt.gca()
        axes.set_ylim([y_scale[0], y_scale[1]])
    if path_to_store:
        if svg:
            plt.savefig(path_to_store + ".svg", bbox_inches="tight")
        else:
            plt.savefig(path_to_store + ".pdf", bbox_inches="tight")
    else:
        plt.show()
    plt.clf()
    plt.close()


def extract_x_y_axis(csv_file_path: str, x_key: str, y_key: str):
    x = []
    y = []
    with open(csv_file_path, "rb") as input_file:
        data = pickle.load(input_file)
        data = sorted(data, key=lambda k: k[x_key])
        for entry in data:
            x.append(entry[x_key])
            y.append(entry[y_key])
    return x, y


def plot_attribute(
    results_file_path,
    result_file_name,
    attribute,
    save_as_file=False,
    x_axis_unit="variants",
    model_repair_y=None,
):
    data_to_plot = []
    x, y = extract_x_y_axis(
        results_file_path + result_file_name,
        x_axis_unit + "_processed_so_far",
        "incremental_" + attribute,
    )
    data_to_plot.append({"name": "LCA Approach", "x": x, "y": y})
    x, y = extract_x_y_axis(
        results_file_path + result_file_name,
        x_axis_unit + "_processed_so_far",
        "im_" + attribute,
    )
    data_to_plot.append({"name": "IM Algorithm", "x": x, "y": y})
    if model_repair_y:
        data_to_plot.append(
            {"name": "Model repair algorithm", "x": x, "y": model_repair_y}
        )
    path_to_store = None
    if save_as_file:
        filename = result_file_name + "_" + attribute
        path_to_store = results_file_path + filename
    create_line_plot(
        data_to_plot,
        x_axis_name="processed " + x_axis_unit,
        y_axis_name=attribute.replace("_", " "),
        # y_scale=(0, 1),
        # title="Road Traffic Fine Management Event Log",
        path_to_store=path_to_store,
        svg=True,
    )


def plot_variants_log_coverage(results_file_path, path_to_store=None):
    res = pickle.load(open(results_file_path, "rb"))
    data_to_plot = []
    total_number_traces = res[-1]["traces_processed_so_far"]
    x = []
    y = []
    for r in res:
        x.append(r["variants_processed_so_far"])
        y.append(r["traces_processed_so_far"] / total_number_traces)

    data_to_plot.append({"x": x, "y": y})
    create_line_plot(
        data_to_plot,
        x_axis_name="processed variants",
        y_axis_name="% covered traces",
        # y_scale=(0, 1),
        # title="Road Traffic Fine Management Event Log",
        path_to_store=path_to_store,
        svg=True,
    )


if __name__ == "__main__":
    save = True
    x_axis_unit = "variants"
    result_file_name = "results_sorted_most_frequent_var.pkl"
    # result_file_name = "results_unsorted.pkl"
    log_path = "logs/hospital_billing/"
    log_path = "logs/road_traffic_fine_management/"

    # plot_variants_log_coverage(log_path+result_file_name,path_to_store=log_path+"log_coverage_variants")

    model_repair_result_sorted = [
        {
            "precision": 1.0,
            "fitness": {
                "percFitTraces": 37.56201369954113,
                "averageFitness": 0.6730895408884454,
            },
            "f_measure": 0.8046067164235823,
        },
        {
            "precision": 0.5691910622594656,
            "fitness": {
                "percFitTraces": 90.36975460530691,
                "averageFitness": 0.9807716304216332,
            },
            "f_measure": 0.7203353329595209,
        },
        {
            "precision": 0.5691910622594656,
            "fitness": {
                "percFitTraces": 90.36975460530691,
                "averageFitness": 0.9807716304216332,
            },
            "f_measure": 0.7203353329595209,
        },
        {
            "precision": 0.5691910622594656,
            "fitness": {
                "percFitTraces": 90.36975460530691,
                "averageFitness": 0.9807716304216332,
            },
            "f_measure": 0.7203353329595209,
        },
        {
            "precision": 0.5718570454168382,
            "fitness": {
                "percFitTraces": 92.94939150096428,
                "averageFitness": 0.9847309015026183,
            },
            "f_measure": 0.7235380499744869,
        },
        {
            "precision": 0.5730538051764729,
            "fitness": {
                "percFitTraces": 95.5050874509543,
                "averageFitness": 0.9884106497800684,
            },
            "f_measure": 0.7254887962841046,
        },
        {
            "precision": 0.5730538051764729,
            "fitness": {
                "percFitTraces": 95.5050874509543,
                "averageFitness": 0.9884106497800684,
            },
            "f_measure": 0.7254887962841046,
        },
        {
            "precision": 0.6040795584632439,
            "fitness": {
                "percFitTraces": 97.27272727272727,
                "averageFitness": 0.9949557471985868,
            },
            "f_measure": 0.7517437874324179,
        },
        {
            "precision": 0.6049776817497743,
            "fitness": {
                "percFitTraces": 98.2882223847842,
                "averageFitness": 0.9964123573293976,
            },
            "f_measure": 0.7528549863474756,
        },
        {
            "precision": 0.6049776817497743,
            "fitness": {
                "percFitTraces": 98.2882223847842,
                "averageFitness": 0.9964123573293976,
            },
            "f_measure": 0.7528549863474756,
        },
        {
            "precision": 0.6145007495669191,
            "fitness": {
                "percFitTraces": 98.5908093369688,
                "averageFitness": 0.9969835977617854,
            },
            "f_measure": 0.7603513731251594,
        },
        {
            "precision": 0.5748550841484732,
            "fitness": {
                "percFitTraces": 98.87145042229169,
                "averageFitness": 0.9976652496870371,
            },
            "f_measure": 0.7294187918855112,
        },
        {
            "precision": 0.5516162409218719,
            "fitness": {
                "percFitTraces": 99.15541663895723,
                "averageFitness": 0.9988502292423566,
            },
            "f_measure": 0.7107306340397771,
        },
        {
            "precision": 0.5932028257313948,
            "fitness": {
                "percFitTraces": 99.35026933563876,
                "averageFitness": 0.9991375467615617,
            },
            "f_measure": 0.7444277948004032,
        },
        {
            "precision": 0.5932028257313948,
            "fitness": {
                "percFitTraces": 99.35026933563876,
                "averageFitness": 0.9991375467615617,
            },
            "f_measure": 0.7444277948004032,
        },
        {
            "precision": 0.5795141017780503,
            "fitness": {
                "percFitTraces": 99.51519585023608,
                "averageFitness": 0.999376431806014,
            },
            "f_measure": 0.7336198715455525,
        },
        {
            "precision": 0.5795141017780503,
            "fitness": {
                "percFitTraces": 99.51519585023608,
                "averageFitness": 0.999376431806014,
            },
            "f_measure": 0.7336198715455525,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5650632254138082,
            "fitness": {
                "percFitTraces": 99.67613220722218,
                "averageFitness": 0.9996065232178591,
            },
            "f_measure": 0.7219937455148348,
        },
        {
            "precision": 0.5732547801166995,
            "fitness": {
                "percFitTraces": 99.81711777615216,
                "averageFitness": 0.9997846417730673,
            },
            "f_measure": 0.72869289479744,
        },
        {
            "precision": 0.5732547801166995,
            "fitness": {
                "percFitTraces": 99.81711777615216,
                "averageFitness": 0.9997846417730673,
            },
            "f_measure": 0.72869289479744,
        },
        {
            "precision": 0.5732547801166995,
            "fitness": {
                "percFitTraces": 99.81711777615216,
                "averageFitness": 0.9997846417730673,
            },
            "f_measure": 0.72869289479744,
        },
        {
            "precision": 0.5732547801166995,
            "fitness": {
                "percFitTraces": 99.81711777615216,
                "averageFitness": 0.9997846417730673,
            },
            "f_measure": 0.72869289479744,
        },
        {
            "precision": 0.5732547801166995,
            "fitness": {
                "percFitTraces": 99.81711777615216,
                "averageFitness": 0.9997846417730673,
            },
            "f_measure": 0.72869289479744,
        },
        {
            "precision": 0.5732555541751249,
            "fitness": {
                "percFitTraces": 99.87497506151493,
                "averageFitness": 0.9998567417043485,
            },
            "f_measure": 0.7287126698619049,
        },
        {
            "precision": 0.5732152561611918,
            "fitness": {
                "percFitTraces": 99.89359579703398,
                "averageFitness": 0.9998805163934488,
            },
            "f_measure": 0.7286864237188682,
        },
        {
            "precision": 0.5732152561611918,
            "fitness": {
                "percFitTraces": 99.89359579703398,
                "averageFitness": 0.9998805163934488,
            },
            "f_measure": 0.7286864237188682,
        },
        {
            "precision": 0.5732152561611918,
            "fitness": {
                "percFitTraces": 99.89359579703398,
                "averageFitness": 0.9998805163934488,
            },
            "f_measure": 0.7286864237188682,
        },
        {
            "precision": 0.5732152561611918,
            "fitness": {
                "percFitTraces": 99.89359579703398,
                "averageFitness": 0.9998805163934488,
            },
            "f_measure": 0.7286864237188682,
        },
        {
            "precision": 0.5732152561611918,
            "fitness": {
                "percFitTraces": 99.89359579703398,
                "averageFitness": 0.9998805163934488,
            },
            "f_measure": 0.7286864237188682,
        },
        {
            "precision": 0.5732152561611918,
            "fitness": {
                "percFitTraces": 99.89359579703398,
                "averageFitness": 0.9998805163934488,
            },
            "f_measure": 0.7286864237188682,
        },
        {
            "precision": 0.5732152561611918,
            "fitness": {
                "percFitTraces": 99.89359579703398,
                "averageFitness": 0.9998805163934488,
            },
            "f_measure": 0.7286864237188682,
        },
        {
            "precision": 0.5732152561611918,
            "fitness": {
                "percFitTraces": 99.89359579703398,
                "averageFitness": 0.9998805163934488,
            },
            "f_measure": 0.7286864237188682,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5730946119451191,
            "fitness": {
                "percFitTraces": 99.9275121367294,
                "averageFitness": 0.9999173766499347,
            },
            "f_measure": 0.7285987202935393,
        },
        {
            "precision": 0.5717087249417102,
            "fitness": {
                "percFitTraces": 99.95078805612822,
                "averageFitness": 0.9999425687182545,
            },
            "f_measure": 0.7274843895500103,
        },
        {
            "precision": 0.5717087249417102,
            "fitness": {
                "percFitTraces": 99.95078805612822,
                "averageFitness": 0.9999425687182545,
            },
            "f_measure": 0.7274843895500103,
        },
        {
            "precision": 0.5717087249417102,
            "fitness": {
                "percFitTraces": 99.95078805612822,
                "averageFitness": 0.9999425687182545,
            },
            "f_measure": 0.7274843895500103,
        },
        {
            "precision": 0.5717087249417102,
            "fitness": {
                "percFitTraces": 99.95078805612822,
                "averageFitness": 0.9999425687182545,
            },
            "f_measure": 0.7274843895500103,
        },
        {
            "precision": 0.5717087249417102,
            "fitness": {
                "percFitTraces": 99.95078805612822,
                "averageFitness": 0.9999425687182545,
            },
            "f_measure": 0.7274843895500103,
        },
        {
            "precision": 0.5717087249417102,
            "fitness": {
                "percFitTraces": 99.95078805612822,
                "averageFitness": 0.9999425687182545,
            },
            "f_measure": 0.7274843895500103,
        },
        {
            "precision": 0.5716967871127465,
            "fitness": {
                "percFitTraces": 99.95876837135067,
                "averageFitness": 0.9999532830303587,
            },
            "f_measure": 0.7274775600897547,
        },
        {
            "precision": 0.5716967871127465,
            "fitness": {
                "percFitTraces": 99.95876837135067,
                "averageFitness": 0.9999532830303587,
            },
            "f_measure": 0.7274775600897547,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5715167436159885,
            "fitness": {
                "percFitTraces": 99.96674868657313,
                "averageFitness": 0.9999612633455812,
            },
            "f_measure": 0.7273338887820303,
        },
        {
            "precision": 0.5589977579420653,
            "fitness": {
                "percFitTraces": 99.9700738179158,
                "averageFitness": 0.9999676919328438,
            },
            "f_measure": 0.7171162104327764,
        },
        {
            "precision": 0.5589977579420653,
            "fitness": {
                "percFitTraces": 99.9700738179158,
                "averageFitness": 0.9999676919328438,
            },
            "f_measure": 0.7171162104327764,
        },
        {
            "precision": 0.5589977579420653,
            "fitness": {
                "percFitTraces": 99.9700738179158,
                "averageFitness": 0.9999676919328438,
            },
            "f_measure": 0.7171162104327764,
        },
        {
            "precision": 0.5589977579420653,
            "fitness": {
                "percFitTraces": 99.9700738179158,
                "averageFitness": 0.9999676919328438,
            },
            "f_measure": 0.7171162104327764,
        },
        {
            "precision": 0.5589920038196483,
            "fitness": {
                "percFitTraces": 99.9753940280641,
                "averageFitness": 0.9999730860348,
            },
            "f_measure": 0.7171128625856514,
        },
        {
            "precision": 0.5589920038196483,
            "fitness": {
                "percFitTraces": 99.9753940280641,
                "averageFitness": 0.9999730860348,
            },
            "f_measure": 0.7171128625856514,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.5218139505702418,
            "fitness": {
                "percFitTraces": 99.98137926448095,
                "averageFitness": 0.9999788294434828,
            },
            "f_measure": 0.685773924454818,
        },
        {
            "precision": 0.521804396172443,
            "fitness": {
                "percFitTraces": 99.98403936955509,
                "averageFitness": 0.999981877480547,
            },
            "f_measure": 0.6857663901903568,
        },
        {
            "precision": 0.521804396172443,
            "fitness": {
                "percFitTraces": 99.98403936955509,
                "averageFitness": 0.999981877480547,
            },
            "f_measure": 0.6857663901903568,
        },
        {
            "precision": 0.521804396172443,
            "fitness": {
                "percFitTraces": 99.98403936955509,
                "averageFitness": 0.999981877480547,
            },
            "f_measure": 0.6857663901903568,
        },
        {
            "precision": 0.521804396172443,
            "fitness": {
                "percFitTraces": 99.98403936955509,
                "averageFitness": 0.999981877480547,
            },
            "f_measure": 0.6857663901903568,
        },
        {
            "precision": 0.521804396172443,
            "fitness": {
                "percFitTraces": 99.98403936955509,
                "averageFitness": 0.999981877480547,
            },
            "f_measure": 0.6857663901903568,
        },
        {
            "precision": 0.5321142819314292,
            "fitness": {
                "percFitTraces": 99.98736450089778,
                "averageFitness": 0.9999864957185229,
            },
            "f_measure": 0.6946110906967524,
        },
        {
            "precision": 0.5321142819314292,
            "fitness": {
                "percFitTraces": 99.98736450089778,
                "averageFitness": 0.9999864957185229,
            },
            "f_measure": 0.6946110906967524,
        },
        {
            "precision": 0.5321142819314292,
            "fitness": {
                "percFitTraces": 99.98736450089778,
                "averageFitness": 0.9999864957185229,
            },
            "f_measure": 0.6946110906967524,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5321020402466999,
            "fitness": {
                "percFitTraces": 99.99135465850901,
                "averageFitness": 0.9999906783565421,
            },
            "f_measure": 0.6946016696251813,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.5320982808775259,
            "fitness": {
                "percFitTraces": 99.99201968477755,
                "averageFitness": 0.9999914172746184,
            },
            "f_measure": 0.6945986447997976,
        },
        {
            "precision": 0.532079675252582,
            "fitness": {
                "percFitTraces": 99.99334973731463,
                "averageFitness": 0.9999925256517327,
            },
            "f_measure": 0.694583059453363,
        },
        {
            "precision": 0.532079675252582,
            "fitness": {
                "percFitTraces": 99.99334973731463,
                "averageFitness": 0.9999925256517327,
            },
            "f_measure": 0.694583059453363,
        },
        {
            "precision": 0.532079675252582,
            "fitness": {
                "percFitTraces": 99.99334973731463,
                "averageFitness": 0.9999925256517327,
            },
            "f_measure": 0.694583059453363,
        },
        {
            "precision": 0.5326254157299768,
            "fitness": {
                "percFitTraces": 99.9946797898517,
                "averageFitness": 0.9999939615039036,
            },
            "f_measure": 0.6950482388324964,
        },
        {
            "precision": 0.5326254157299768,
            "fitness": {
                "percFitTraces": 99.9946797898517,
                "averageFitness": 0.9999939615039036,
            },
            "f_measure": 0.6950482388324964,
        },
        {
            "precision": 0.5326254157299768,
            "fitness": {
                "percFitTraces": 99.9946797898517,
                "averageFitness": 0.9999939615039036,
            },
            "f_measure": 0.6950482388324964,
        },
        {
            "precision": 0.5326254157299768,
            "fitness": {
                "percFitTraces": 99.9946797898517,
                "averageFitness": 0.9999939615039036,
            },
            "f_measure": 0.6950482388324964,
        },
        {
            "precision": 0.5326254157299768,
            "fitness": {
                "percFitTraces": 99.9946797898517,
                "averageFitness": 0.9999939615039036,
            },
            "f_measure": 0.6950482388324964,
        },
        {
            "precision": 0.5326254157299768,
            "fitness": {
                "percFitTraces": 99.9946797898517,
                "averageFitness": 0.9999939615039036,
            },
            "f_measure": 0.6950482388324964,
        },
        {
            "precision": 0.5311020430992779,
            "fitness": {
                "percFitTraces": 99.99600984238877,
                "averageFitness": 0.9999953973560743,
            },
            "f_measure": 0.6937502272458048,
        },
        {
            "precision": 0.5311020430992779,
            "fitness": {
                "percFitTraces": 99.99600984238877,
                "averageFitness": 0.9999953973560743,
            },
            "f_measure": 0.6937502272458048,
        },
        {
            "precision": 0.5311020430992779,
            "fitness": {
                "percFitTraces": 99.99600984238877,
                "averageFitness": 0.9999953973560743,
            },
            "f_measure": 0.6937502272458048,
        },
        {
            "precision": 0.5311020430992779,
            "fitness": {
                "percFitTraces": 99.99600984238877,
                "averageFitness": 0.9999953973560743,
            },
            "f_measure": 0.6937502272458048,
        },
        {
            "precision": 0.5311020430992779,
            "fitness": {
                "percFitTraces": 99.99600984238877,
                "averageFitness": 0.9999953973560743,
            },
            "f_measure": 0.6937502272458048,
        },
        {
            "precision": 0.5311020430992779,
            "fitness": {
                "percFitTraces": 99.99600984238877,
                "averageFitness": 0.9999953973560743,
            },
            "f_measure": 0.6937502272458048,
        },
        {
            "precision": 0.5311020430992779,
            "fitness": {
                "percFitTraces": 99.99600984238877,
                "averageFitness": 0.9999953973560743,
            },
            "f_measure": 0.6937502272458048,
        },
        {
            "precision": 0.5311009702755001,
            "fitness": {
                "percFitTraces": 99.99667486865731,
                "averageFitness": 0.9999960019254093,
            },
            "f_measure": 0.693749457463515,
        },
        {
            "precision": 0.5311009702755001,
            "fitness": {
                "percFitTraces": 99.99667486865731,
                "averageFitness": 0.9999960019254093,
            },
            "f_measure": 0.693749457463515,
        },
        {
            "precision": 0.5311009702755001,
            "fitness": {
                "percFitTraces": 99.99667486865731,
                "averageFitness": 0.9999960019254093,
            },
            "f_measure": 0.693749457463515,
        },
        {
            "precision": 0.5311009702755001,
            "fitness": {
                "percFitTraces": 99.99667486865731,
                "averageFitness": 0.9999960019254093,
            },
            "f_measure": 0.693749457463515,
        },
        {
            "precision": 0.5311009702755001,
            "fitness": {
                "percFitTraces": 99.99667486865731,
                "averageFitness": 0.9999960019254093,
            },
            "f_measure": 0.693749457463515,
        },
        {
            "precision": 0.5311009702755001,
            "fitness": {
                "percFitTraces": 99.99667486865731,
                "averageFitness": 0.9999960019254093,
            },
            "f_measure": 0.693749457463515,
        },
        {
            "precision": 0.5311009702755001,
            "fitness": {
                "percFitTraces": 99.99667486865731,
                "averageFitness": 0.9999960019254093,
            },
            "f_measure": 0.693749457463515,
        },
        {
            "precision": 0.5311009702755001,
            "fitness": {
                "percFitTraces": 99.99667486865731,
                "averageFitness": 0.9999960019254093,
            },
            "f_measure": 0.693749457463515,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308578889617674,
            "fitness": {
                "percFitTraces": 99.99866994746293,
                "averageFitness": 0.9999985221638475,
            },
            "f_measure": 0.6935426478704902,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5308076044501249,
            "fitness": {
                "percFitTraces": 99.99933497373146,
                "averageFitness": 0.9999992610819237,
            },
            "f_measure": 0.693499910640005,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
        {
            "precision": 0.5307944303526851,
            "fitness": {"percFitTraces": 100.0, "averageFitness": 1.0},
            "f_measure": 0.6934888445215907,
        },
    ]

    model_repair_f_measure = []
    model_repair_precision = []
    model_repair_fitness = []
    for v in model_repair_result_sorted:
        model_repair_precision.append(v["precision"])
        model_repair_fitness.append(v["fitness"]["averageFitness"])
        model_repair_f_measure.append(v["f_measure"])

    plot_attribute(
        log_path,
        result_file_name,
        "f_measure",
        save_as_file=save,
        x_axis_unit=x_axis_unit,
        model_repair_y=model_repair_f_measure,
    )
    plot_attribute(
        log_path,
        result_file_name,
        "fitness",
        save_as_file=save,
        x_axis_unit=x_axis_unit,
        model_repair_y=model_repair_fitness,
    )
    plot_attribute(
        log_path,
        result_file_name,
        "precision",
        save_as_file=save,
        x_axis_unit=x_axis_unit,
        model_repair_y=model_repair_precision,
    )
