import csv
import os
from multiprocessing.pool import Pool

from pm4py.objects.petri_net.importer import importer as petri_importer
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.importer.xes import factory as importer
from experiments.EXPERIMENTS import calculate_f_measure_petri_net
from experiments.EXPERIMENTS import save_obj


def calculate_f_measure_for_pnml_files(
    event_log: EventLog, pnml_file_path: str, log_file_path
):
    log = importer.apply(log_file_path)

    files = []
    for r, d, f in os.walk(pnml_file_path):
        for file in f:
            if ".pnml" in file:
                # files.append(os.path.join(r, file))
                files.append(file)

    processes = {}
    res = {}
    pool = Pool(processes=8)

    # for f in files:
    #     net, marking, fmarking = import_net(os.path.join(pnml_file_path, f))
    #     print("initial marking")
    #     for m in marking:
    #         print(m)
    #     print("final marking")
    #     for m in fmarking:
    #         print(m)
    #     pn_vis_factory.view(pn_vis_factory.apply(net,marking,fmarking))
    #     res = calculate_f_measure_petri_net(net, marking, fmarking, log)
    #     print(f)
    #     print(res)
    for f in files:
        print(f)

        with open(os.path.join(pnml_file_path, f), "r") as file:
            data = file.read().replace("+complete", "")
            # net, marking, fmarking = import_net(os.path.join(pnml_file_path, f))
            net, marking, fmarking = petri_importer.deserialize(data)
            p_net_number = int(f.replace("test", "").replace(".pnml", ""))
            if fmarking:
                processes[p_net_number] = pool.apply_async(
                    calculate_f_measure_petri_net,
                    (net, marking, fmarking, log),
                )
            else:
                res[p_net_number] = None
    pool.close()
    pool.join()
    for p in processes:
        res[p] = processes[p].get()
    keys = list(res.keys())
    keys.sort()
    print(keys)
    output_res = []
    for k in keys:
        output_res.append(res[k])
    print(output_res)
    save_obj(output_res, os.path.join(pnml_file_path, "results"))

    # save results to csv file
    keys = output_res[0].keys()
    filename = "results.csv"
    with open(pnml_file_path + filename, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(output_res)


if __name__ == "__main__":
    file_path = "C:/Users/schuster/Desktop/model_repair_fahrland_results/sorted_minimal_settings"
    calculate_f_measure_for_pnml_files(
        None, file_path, "logs/road_traffic_fine_management/log.xes"
    )
