import pickle
import sys
import os
from tqdm import tqdm

from pm4py.read import read_xes
from pm4py.objects.petri_net.importer.importer import deserialize as deserialize_pn
from pm4py.conformance import fitness_alignments, precision_alignments

model_cache = {}


def evaluate_models_in_folder(folder_path, log):
    pickle_files = [file for file in os.listdir(folder_path) if file.endswith(".p")]
    for file_name in pickle_files:
        print(f"Evalutating {file_name}")
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            if isinstance(data, list):
                for iteration_dict in tqdm(
                    data, desc="Evaluation for each iteration", leave=True
                ):
                    try:
                        pn, im, fm = iteration_dict["output_model"]
                    except:
                        pn, im, fm = deserialize_pn(iteration_dict["output_model"])
                    fitness, precision = evaluate_model(pn, im, fm, log)
                    iteration_dict["fitness"] = fitness
                    iteration_dict["precision"] = precision

                with open(f"{file_name}__evaluated.p", "wb") as file:
                    pickle.dump(data, file)
            else:
                raise ValueError("Wrong data in pickle.")


def evaluate_model(pn, im, fm, log):
    if (pn, im, fm) in model_cache:
        return model_cache[(pn, im, fm)]

    fitness = fitness_alignments(log, pn, im, fm, multi_processing=True)["log_fitness"]
    precision = precision_alignments(log, pn, im, fm, multi_processing=True)

    model_cache[(pn, im, fm)] = fitness, precision
    return fitness, precision


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise TypeError("Input has to be the following 2 args: `logpath model_folder`")
    log_path = sys.argv[1]
    path_head, path_tail = os.path.split(log_path)
    log_name = path_tail.split(".")[-2]

    models_folder = sys.argv[2]

    print(f"Evaluating all models in {models_folder} with {log_name}")

    if log_path.endswith(".xes"):
        log = read_xes(log_path)
        pass
    else:
        raise NotImplementedError

    evaluate_models_in_folder(models_folder, log)
