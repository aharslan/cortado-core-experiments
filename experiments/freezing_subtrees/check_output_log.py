import pickle


def get_experiment_output_log(filename: str):
    data2 = []
    infile = open(filename, "rb")
    while 1:
        try:
            data2.append(pickle.load(infile))
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()

    return data2
