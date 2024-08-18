import sys
from subprocess import call


def run_all_variations():
    print("Hello!")



    # try:
    #     retcode_1 = call(
    #         "python -m experiments.negative_process_model_repair.negative_repair_experiment" +
    #         r" --log C:\Users\ahmad\OneDrive\Desktop\MS_DS\Cortado_HiWi\Tasks_Artifacts\Processes_Event_Logs\bpi_ch_20\RequestForPayment.xes"
    #         r" --output C:\Users\ahmad\OneDrive\Desktop\Sem6\Thesis\experiments\pickle_cbf_mf_sw_10_20_RequestForPayment_85_5.dat"
    #         r" --total_iterations 20"
    #         r" --iteration_sample_size 10"
    #         r" --iteration_sampling_method shifting_window"
    #         r" --iteration_sampling_method_neg_variant most_frequent"
    #         r" --reduction_approach complete_brute_force"
    #         r" --experiment_identifier 1",
    #         shell=True)
    #     if retcode_1 < 0:
    #         print("Child retcode_1 was terminated by signal", -retcode_1, file=sys.stderr)
    #     else:
    #         print("Child retcode_1 returned", retcode_1, file=sys.stderr)
    # except OSError as e:
    #     print("Execution failed retcode_1 :", e, file=sys.stderr)

    try:
        retcode_2 = call(
            "python -m experiments.negative_process_model_repair.negative_repair_experiment" +
            r" --log C:\Users\ahmad\OneDrive\Desktop\MS_DS\Cortado_HiWi\Tasks_Artifacts\Processes_Event_Logs\bpi_ch_20\RequestForPayment.xes"
            r" --output C:\Users\ahmad\OneDrive\Desktop\Sem6\Thesis\experiments\pickle_cbf_lf_sw_10_20_RequestForPayment_85_5.dat"
            r" --total_iterations 20"
            r" --iteration_sample_size 10"
            r" --iteration_sampling_method shifting_window"
            r" --iteration_sampling_method_neg_variant least_frequent"
            r" --reduction_approach complete_brute_force"
            r" --experiment_identifier 2",
            shell=True)
        if retcode_2 < 0:
            print("Child retcode_2 was terminated by signal", -retcode_2, file=sys.stderr)
        else:
            print("Child retcode_2 returned", retcode_2, file=sys.stderr)
    except OSError as e:
        print("Execution failed retcode_2 :", e, file=sys.stderr)

    # try:
    #     retcode_3 = call(
    #         "python -m experiments.negative_process_model_repair.negative_repair_experiment" +
    #         r" --log C:\Users\ahmad\OneDrive\Desktop\MS_DS\Cortado_HiWi\Tasks_Artifacts\Processes_Event_Logs\bpi_ch_20\RequestForPayment.xes"
    #         r" --output C:\Users\ahmad\OneDrive\Desktop\Sem6\Thesis\experiments\pickle_hbf_mf_sw_10_20_RequestForPayment_85_5.dat"
    #         r" --total_iterations 20"
    #         r" --iteration_sample_size 10"
    #         r" --iteration_sampling_method most_frequent_incremental"
    #         r" --iteration_sampling_method_neg_variant most_frequent"
    #         r" --reduction_approach heuristic_brute_force"
    #         r" --experiment_identifier 3",
    #         shell=True)
    #     if retcode_3 < 0:
    #         print("Child retcode_3 was terminated by signal", -retcode_3, file=sys.stderr)
    #     else:
    #         print("Child retcode_3 returned", retcode_3, file=sys.stderr)
    # except OSError as e:
    #     print("Execution failed retcode_3 :", e, file=sys.stderr)
    #
    # try:
    #     retcode_4 = call(
    #         "python -m experiments.negative_process_model_repair.negative_repair_experiment" +
    #         r" --log C:\Users\ahmad\OneDrive\Desktop\MS_DS\Cortado_HiWi\Tasks_Artifacts\Processes_Event_Logs\bpi_ch_20\RequestForPayment.xes"
    #         r" --output C:\Users\ahmad\OneDrive\Desktop\Sem6\Thesis\experiments\pickle_hbf_lf_sw_10_20_RequestForPayment_85_5.dat"
    #         r" --total_iterations 20"
    #         r" --iteration_sample_size 10"
    #         r" --iteration_sampling_method most_frequent_incremental"
    #         r" --iteration_sampling_method_neg_variant least_frequent"
    #         r" --reduction_approach heuristic_brute_force"
    #         r" --experiment_identifier 4",
    #         shell=True)
    #     if retcode_4 < 0:
    #         print("Child retcode_4 was terminated by signal", -retcode_4, file=sys.stderr)
    #     else:
    #         print("Child retcode_4 returned", retcode_4, file=sys.stderr)
    # except OSError as e:
    #     print("Execution failed retcode_4 :", e, file=sys.stderr)
    #
    # try:
    #     retcode_5 = call(
    #         "python -m experiments.negative_process_model_repair.negative_repair_experiment" +
    #         r" --log C:\Users\ahmad\OneDrive\Desktop\MS_DS\Cortado_HiWi\Tasks_Artifacts\Processes_Event_Logs\bpi_ch_20\RequestForPayment.xes"
    #         r" --output C:\Users\ahmad\OneDrive\Desktop\Sem6\Thesis\experiments\pickle_npmr_mf_sw_10_20_RequestForPayment_85_5.dat"
    #         r" --total_iterations 20"
    #         r" --iteration_sample_size 10"
    #         r" --iteration_sampling_method most_frequent_incremental"
    #         r" --iteration_sampling_method_neg_variant most_frequent"
    #         r" --reduction_approach negative_process_model_repair"
    #         r" --experiment_identifier 5",
    #         shell=True)
    #     if retcode_5 < 0:
    #         print("Child retcode_5 was terminated by signal", -retcode_5, file=sys.stderr)
    #     else:
    #         print("Child retcode_5 returned", retcode_5, file=sys.stderr)
    # except OSError as e:
    #     print("Execution failed retcode_5 :", e, file=sys.stderr)
    #
    # try:
    #     retcode_6 = call(
    #         "python -m experiments.negative_process_model_repair.negative_repair_experiment" +
    #         r" --log C:\Users\ahmad\OneDrive\Desktop\MS_DS\Cortado_HiWi\Tasks_Artifacts\Processes_Event_Logs\bpi_ch_20\RequestForPayment.xes"
    #         r" --output C:\Users\ahmad\OneDrive\Desktop\Sem6\Thesis\experiments\pickle_npmr_lf_sw_10_20_RequestForPayment_85_5.dat"
    #         r" --total_iterations 20"
    #         r" --iteration_sample_size 10"
    #         r" --iteration_sampling_method most_frequent_incremental"
    #         r" --iteration_sampling_method_neg_variant least_frequent"
    #         r" --reduction_approach negative_process_model_repair"
    #         r" --experiment_identifier 6",
    #         shell=True)
    #     if retcode_6 < 0:
    #         print("Child retcode_6 was terminated by signal", -retcode_6, file=sys.stderr)
    #     else:
    #         print("Child retcode_6 returned", retcode_6, file=sys.stderr)
    # except OSError as e:
    #     print("Execution failed retcode_6 :", e, file=sys.stderr)

if __name__ == "__main__":
    run_all_variations()
