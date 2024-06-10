from __future__ import print_function

import signal
import sys
import threading
from time import sleep
import timeit
import gc
import _thread


# https://gist.github.com/aaronchall/6331661fe0185c30a0b4


def quit_function():
    sys.stderr.flush()
    _thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    """
    use as decorator to exit process if
    function takes longer than s seconds
    """

    def outer(fn):
        def inner(*args, **kwargs):
            signal.signal(signal.SIGINT, signal.default_int_handler)
            timer = threading.Timer(s, quit_function)
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


def run_mining_eval(fnc, timeout, prms, repeats):
    @exit_after(timeout)
    def run_algo(fnc, prms):
        return fnc(**prms)

    rTimes = []

    bail_out = False

    for rep in range(repeats):
        start_time = timeit.default_timer()

        try:
            k_patterns, nC = run_algo(fnc, prms)

        except Exception as e:
            print("Exception caught during Mining", e)

            bail_out = True
            nC = 0
            k_patterns = {}
            rTimes = []
            gc.collect()
            break

        except KeyboardInterrupt as k:
            print("Bailout due to Time", k)
            bail_out = True
            nC = 0
            k_patterns = {}
            rTimes = []
            gc.collect()
            break

        rTimes.append(timeit.default_timer() - start_time)
        gc.collect()

    gc.collect()
    return rTimes, k_patterns, nC, bail_out


def run_mining_memory_eval(fnc, timeout, prms):
    @exit_after(timeout)
    def run_algo(fnc, prms):
        return fnc(**prms)

    rTimes = []

    bail_out = False

    start_time = timeit.default_timer()

    try:
        k_patterns, nC = run_algo(fnc, prms)

    except Exception as e:
        print("Exception caught during Mining", e)

        bail_out = True
        nC = 0
        k_patterns = {}
        rTimes = []

    except KeyboardInterrupt as k:
        print("Bailout due to Time", k)
        bail_out = True
        nC = 0
        k_patterns = {}
        rTimes = []

    rTimes.append(timeit.default_timer() - start_time)

    return rTimes, k_patterns, nC, bail_out
