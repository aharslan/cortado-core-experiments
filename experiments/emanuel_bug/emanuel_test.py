from pm4py.objects.log.importer.xes.importer import apply as import_xes
from pm4py.algo.discovery.inductive.algorithm import apply_tree
from pm4py.objects.log.obj import EventLog
from pm4py.visualization.process_tree.visualizer import apply as visualize_pt
from pm4py.visualization.process_tree.visualizer import view as view_pt
from cortado_core.lca_approach import add_trace_to_pt_language


def emanuelTest():
    log = import_xes("debug_road_traffic_ASC.xes")

    im_log = EventLog()
    im_log.append(log[0])
    init_tree = apply_tree(im_log)
    view_pt(visualize_pt(init_tree, parameters={"format": "svg"}))
    print_trace(log[0])

    pt_1 = add_trace_to_pt_language(
        init_tree, im_log, log[1], try_pulling_lca_down=True
    )
    view_pt(visualize_pt(pt_1, parameters={"format": "svg"}))
    im_log.append(log[1])
    print_trace(log[1])

    pt_2 = add_trace_to_pt_language(pt_1, im_log, log[2], try_pulling_lca_down=True)
    view_pt(visualize_pt(pt_2, parameters={"format": "svg"}))
    im_log.append(log[2])
    print_trace(log[2])

    pt_3 = add_trace_to_pt_language(pt_2, im_log, log[3], try_pulling_lca_down=True)
    view_pt(visualize_pt(pt_3, parameters={"format": "svg"}))
    im_log.append(log[3])
    print_trace(log[3])


def print_trace(t):
    res = ""
    for a in t:
        res += a["concept:name"] + ", "
    print(res)


if __name__ == "__main__":
    emanuelTest()
