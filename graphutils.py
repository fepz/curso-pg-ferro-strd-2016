import networkx as nx
import matplotlib.pyplot as plt


def create_graph_from_rts(rts):
    graph = nx.DiGraph()

    for task in rts:
        graph.add_node(int(task["id"]), cpu="cpu "+ str(task["cpu"]) if "cpu" in task else "")

    for task in rts:
        if task["p"]:
            for p in task["p"]:
                graph.add_edge(task["id"], p["id"], payload=p["payload"])

    return graph


def plot_graph(graph):
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(graph, prog='dot')

    for h in nx.connected_component_subgraphs(graph.to_undirected()):
        nx.draw(h, pos, with_labels=True)

    plt.show()


def save_graph_img(graph, file):
    """ generate a dot file and a image from graph """
    from networkx.drawing.nx_pydot import to_pydot
    for node, data in graph.nodes(data=True):
        if "cpu" in data:
            data["xlabel"] = "cpu "+ str(data["cpu"])
        data["shape"] = "circle"
    P = to_pydot(graph)  #
    for edge in P.get_edges():
        edge.set_label(edge.get_attributes()["payload"])
    P.write_png(file + ".png")


def gen_dot_file(graph):
    """ Writes to file the NetworkX graph """
    from networkx.drawing.nx_pydot import write_dot
    write_dot(graph, 'graph.dot')


