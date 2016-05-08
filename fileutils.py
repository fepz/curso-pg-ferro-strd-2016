def get_int(string):
    """ convierte string a int """
    try:
        return int(string)
    except ValueError:
        return int(float(string))


def get_rts_from_xmlfile(id, xmlfile):
    import xml.etree.cElementTree as et

    def get_rts_from_element(elem):
        """ extrae el str de elem """

        rts_id, rts = 0, []

        if elem.tag == 'S':
            rts_id = get_int(elem.get("count"))
            for t in elem.iter("i"):
                xtask = t.attrib
                task = {"p":[]}
                for k, v in xtask.items():
                    task[k] = get_int(v)
                for p in t.iter("p"):
                    pv = {}
                    for k, v in p.items():
                        pv[k] = get_int(v)
                    task["p"].append(pv)
                rts.append(task)

        return rts_id, rts

    rts = []

    context = et.iterparse(xmlfile, events=('start', 'end', ))
    context = iter(context)

    event, root = context.__next__()
    #event, root = context.next()  # python 2

    for event, elem in context:
        if event == "end":
            rts_id, rts = get_rts_from_element(elem)
            if rts_id == id and rts and event == "end":
                break
        root.clear()

    del context
    return rts


def get_rts_from_pot_file(pot_file):
    """ return an rts dict and networkx graph """
    from networkx.drawing.nx_pydot import from_pydot, read_dot
    import networkx as nx

    with open(pot_file) as file:
        pot = read_dot(file)
        graph = nx.DiGraph(pot)

        rts = []

        for node, data in graph.nodes(data=True):
            task = {"id":get_int(node)}
            for k, v in data.items():
                task[str(k)] = get_int(v)
            for successor in graph.successors(node):
                task["p"] = []
                for k, v in graph.get_edge_data(node, successor).items():
                    task["p"].append({k:get_int(v),"id":get_int(successor)})
            rts.append(task)

        # sort by id
        rts.sort(key=lambda t: t["id"])

        return (rts, graph)
