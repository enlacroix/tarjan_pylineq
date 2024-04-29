import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def matr2dct(m):
    gr = {}
    num_nodes = m.shape[0]
    for i in range(num_nodes):
        neighbors = set(np.nonzero(m[i])[0])
        gr[i] = neighbors
    return gr


def colorGraph(matrix, scc):
    G = nx.DiGraph(matrix)
    pos = nx.spring_layout(G)
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'violet', 'pink']
    for component, color in zip(scc, colors):
        nx.draw_networkx_nodes(G, pos, nodelist=component, node_color=color, node_size=500)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    # plt.savefig('graph.png')
    plt.show()


def tarjan(graph: dict[int, set[int]]):
    """
    Пример представления графа.
    graph = {
    0: {1},
    1: {3},
    2: {1, 4},
    3: {0},
    4: {2, 5},
    5: set()
    }
    :param graph:
    :return:
    """
    n = len(graph)
    sccs = []
    index = [0]
    indexes = [-1] * n
    lows = [float('Inf')] * n
    S = []

    def strongconnect(v):
        indexes[v] = index[0]
        lows[v] = index[0]
        index[0] += 1
        S.append(v)
        for chld in graph[v]:
            if indexes[chld] == -1:
                strongconnect(chld)
                lows[v] = min(lows[v], lows[chld])
            elif chld in S:
                lows[v] = min(lows[v], lows[chld])

        if lows[v] == indexes[v]:
            scc = [v]
            w = S.pop()
            while w != v:
                if w not in scc:
                    scc.append(w)
                w = S.pop()
            sccs.append(scc)

    for v in graph.keys():
        if indexes[v] == -1:
            strongconnect(v)

    return sccs


def tarjan_algorithm_with_start(graph: dict[int, set[int]], start_node: int):
    """
    И
    :param graph:
    :param start_node:
    :return:
    """

    def dfs(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for neighbor in graph[v]:
            if indices[neighbor] == -1:
                dfs(neighbor)
                lowlink[v] = min(lowlink[v], lowlink[neighbor])
            elif on_stack[neighbor]:
                lowlink[v] = min(lowlink[v], indices[neighbor])

        if indices[v] == lowlink[v]:
            scc = []
            while True:
                node = stack.pop()
                on_stack[node] = False
                scc.append(node)
                if node == v:
                    break
            strongly_connected_components.append(scc)

    index = 0
    stack = []
    indices = [-1] * len(graph)
    lowlink = [-1] * len(graph)
    on_stack = [False] * len(graph)
    strongly_connected_components = []

    dfs(start_node)

    return strongly_connected_components
