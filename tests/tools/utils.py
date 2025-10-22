import networkx as nx

from yamyam_lab.tools.generate_walks import (
    precompute_probabilities_metapath,
)


def test_precompute_probabilities_metapath():
    """
    user node: 0,3
    diner node: 1,2,4
    category node: 5
    """
    graph = nx.Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(2, 3)
    graph.add_edge(2, 5)
    graph.add_edge(4, 5)
    node_metadata = {
        0: {"meta": "user"},
        1: {"meta": "diner"},
        2: {"meta": "diner"},
        3: {"meta": "user"},
        4: {"meta": "diner"},
        5: {"meta": "category"},
    }
    nx.set_node_attributes(graph, node_metadata)
    d_graph = precompute_probabilities_metapath(graph=graph, meta_field="meta")
    assert set(d_graph[0]["diner"]["neighbors"]) == set([1, 2])
    assert set(d_graph[0]["diner"]["prob"]) == set([0.5, 0.5])
    assert set(d_graph[1]["user"]["neighbors"]) == set([0])
    assert set(d_graph[1]["user"]["prob"]) == set([1])
    assert set(d_graph[2]["user"]["neighbors"]) == set([0, 3])
    assert set(d_graph[2]["user"]["prob"]) == set([0.5, 0.5])
    assert set(d_graph[2]["category"]["neighbors"]) == set([5])
    assert set(d_graph[2]["category"]["prob"]) == set([1])
    assert set(d_graph[3]["diner"]["neighbors"]) == set([2])
    assert set(d_graph[3]["diner"]["prob"]) == set([1])
    assert set(d_graph[4]["category"]["neighbors"]) == set([5])
    assert set(d_graph[4]["category"]["prob"]) == set([1])
    assert set(d_graph[5]["diner"]["neighbors"]) == set([2, 4])
    assert set(d_graph[5]["diner"]["prob"]) == set([0.5, 0.5])
